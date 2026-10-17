from __future__ import annotations

import hashlib
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st

from omr import (
    MAX_QUESTIONS,
    OMRProcessingError,
    answers_from_detections,
    choices_for_count,
    crop_question,
    decode_image_bytes,
    detection_rows,
    draw_overlay,
    encode_png,
    grade_answers,
    scan_sheet,
)
from sheet_template import create_sheet


st.set_page_config(page_title="OMR Grader", page_icon="✅", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 3rem;}
.omr-hero {padding: 1.15rem 1.35rem; border: 1px solid #dbe5f4; border-radius: 18px; background: linear-gradient(135deg,#f7fbff,#eef5ff); margin-bottom: 1rem;}
.omr-hero h1 {margin: 0; font-size: 2.2rem; color: #0b3975;}
.omr-hero p {margin: .35rem 0 0; color: #40536d;}
[data-testid="stMetric"] {border: 1px solid #dfe6ef; padding: .75rem; border-radius: 12px; background: white;}
.mode-card {padding: .85rem 1rem; border: 1px solid #d9e5f4; border-radius: 14px; background: #f8fbff;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="omr-hero">
  <h1>OMR Grader</h1>
  <p>Create flexible answer sheets, set the exam mode from the answer key, then grade every student sheet using that same configuration.</p>
</div>
""",
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "key_scan": None,
        "key_hash": None,
        "key_overrides": {},
        "key_answers": None,
        "key_confirmed_at": None,
        "exam_question_count": 20,
        "exam_choice_count": 4,
        "saved_exam_question_count": None,
        "saved_exam_choice_count": None,
        "exam_name": "",
        "student_scan": None,
        "student_hash": None,
        "student_overrides": {},
        "history": [],
        "sheet_question_count": 20,
        "sheet_choice_count": 4,
        "sheet_title": "OMR ANSWER SHEET",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


def source_picker(prefix: str, label: str):
    upload_tab, camera_tab = st.tabs(["Upload image", "Scan with camera"])
    with upload_tab:
        upload = st.file_uploader(label, type=["jpg", "jpeg", "png"], key=f"{prefix}_upload")
        if upload is not None:
            return upload, False
    with camera_tab:
        st.caption("Keep the full page and all four black corner squares visible.")
        camera = st.camera_input("Capture the full sheet", key=f"{prefix}_camera")
        if camera is not None:
            return camera, True
    return None, False


def file_bytes(uploaded) -> bytes:
    uploaded.seek(0)
    return uploaded.read()


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def problem_detections(scan):
    return [d for d in scan.detections if d.status != "clear"]


def clear_review_widget_state(state_key: str, question_count: int) -> None:
    for question in range(1, int(question_count) + 1):
        st.session_state.pop(f"{state_key}_{question}", None)
        st.session_state.pop(f"{state_key}_clear_{question}", None)


def reset_exam_scans() -> None:
    st.session_state.key_scan = None
    st.session_state.key_hash = None
    st.session_state.key_overrides = {}
    st.session_state.key_answers = None
    st.session_state.key_confirmed_at = None
    st.session_state.saved_exam_question_count = None
    st.session_state.saved_exam_choice_count = None
    st.session_state.student_scan = None
    st.session_state.student_hash = None
    st.session_state.student_overrides = {}


def edit_clear_answers(scan, *, state_key: str, choice_count: int) -> dict[int, str | None]:
    overrides = dict(st.session_state[state_key])
    clear_items = [d for d in scan.detections if d.status == "clear"]
    if not clear_items:
        return overrides

    with st.expander("Optional: correct a clearly detected answer"):
        st.caption("Use this only when the overlay shows that the automatic choice is wrong.")
        columns = st.columns(4)
        options = list(choices_for_count(choice_count))
        for index, detection in enumerate(clear_items):
            q = detection.question
            current = overrides.get(q, detection.selected)
            selected = columns[index % 4].selectbox(
                f"Question {q}",
                options,
                index=options.index(current) if current in options else 0,
                key=f"{state_key}_clear_{q}",
            )
            if selected == detection.selected:
                overrides.pop(q, None)
            else:
                overrides[q] = selected

    st.session_state[state_key] = overrides
    return overrides


def review_controls(scan, *, state_key: str, include_blank: bool, choice_count: int) -> dict[int, str | None]:
    overrides = dict(st.session_state[state_key])
    problems = problem_detections(scan)
    if not problems:
        st.success(f"All {len(scan.detections)} questions were detected clearly.")
        return overrides

    st.warning(f"{len(problems)} question(s) need review before continuing.")
    active_choices = choices_for_count(choice_count)
    valid_choices = list(active_choices) + (["Blank"] if include_blank else [])
    radio_choices = ["Select..."] + valid_choices

    for detection in problems:
        q = detection.question
        with st.expander(f"Question {q}: {detection.status.title()} — confidence {detection.confidence:.0f}%", expanded=True):
            crop = crop_question(scan.aligned_bgr, q, bubble_rows=scan.bubble_rows)
            st.image(bgr_to_rgb(crop), use_container_width=True)
            density_text = " · ".join(
                f"{choice}: {density * 100:.1f}%" for choice, density in zip(active_choices, detection.densities)
            )
            st.caption(f"Inner-bubble ink density — {density_text}")
            default_value = overrides.get(q)
            default_index = radio_choices.index(default_value) if default_value in valid_choices else 0
            selected = st.radio(
                f"Choose the intended answer for Question {q}",
                radio_choices,
                index=default_index,
                horizontal=True,
                key=f"{state_key}_{q}",
            )
            if selected == "Select...":
                overrides.pop(q, None)
            else:
                overrides[q] = selected

    st.session_state[state_key] = overrides
    return overrides


with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["1. Sheet Maker", "2. Answer Key", "3. Grader", "4. Results"],
        label_visibility="collapsed",
    )

    st.divider()
    st.header("Active exam")
    if st.session_state.key_answers:
        qn = int(st.session_state.saved_exam_question_count)
        cc = int(st.session_state.saved_exam_choice_count)
        labels = choices_for_count(cc)
        st.success("Answer key ready")
        st.caption(f"{st.session_state.exam_name or 'Current exam'}")
        st.caption(f"{qn} questions · choices {labels[0]}-{labels[-1]}")
        st.caption(f"Saved {st.session_state.key_confirmed_at}")
    else:
        st.info("No answer key saved yet")
        st.caption("Set the question count on the Answer Key tab. That becomes the grading mode.")


if page == "1. Sheet Maker":
    st.subheader("Create a printable OMR sheet")
    st.write("Choose the sheet size you need. The layout automatically changes to fit the selected number of questions.")

    c1, c2, c3 = st.columns([1, 1, 1.5])
    with c1:
        sheet_questions = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=MAX_QUESTIONS,
            value=int(st.session_state.sheet_question_count),
            step=1,
            key="sheet_question_input",
        )
    with c2:
        sheet_choices = st.selectbox(
            "Choices per question",
            options=list(range(2, 9)),
            index=list(range(2, 9)).index(int(st.session_state.sheet_choice_count)),
            format_func=lambda n: f"{n} choices (A-{choices_for_count(n)[-1]})",
            key="sheet_choice_input",
        )
    with c3:
        sheet_title = st.text_input("Header / exam title", value=st.session_state.sheet_title)

    st.session_state.sheet_question_count = int(sheet_questions)
    st.session_state.sheet_choice_count = int(sheet_choices)
    st.session_state.sheet_title = sheet_title

    template = create_sheet(
        title=sheet_title,
        choice_count=int(sheet_choices),
        question_count=int(sheet_questions),
    )
    st.image(
        bgr_to_rgb(template),
        caption=f"Printable {int(sheet_questions)}-question OMR sheet",
        use_container_width=True,
    )
    st.download_button(
        "Download printable answer sheet (PNG)",
        data=encode_png(template),
        file_name=f"omr_answer_sheet_{int(sheet_questions)}q_{int(sheet_choices)}choices.png",
        mime="image/png",
        use_container_width=True,
    )
    st.info("Use the same sheet format for the teacher's answer key and all student papers for that exam.")


elif page == "2. Answer Key":
    st.subheader("Set exam mode and create the answer key")
    st.write("Tell the app how many questions are in this exam before uploading the answer key. This becomes the scanner mode for the exam.")

    c1, c2, c3 = st.columns([1, 1, 1.5])
    with c1:
        q_count = st.number_input(
            "How many questions are in this exam?",
            min_value=1,
            max_value=MAX_QUESTIONS,
            value=int(st.session_state.exam_question_count),
            step=1,
        )
    with c2:
        c_count = st.selectbox(
            "Choices per question",
            options=list(range(2, 9)),
            index=list(range(2, 9)).index(int(st.session_state.exam_choice_count)),
            format_func=lambda n: f"A-{choices_for_count(n)[-1]}",
        )
    with c3:
        exam_name = st.text_input("Exam name (optional)", value=st.session_state.exam_name)

    q_count, c_count = int(q_count), int(c_count)
    if q_count != int(st.session_state.exam_question_count) or c_count != int(st.session_state.exam_choice_count):
        reset_exam_scans()
        st.session_state.exam_question_count = q_count
        st.session_state.exam_choice_count = c_count
    st.session_state.exam_name = exam_name

    labels = choices_for_count(c_count)
    st.markdown(
        f"<div class='mode-card'><b>Scanner mode:</b> {q_count} questions · choices {labels[0]}-{labels[-1]}<br>"
        f"The OMR will build and analyze exactly {q_count} question positions.</div>",
        unsafe_allow_html=True,
    )

    source, from_camera = source_picker("key", "Upload the completed answer-key sheet")

    if source is not None:
        data = file_bytes(source)
        digest = image_hash(data) + f"_{q_count}_{c_count}"
        if digest != st.session_state.key_hash:
            try:
                with st.spinner(f"Reading the answer key in {q_count}-question mode..."):
                    image = decode_image_bytes(data)
                    st.session_state.key_scan = scan_sheet(
                        image,
                        answer_key_mode=True,
                        scanner_mode=from_camera,
                        choice_count=c_count,
                        question_count=q_count,
                    )
                    st.session_state.key_hash = digest
                    st.session_state.key_overrides = {}
                    st.session_state.key_answers = None
                    st.session_state.key_confirmed_at = None
                    st.session_state.student_scan = None
                    st.session_state.student_hash = None
                    st.session_state.student_overrides = {}
                    clear_review_widget_state("key_overrides", q_count)
                    clear_review_widget_state("student_overrides", q_count)
            except (OMRProcessingError, ValueError) as exc:
                st.error(str(exc))

    scan = st.session_state.key_scan
    if scan is not None:
        if scan.warning:
            st.warning(scan.warning)
        if scan.grid_refinement_used:
            st.success(f"Bubble-grid alignment active · confidence {scan.grid_confidence:.0%}")
        else:
            st.caption(f"Bubble-grid alignment fallback · confidence {scan.grid_confidence:.0%}")

        c1, c2 = st.columns([1.2, 0.8], gap="large")
        with c1:
            overlay = draw_overlay(scan.aligned_bgr, scan.detections, overrides=st.session_state.key_overrides, bubble_rows=scan.bubble_rows)
            st.image(bgr_to_rgb(overlay), caption=f"Detected {q_count}-question answer key", use_container_width=True)
        with c2:
            rows = pd.DataFrame(detection_rows(scan.detections, choices=scan.choices))
            clear_count = int((rows["Status"] == "Clear").sum())
            m1, m2 = st.columns(2)
            m1.metric("Clear", clear_count)
            m2.metric("Needs review", q_count - clear_count)
            st.dataframe(rows[["Question", "Detected", "Status", "Confidence"]], hide_index=True, use_container_width=True)

        st.divider()
        overrides = review_controls(scan, state_key="key_overrides", include_blank=False, choice_count=c_count)
        overrides = edit_clear_answers(scan, state_key="key_overrides", choice_count=c_count)
        answers = answers_from_detections(scan.detections, overrides)
        unresolved = [i + 1 for i, answer in enumerate(answers) if answer is None]

        if unresolved:
            st.error(f"Confirm every answer before saving. {len(unresolved)} question(s) are still unresolved.")
        else:
            if st.button("Confirm and save answer key", type="primary", use_container_width=True):
                st.session_state.key_answers = [str(answer) for answer in answers]
                st.session_state.saved_exam_question_count = q_count
                st.session_state.saved_exam_choice_count = c_count
                st.session_state.key_confirmed_at = datetime.now().strftime("%Y-%m-%d %I:%M %p")
                st.success(f"Answer key saved. Grader is now locked to {q_count}-question mode.")
                st.rerun()


elif page == "3. Grader":
    st.subheader("Grade student papers")
    if not st.session_state.key_answers:
        st.warning("Create and save an answer key first. The answer key sets the grading mode.")
        st.stop()

    q_count = int(st.session_state.saved_exam_question_count)
    c_count = int(st.session_state.saved_exam_choice_count)
    labels = choices_for_count(c_count)
    st.markdown(
        f"<div class='mode-card'><b>Active grading mode:</b> {q_count} questions · choices {labels[0]}-{labels[-1]}<br>"
        "Every student sheet uploaded here will use this exact configuration.</div>",
        unsafe_allow_html=True,
    )

    source, from_camera = source_picker("student", "Upload the student's answer sheet")

    if source is not None:
        data = file_bytes(source)
        digest = image_hash(data) + f"_{q_count}_{c_count}"
        if digest != st.session_state.student_hash:
            try:
                with st.spinner(f"Reading student answers in {q_count}-question mode..."):
                    image = decode_image_bytes(data)
                    st.session_state.student_scan = scan_sheet(
                        image,
                        answer_key_mode=False,
                        scanner_mode=from_camera,
                        choice_count=c_count,
                        question_count=q_count,
                    )
                    st.session_state.student_hash = digest
                    st.session_state.student_overrides = {}
                    clear_review_widget_state("student_overrides", q_count)
            except (OMRProcessingError, ValueError) as exc:
                st.error(str(exc))

    scan = st.session_state.student_scan
    if scan is not None:
        if scan.warning:
            st.warning(scan.warning)
        if scan.grid_refinement_used:
            st.success(f"Bubble-grid alignment active · confidence {scan.grid_confidence:.0%}")

        initial_answers = answers_from_detections(scan.detections, st.session_state.student_overrides)
        initial_grade = grade_answers(initial_answers, st.session_state.key_answers)
        overlay = draw_overlay(
            scan.aligned_bgr,
            scan.detections,
            key_answers=st.session_state.key_answers,
            overrides=st.session_state.student_overrides,
            bubble_rows=scan.bubble_rows,
        )

        left, right = st.columns([1.2, 0.8], gap="large")
        with left:
            st.image(bgr_to_rgb(overlay), caption="Green = correct · Red = incorrect · Orange = review · Gray = blank", use_container_width=True)
        with right:
            r1, r2 = st.columns(2)
            r1.metric("Current score", f"{initial_grade['score']}/{initial_grade['total']}")
            r2.metric("Percentage", f"{initial_grade['percentage']:.1f}%")
            r3, r4 = st.columns(2)
            r3.metric("Incorrect", initial_grade["incorrect"])
            r4.metric("Blank", initial_grade["blank"])
            st.dataframe(pd.DataFrame(initial_grade["rows"]), hide_index=True, use_container_width=True)

        st.divider()
        overrides = review_controls(scan, state_key="student_overrides", include_blank=True, choice_count=c_count)
        final_answers = answers_from_detections(scan.detections, overrides)
        unresolved = [d.question for d in scan.detections if d.status != "clear" and d.question not in overrides]

        if unresolved:
            st.warning("Review all orange questions to finalize the grade.")
        else:
            final_grade = grade_answers(final_answers, st.session_state.key_answers)
            final_overlay = draw_overlay(
                scan.aligned_bgr,
                scan.detections,
                key_answers=st.session_state.key_answers,
                overrides=overrides,
                bubble_rows=scan.bubble_rows,
            )
            st.success(f"Final score: {final_grade['score']}/{final_grade['total']} ({final_grade['percentage']:.1f}%)")

            result_df = pd.DataFrame(final_grade["rows"])
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "Download question results (CSV)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="omr_question_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "Download marked image (PNG)",
                    data=encode_png(final_overlay),
                    file_name="omr_graded_sheet.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with c3:
                if st.button("Add result to session", use_container_width=True):
                    st.session_state.history.append(
                        {
                            "Scan": len(st.session_state.history) + 1,
                            "Time": datetime.now().strftime("%I:%M:%S %p"),
                            "Questions": q_count,
                            "Score": final_grade["score"],
                            "Total": final_grade["total"],
                            "Percentage": round(final_grade["percentage"], 1),
                            "Incorrect": final_grade["incorrect"],
                            "Blank": final_grade["blank"],
                        }
                    )
                    st.success("Result added to this session.")


elif page == "4. Results":
    st.subheader("Session results")
    if not st.session_state.history:
        st.info("No student results have been added during this session yet.")
    else:
        history = pd.DataFrame(st.session_state.history)
        st.dataframe(history, hide_index=True, use_container_width=True)
        st.download_button(
            "Download session results (CSV)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="omr_session_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Clear session results"):
            st.session_state.history = []
            st.rerun()

st.divider()
st.caption("Dynamic OMR prototype · 1-100 questions · A-H choices · answer-key-driven grading modes")
