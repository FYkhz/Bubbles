from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from omr import (
    MARKER_MARGIN,
    MARKER_SIZE,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    MAX_QUESTIONS,
    bubble_centers,
    bubble_radius_for_layout,
    choices_for_count,
    layout_columns,
    layout_rows_per_column,
    question_number_position,
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def create_sheet(
    answer_marks: Sequence[str | None] | None = None,
    *,
    title: str = "OMR ANSWER SHEET",
    choice_count: int = 4,
    question_count: int = 20,
) -> np.ndarray:
    question_count = int(question_count)
    if not 1 <= question_count <= MAX_QUESTIONS:
        raise ValueError(f"Question count must be between 1 and {MAX_QUESTIONS}.")

    choices = choices_for_count(choice_count)
    rows = bubble_centers(choice_count=choice_count, question_count=question_count)
    radius = bubble_radius_for_layout(choice_count=choice_count, question_count=question_count)

    if answer_marks is not None and len(answer_marks) != question_count:
        raise ValueError(f"Expected {question_count} answers.")

    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    marker_positions = [
        (MARKER_MARGIN, MARKER_MARGIN),
        (PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE, MARKER_MARGIN),
        (PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE),
        (MARKER_MARGIN, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE),
    ]
    for x, y in marker_positions:
        draw.rectangle((x, y, x + MARKER_SIZE, y + MARKER_SIZE), fill="black")

    safe_title = (title or "OMR ANSWER SHEET").strip()[:60]
    draw.text((PAGE_WIDTH // 2, 115), safe_title, font=_font(52, bold=True), fill="black", anchor="ma")
    draw.text(
        (PAGE_WIDTH // 2, 176),
        "Fill one bubble completely. Keep all four corner squares visible when photographing.",
        font=_font(22), fill="black", anchor="ma",
    )
    draw.line((180, 235, 1520, 235), fill="black", width=3)
    draw.text((205, 280), "Name: __________________________________________", font=_font(27), fill="black")

    columns = layout_columns(question_count)
    rows_per_col = layout_rows_per_column(question_count)

    # Choice labels above each column, aligned to that column's first row.
    for col in range(columns):
        q_index = col * rows_per_col
        if q_index >= question_count:
            continue
        first_row = rows[q_index]
        number_x, _ = question_number_position(q_index + 1, choice_count=choice_count, question_count=question_count)
        header_y = 390
        draw.text((number_x, header_y), "Q", font=_font(22, bold=True), fill="black", anchor="mm")
        for (x, _), choice in zip(first_row, choices):
            draw.text((x, header_y), choice, font=_font(22, bold=True), fill="black", anchor="mm")

    q_font = _font(22 if columns <= 2 else 18, bold=True)
    for q_index, row in enumerate(rows):
        q = q_index + 1
        number_x, y = question_number_position(q, choice_count=choice_count, question_count=question_count)
        draw.text((number_x, y), str(q), font=q_font, fill="black", anchor="mm")
        for choice_index, (x, cy) in enumerate(row):
            box = (x - radius, cy - radius, x + radius, cy + radius)
            draw.ellipse(box, outline="black", width=max(2, radius // 7))
            if answer_marks is not None and answer_marks[q_index] == choices[choice_index]:
                inner = max(5, radius - max(5, radius // 4))
                draw.ellipse((x-inner, cy-inner, x+inner, cy+inner), fill="black")

    footer_y = 2025
    draw.line((180, footer_y, 1520, footer_y), fill="black", width=3)
    draw.text(
        (PAGE_WIDTH // 2, footer_y + 43),
        f"{question_count} questions · choices {choices[0]}-{choices[-1]} · {columns} answer block{'s' if columns != 1 else ''}",
        font=_font(21), fill="black", anchor="ma",
    )

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def save_sheet(path: str | Path, answer_marks: Sequence[str | None] | None = None, *, title: str = "OMR ANSWER SHEET", choice_count: int = 4, question_count: int = 20) -> None:
    cv2.imwrite(str(path), create_sheet(answer_marks, title=title, choice_count=choice_count, question_count=question_count))


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "assets" / "omr_answer_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_sheet(output)
    print(output)
