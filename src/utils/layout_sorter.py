import json
from typing import List, Dict, Any

def get_box_bounds(box):
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    return min(xs), max(xs), min(ys), max(ys)

def sort_reading_order(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort OCR lines in natural human reading order:
    - Detects multi-column / 2-page layouts.
    - If 2 columns detected: reads Top Headers -> Column 1 (top-to-bottom) -> Column 2 (top-to-bottom) -> Footers.
    - If 1 column: reads top-to-bottom with horizontal grouping.
    """
    if len(lines) <= 2:
        return sorted(lines, key=lambda item: get_box_bounds(item["box"])[2])

    # 1. Compute bounds
    items_with_bounds = []
    max_x = 0
    max_y = 0
    for item in lines:
        xmin, xmax, ymin, ymax = get_box_bounds(item["box"])
        max_x = max(max_x, xmax)
        max_y = max(max_y, ymax)
        items_with_bounds.append({
            "item": item,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "width": xmax - xmin,
            "height": ymax - ymin,
            "xcenter": (xmin + xmax) / 2,
            "ycenter": (ymin + ymax) / 2,
        })

    page_width = max_x

    # 2. Check for vertical gutter / column split in candidate range [0.30 * W, 0.70 * W]
    # We test potential split points X from 0.35W to 0.65W
    best_split_x = None
    min_crossing = 999999
    
    # Sample 30 candidate split lines across middle region
    step = (0.65 * page_width - 0.35 * page_width) / 30.0
    candidates = [0.35 * page_width + i * step for i in range(30)]

    for split_x in candidates:
        left_count = 0
        right_count = 0
        crossing_count = 0

        for b in items_with_bounds:
            # If box crosses split_x significantly (more than 10% of width on each side)
            if b["xmin"] < split_x - 10 and b["xmax"] > split_x + 10:
                crossing_count += 1
            elif b["xmax"] <= split_x + 10:
                left_count += 1
            else:
                right_count += 1

        # A valid 2-column split must have substantial lines on BOTH left and right
        # and very few crossing lines (< 15% of total lines)
        if left_count >= 5 and right_count >= 5:
            if crossing_count < min_crossing:
                min_crossing = crossing_count
                best_split_x = split_x

    # If we found a clean column separator
    is_two_column = False
    if best_split_x is not None and min_crossing <= max(3, len(lines) * 0.15):
        is_two_column = True

    if not is_two_column:
        # Standard 1-column reading order: line-group by Y, then sort by X
        # Group lines with vertical overlap
        return [b["item"] for b in _sort_single_column(items_with_bounds)]


    # For 2-column layout:
    # Separate into:
    # 1. Header lines (full width or above both columns)
    # 2. Left column
    # 3. Right column
    # 4. Footer lines
    headers = []
    left_col = []
    right_col = []
    footers = []

    for b in items_with_bounds:
        # Crossing line (spans across the gutter)
        if b["xmin"] < best_split_x - 15 and b["xmax"] > best_split_x + 15:
            if b["ycenter"] < max_y * 0.25:
                headers.append(b)
            elif b["ycenter"] > max_y * 0.85:
                footers.append(b)
            else:
                # If in middle, assign to column where majority of box lies
                if (best_split_x - b["xmin"]) > (b["xmax"] - best_split_x):
                    left_col.append(b)
                else:
                    right_col.append(b)
        elif b["xmax"] <= best_split_x + 15:
            left_col.append(b)
        else:
            right_col.append(b)

    sorted_headers = _sort_single_column(headers)
    sorted_left = _sort_single_column(left_col)
    sorted_right = _sort_single_column(right_col)
    sorted_footers = _sort_single_column(footers)

    return [b["item"] for b in (sorted_headers + sorted_left + sorted_right + sorted_footers)]


def _sort_single_column(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort a single column top-to-bottom, grouping boxes that share the same horizontal line."""
    if not items:
        return []

    # Sort primarily by ymin
    sorted_by_y = sorted(items, key=lambda b: (b["ymin"], b["xmin"]))

    # Group into lines where boxes overlap vertically by at least 40% of height
    lines_grouped = []
    current_line = [sorted_by_y[0]]

    for b in sorted_by_y[1:]:
        # Compare with reference box in current line
        ref = current_line[-1]
        line_ymin = min(x["ymin"] for x in current_line)
        line_ymax = max(x["ymax"] for x in current_line)
        overlap = min(line_ymax, b["ymax"]) - max(line_ymin, b["ymin"])
        min_h = min(b["height"], ref["height"])

        if min_h > 0 and overlap / min_h > 0.35:
            current_line.append(b)
        else:
            # Sort current line horizontally (left-to-right)
            lines_grouped.extend(sorted(current_line, key=lambda x: x["xmin"]))
            current_line = [b]

    if current_line:
        lines_grouped.extend(sorted(current_line, key=lambda x: x["xmin"]))

    return lines_grouped
