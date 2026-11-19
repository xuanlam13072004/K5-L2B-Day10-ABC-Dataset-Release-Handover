#!/usr/bin/env python3
"""Detection rules for Day 10 dataset release validator.

Implements the three detection rules assigned to Person 2:
- invalid_class
- bbox_out_of_bounds
- duplicate_box

Public interface: check(L) -> list[dict]. Only reads the delivery data
already loaded by validator/validate.py and files referenced by its manifest.
"""

from __future__ import annotations

import math
from pathlib import Path

# Dung sai 0,5 pixel: toạ độ YOLO được làm tròn 6 chữ số, box sát mép tự nhiên có thể
# vượt mép một phần nghìn pixel. Không có dung sai thì lô sạch bị báo 11 lỗi giả.
TOL_PX = 0.5


def _image_dimensions(L: dict) -> dict[str, tuple[int, int]]:
    """Return {image_stem: (width, height)} from panoptic.json images."""
    dimensions: dict[str, tuple[int, int]] = {}
    for image in L.get("pan", {}).get("images", []):
        filename = str(image.get("file_name", ""))
        stem = Path(filename).stem
        try:
            width, height = int(image["width"]), int(image["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if width > 0 and height > 0:
            dimensions[stem] = (width, height)
    return dimensions


def _parse_yolo(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read valid-shaped YOLO rows; malformed rows are ignored by these rules."""
    rows: list[tuple[int, float, float, float, float]] = []
    if not label_path.is_file():
        return rows

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            xc, yc, width, height = map(float, parts[1:])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (xc, yc, width, height)):
            continue
        rows.append((class_id, xc, yc, width, height))
    return rows


def _check_one_image(name: str, item: dict, dimensions: dict[str, tuple[int, int]]) -> list[dict]:
    errors: list[dict] = []
    label_rel = item.get("label")
    if not label_rel:
        return errors

    rows = _parse_yolo(Path(item.get("_delivery_dir", ".")) / label_rel)
    if not rows:
        return errors

    # The panoptic image file name and the manifest image file name share the
    # same stem (e.g. img_0001.jpg); use metadata dimensions, not hardcoded sizes.
    image_rel = str(item.get("image", ""))
    image_stem = Path(image_rel).stem
    size = dimensions.get(image_stem)
    if size is None:
        # No reliable dimensions means we cannot make a bounds judgment.
        return errors
    image_width, image_height = size

    for class_id, xc, yc, box_width, box_height in rows:
        if not 0 <= class_id <= 79:
            errors.append({"type": "invalid_class", "target": name})
            break

    for _, xc, yc, box_width, box_height in rows:
        # YOLO coordinates are normalized center-x, center-y, width, height.
        x1 = (xc - box_width / 2.0) * image_width
        y1 = (yc - box_height / 2.0) * image_height
        x2 = (xc + box_width / 2.0) * image_width
        y2 = (yc + box_height / 2.0) * image_height
        if (box_width <= 0 or box_height <= 0 or
                x1 < -TOL_PX or y1 < -TOL_PX or x2 > image_width + TOL_PX or y2 > image_height + TOL_PX):
            errors.append({"type": "bbox_out_of_bounds", "target": name})
            break

    # A duplicate is the same class and same box size, with its center shifted
    # by only 1-2 pixels. IoU alone is deliberately not used: natural COCO boxes
    # can overlap substantially while representing different objects.
    for i, first in enumerate(rows):
        class_a, xa, ya, wa, ha = first
        for class_b, xb, yb, wb, hb in rows[i + 1:]:
            if class_a != class_b:
                continue
            same_size = (abs(wa - wb) * image_width <= 0.5 and
                         abs(ha - hb) * image_height <= 0.5)
            center_dx = abs(xa - xb) * image_width
            center_dy = abs(ya - yb) * image_height
            shifted_1_to_2_px = (
                center_dx <= 2.0 and center_dy <= 2.0 and
                0.5 <= max(center_dx, center_dy) <= 2.0
            )
            if same_size and shifted_1_to_2_px:
                errors.append({"type": "duplicate_box", "target": name})
                break
        else:
            continue
        break

    return errors


def check(L: dict) -> list[dict]:
    """Run all detection rules and return errors in validate.py's format."""
    dimensions = _image_dimensions(L)
    delivery_dir = Path(L["dir"])
    errors: list[dict] = []

    for name, item in sorted(L.get("man", {}).items()):
        # Avoid mutating the shared manifest entries in L.
        local_item = dict(item)
        local_item["_delivery_dir"] = str(delivery_dir)
        errors.extend(_check_one_image(name, local_item, dimensions))

    return errors
