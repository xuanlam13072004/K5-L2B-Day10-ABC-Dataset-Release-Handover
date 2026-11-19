#!/usr/bin/env python3
"""Panoptic rules for Day 10 dataset release validator.

Implements the two panoptic rules assigned to Person 2:
- invalid_category_id
- thing_overlap

Public interface: check(L) -> list[dict]. Uses only the Python standard
library plus scripts/pngutil.py, as allowed by the lab rules.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# validate.py adds the repository's scripts/ directory to sys.path before
# importing this module. The fallback keeps this module directly testable.
try:
    import pngutil
except ModuleNotFoundError:
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import pngutil


def _category_ids(panoptic: dict) -> set[int]:
    ids: set[int] = set()
    for category in panoptic.get("categories", []):
        try:
            ids.add(int(category["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ids


def _image_annotation_map(L: dict) -> dict[str, dict]:
    """Map manifest image name -> panoptic annotation via mask file name."""
    annotations = L.get("pan", {}).get("annotations", [])
    by_mask_name = {
        Path(str(annotation.get("file_name", ""))).name: annotation
        for annotation in annotations
        if annotation.get("file_name")
    }
    result: dict[str, dict] = {}
    for name, item in L.get("man", {}).items():
        mask_rel = item.get("mask")
        if mask_rel:
            annotation = by_mask_name.get(Path(str(mask_rel)).name)
            if annotation is not None:
                result[name] = annotation
    return result


def _read_segment_pixel_counts(mask_path: Path) -> Counter:
    """Count pixels for each encoded panoptic segment id; ignore background 0."""
    _width, _height, pixels = pngutil.read_rgb(mask_path)
    counts: Counter = Counter()
    for offset in range(0, len(pixels), 3):
        segment_id = pngutil.segment_id(pixels, offset)
        if segment_id != 0:
            counts[segment_id] += 1
    return counts


def check(L: dict) -> list[dict]:
    """Return invalid-category and over-declared thing-area findings."""
    errors: list[dict] = []
    panoptic = L.get("pan", {})
    valid_category_ids = _category_ids(panoptic)
    categories_by_id = {}
    for category in panoptic.get("categories", []):
        try:
            categories_by_id[int(category["id"])] = category
        except (KeyError, TypeError, ValueError):
            continue

    annotations_by_image = _image_annotation_map(L)
    delivery_dir = Path(L["dir"])

    for name, item in sorted(L.get("man", {}).items()):
        annotation = annotations_by_image.get(name)
        if annotation is None:
            continue

        segments = annotation.get("segments_info", [])
        invalid_category = False
        for segment in segments:
            try:
                category_id = int(segment["category_id"])
            except (KeyError, TypeError, ValueError):
                invalid_category = True
                continue
            if category_id not in valid_category_ids:
                invalid_category = True
        if invalid_category:
            errors.append({"type": "invalid_category_id", "target": name})

        mask_rel = item.get("mask")
        if not mask_rel:
            continue
        mask_path = delivery_dir / str(mask_rel)
        if not mask_path.is_file():
            continue

        # The injected thing_overlap fault increases a thing segment's declared
        # area by transferring pixels from another thing. Stuff/crowd segments
        # are not the target of this rule, so only inspect non-crowd isthing IDs.
        thing_segment_ids: set[int] = set()
        for segment in segments:
            try:
                category_id = int(segment["category_id"])
                segment_id = int(segment["id"])
            except (KeyError, TypeError, ValueError):
                continue
            category = categories_by_id.get(category_id)
            if category and category.get("isthing") and not segment.get("iscrowd"):
                thing_segment_ids.add(segment_id)

        if not thing_segment_ids:
            continue

        pixel_counts = _read_segment_pixel_counts(mask_path)
        overlap_found = False
        for segment in segments:
            try:
                category_id = int(segment["category_id"])
                segment_id = int(segment["id"])
                declared_area = int(segment["area"])
            except (KeyError, TypeError, ValueError):
                continue
            if segment_id not in thing_segment_ids:
                continue
            actual_area = pixel_counts.get(segment_id, 0)
            if declared_area > actual_area:
                overlap_found = True
                break

        if overlap_found:
            errors.append({"type": "thing_overlap", "target": name})

    return errors
