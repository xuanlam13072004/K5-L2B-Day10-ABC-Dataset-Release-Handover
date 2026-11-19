# Doi chieu Detection-Panoptic qua IoU, mask_in_box va loc bo skip_images
"""Đối chiếu nhãn Detection (YOLO) và Panoptic mask để sinh candidates.json.

Chỉ sử dụng thư viện chuẩn của Python và scripts/pngutil.py.
Tuyệt đối không đọc _private/ hoặc data_source/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Đảm bảo import được scripts/pngutil.py dù chạy từ thư mục nào
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pngutil import read_rgb, segment_id  # noqa: E402

# ==============================================================================
# CÁC NGƯỠNG LỌC ỨNG VIÊN (Tinh chỉnh dựa trên runs/clean/delivery)
# ==============================================================================
# 1. Lệch lớp (class_swap / category_swap)
SWAP_MASK_IN_BOX_MIN = 0.70   # Tối thiểu 70% pixel của mask nằm trong box detection
SWAP_IOU_MIN = 0.55           # IoU giữa box detection và bbox của mask >= 0.55

# 2. Lệch khung bao (bbox_drift)
DRIFT_IOU_MIN = 0.25          # Vẫn phải là cùng 1 vật thể (IoU >= 0.25)
DRIFT_MASK_IN_BOX_MAX = 0.80  # Nhưng có >20% pixel mask bị tràn ra ngoài box
DRIFT_IOU_MAX = 0.62          # Hoặc khung box bị kéo giãn/lệch hẳn so với mask

# 3. Gộp vật thể (instance_merge)
MERGE_BOX_FILL_MIN = 0.55     # Ít nhất 55% diện tích mỗi box detection chứa pixel của cùng 1 segment
MERGE_MASK_SHARE_MIN = 0.20   # Mỗi box chiếm tối thiểu 20% tổng diện tích segment đó
MERGE_DET_OVERLAP_MAX = 0.45  # Hai box detection phải là 2 vật thể tách biệt (không trùng nhau)

# 4. Thiếu khung bao (missing_box)
MISSING_MIN_AREA = 1600       # Bỏ qua vật thể quá nhỏ (tối thiểu 40x40 = 1600 pixel)
MISSING_MIN_REL_AREA = 0.008  # Hoặc chiếm tối thiểu 0.8% diện tích toàn ảnh
MISSING_MAX_OVERLAP = 0.25    # Không có box detection nào phủ quá 25% pixel của segment


def _load_classes(classes_path: Path) -> dict[int, str]:
    id_to_name: dict[int, str] = {}
    if not classes_path.exists():
        return id_to_name
    for line in classes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            id_to_name[int(parts[0])] = parts[1].strip()
    return id_to_name


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _read_yolo_boxes(
    label_path: Path, width: int, height: int, id_to_name: dict[int, str]
) -> list[dict]:
    boxes: list[dict] = []
    if not label_path.is_file():
        return boxes
    for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError:
            continue
        if cls_id not in id_to_name:
            continue
        x1 = max(0.0, (xc - bw / 2.0) * width)
        y1 = max(0.0, (yc - bh / 2.0) * height)
        x2 = min(float(width), (xc + bw / 2.0) * width)
        y2 = min(float(height), (yc + bh / 2.0) * height)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            {
                "idx": idx,
                "class_id": cls_id,
                "class_name": id_to_name[cls_id],
                "bbox": (x1, y1, x2, y2),
                "area": (x2 - x1) * (y2 - y1),
            }
        )
    return boxes


def _extract_thing_segments(
    mask_path: Path,
    segments_info: list[dict],
    cat_id_to_name: dict[int, str],
    yolo_names: set[str],
) -> tuple[int, int, list[dict]]:
    width, height, pixels = read_rgb(mask_path)

    valid_meta: dict[int, dict] = {}
    for seg in segments_info:
        seg_id = int(seg["id"])
        cat_id = int(seg["category_id"])
        iscrowd = int(seg.get("iscrowd", 0))
        cat_name = cat_id_to_name.get(cat_id, "")
        if seg_id == 0 or iscrowd == 1 or cat_name not in yolo_names:
            continue
        valid_meta[seg_id] = {
            "id": seg_id,
            "category_id": cat_id,
            "class_name": cat_name,
            "pixels": [],
            "min_x": width,
            "min_y": height,
            "max_x": 0,
            "max_y": 0,
        }

    if not valid_meta:
        return width, height, []

    stride = width * 3
    for y in range(height):
        row_offset = y * stride
        for x in range(width):
            sid = segment_id(pixels, row_offset + x * 3)
            info = valid_meta.get(sid)
            if info is not None:
                info["pixels"].append((x, y))
                if x < info["min_x"]:
                    info["min_x"] = x
                if x + 1 > info["max_x"]:
                    info["max_x"] = x + 1
                if y < info["min_y"]:
                    info["min_y"] = y
                if y + 1 > info["max_y"]:
                    info["max_y"] = y + 1

    segments: list[dict] = []
    for info in valid_meta.values():
        px_count = len(info["pixels"])
        if px_count == 0:
            continue
        info["pixel_count"] = px_count
        info["bbox"] = (
            float(info["min_x"]),
            float(info["min_y"]),
            float(info["max_x"]),
            float(info["max_y"]),
        )
        segments.append(info)

    return width, height, segments


def build(L: dict, skip_images: set[str] | None = None) -> list[dict]:
    """Hàm chuẩn được gọi từ validate.py: đối chiếu det <-> panoptic trên dữ liệu L."""
    delivery: Path = L["dir"]
    id_to_name: dict[int, str] = L["classes"]
    yolo_names = set(id_to_name.values())
    panoptic_data: dict = L["pan"]
    manifest: dict = L.get("man", {})
    skip = skip_images or set()

    cat_id_to_name = {
        int(c["id"]): str(c["name"]).strip()
        for c in panoptic_data.get("categories", [])
    }
    ann_by_image_id = {
        int(a["image_id"]): a for a in panoptic_data.get("annotations", [])
    }

    candidates: list[dict] = []

    for img_info in sorted(panoptic_data.get("images", []), key=lambda x: x["file_name"]):
        file_name = img_info["file_name"]
        stem = Path(file_name).stem

        # Bỏ qua ảnh đã bị lỗi luật (missing_image, wrong_hash, scene_leak, ...)
        if file_name in skip or stem in skip:
            continue

        img_id = int(img_info["id"])
        ann = ann_by_image_id.get(img_id)
        if ann is None:
            continue

        man_item = manifest.get(stem, {})
        mask_rel = man_item.get("mask") or f"masks/{ann.get('file_name', f'{stem}.png')}"
        label_rel = man_item.get("label") or f"labels/{stem}.txt"

        mask_path = delivery / mask_rel
        label_path = delivery / label_rel
        if not mask_path.is_file() or not label_path.is_file():
            continue

        width, height, segments = _extract_thing_segments(
            mask_path, ann.get("segments_info", []), cat_id_to_name, yolo_names
        )
        det_boxes = _read_yolo_boxes(label_path, width, height, id_to_name)
        img_area = float(width * height)

        # Tính ma trận đối chiếu pixel giữa từng segment và từng det_box
        seg_stats: list[list[tuple[int, float, float, float]]] = []
        for seg in segments:
            row_stats: list[tuple[int, float, float, float]] = []
            seg_px = seg["pixels"]
            total_px = seg["pixel_count"]
            for det in det_boxes:
                bx1, by1, bx2, by2 = det["bbox"]
                in_box = sum(1 for (px, py) in seg_px if bx1 <= px < bx2 and by1 <= py < by2)
                m_ratio = in_box / total_px if total_px > 0 else 0.0
                b_fill = in_box / det["area"] if det["area"] > 0 else 0.0
                iou = _bbox_iou(det["bbox"], seg["bbox"])
                row_stats.append((in_box, m_ratio, b_fill, iou))
            seg_stats.append(row_stats)

        image_candidates: list[dict] = []

        # 1. Kiểm tra instance_merge (1 segment phủ >= 2 box detection cùng lớp)
        merged_seg_indices: set[int] = set()
        for s_idx, seg in enumerate(segments):
            matched_dets: list[tuple[int, float, float]] = []
            for d_idx, det in enumerate(det_boxes):
                if det["class_name"] != seg["class_name"]:
                    continue
                _, m_ratio, b_fill, _ = seg_stats[s_idx][d_idx]
                if b_fill >= MERGE_BOX_FILL_MIN and m_ratio >= MERGE_MASK_SHARE_MIN:
                    matched_dets.append((d_idx, m_ratio, b_fill))

            if len(matched_dets) >= 2:
                has_distinct_pair = False
                for i in range(len(matched_dets)):
                    for j in range(i + 1, len(matched_dets)):
                        d1 = det_boxes[matched_dets[i][0]]["bbox"]
                        d2 = det_boxes[matched_dets[j][0]]["bbox"]
                        if _bbox_iou(d1, d2) < MERGE_DET_OVERLAP_MAX:
                            has_distinct_pair = True
                            break
                    if has_distinct_pair:
                        break
                if has_distinct_pair:
                    merged_seg_indices.add(s_idx)
                    image_candidates.append(
                        {
                            "image": stem,
                            "type": "instance_merge",
                            "det_class": seg["class_name"],
                            "panoptic_class": seg["class_name"],
                            "bbox": [round(v, 1) for v in seg["bbox"]],
                            "note": f"1 segment {seg['class_name']} phủ {len(matched_dets)} box detection tách biệt",
                        }
                    )

        # 2. Kiểm tra class_swap / category_swap và bbox_drift
        for s_idx, seg in enumerate(segments):
            if s_idx in merged_seg_indices or not det_boxes:
                continue
            best_d_idx = max(
                range(len(det_boxes)),
                key=lambda d: (seg_stats[s_idx][d][3], seg_stats[s_idx][d][1]),
            )
            det = det_boxes[best_d_idx]
            _, m_ratio, _, iou = seg_stats[s_idx][best_d_idx]

            if det["class_name"] != seg["class_name"]:
                if m_ratio >= SWAP_MASK_IN_BOX_MIN and iou >= SWAP_IOU_MIN:
                    image_candidates.append(
                        {
                            "image": stem,
                            "type": "class_swap",
                            "det_class": det["class_name"],
                            "panoptic_class": seg["class_name"],
                            "bbox": [round(v, 1) for v in det["bbox"]],
                            "iou": round(iou, 3),
                            "mask_in_box": round(m_ratio, 3),
                            "note": f"Lệch lớp (IoU={iou:.2f}): det={det['class_name']} vs panoptic={seg['class_name']}",
                        }
                    )
            else:
                if iou >= DRIFT_IOU_MIN and (m_ratio < DRIFT_MASK_IN_BOX_MAX or iou < DRIFT_IOU_MAX):
                    image_candidates.append(
                        {
                            "image": stem,
                            "type": "bbox_drift",
                            "det_class": det["class_name"],
                            "panoptic_class": seg["class_name"],
                            "bbox": [round(v, 1) for v in det["bbox"]],
                            "iou": round(iou, 3),
                            "mask_in_box": round(m_ratio, 3),
                            "note": f"Box lệch khỏi mask {seg['class_name']} (mask_in_box={m_ratio:.2f}, IoU={iou:.2f})",
                        }
                    )

        # 3. Kiểm tra missing_box (segment thing lớn nhưng không có box detection nào phủ)
        for s_idx, seg in enumerate(segments):
            if s_idx in merged_seg_indices:
                continue
            total_px = seg["pixel_count"]
            if total_px < MISSING_MIN_AREA or (total_px / img_area) < MISSING_MIN_REL_AREA:
                continue
            # Chỉ xét box CÙNG LỚP: box của vật khác lớp phủ lên (vd. người đứng trước ô tô)
            # không có nghĩa là segment này đã có box.
            max_m_ratio = max(
                (seg_stats[s_idx][d][1] for d in range(len(det_boxes))
                 if det_boxes[d]["class_name"] == seg["class_name"]),
                default=0.0,
            )
            if max_m_ratio < MISSING_MAX_OVERLAP:
                image_candidates.append(
                    {
                        "image": stem,
                        "type": "missing_box",
                        "det_class": None,
                        "panoptic_class": seg["class_name"],
                        "bbox": [round(v, 1) for v in seg["bbox"]],
                        "area_px": total_px,
                        "note": f"Panoptic có {seg['class_name']} ({total_px} px) nhưng thiếu box detection",
                    }
                )

        # Mỗi ảnh mang tối đa 1 lỗi -> giữ ứng viên nổi bật nhất
        if image_candidates:
            priority = {
                "class_swap": 1,
                "instance_merge": 2,
                "bbox_drift": 3,
                "missing_box": 4,
            }
            image_candidates.sort(key=lambda c: priority.get(c["type"], 99))
            candidates.append(image_candidates[0])

    return candidates


def find_candidates(delivery_dir: str | Path) -> list[dict]:
    """Hàm hỗ trợ chạy độc lập bằng CLI khi không truyền qua validate.py."""
    delivery = Path(delivery_dir)
    panoptic_path = delivery / "panoptic.json"
    if not panoptic_path.is_file():
        return []
    L = {
        "dir": delivery,
        "classes": _load_classes(delivery / "classes.txt"),
        "pan": json.loads(panoptic_path.read_text(encoding="utf-8")),
        "man": {},
    }
    return build(L, set())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh danh sách ứng viên đối chiếu det <-> panoptic")
    parser.add_argument("delivery", help="Đường dẫn tới thư mục delivery/")
    parser.add_argument("--out", required=True, help="Đường dẫn file candidates.json đầu ra")
    args = parser.parse_args()

    results = find_candidates(args.delivery)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"candidates": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {len(results)} ứng viên vào {out_path}")


if __name__ == "__main__":
    main()