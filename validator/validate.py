#!/usr/bin/env python3
"""Validator của nhóm — điểm vào duy nhất (giao diện cố định của đề).

  python validator/validate.py <thư_mục_delivery> --out <errors.json> [--candidates <candidates.json>]

Chia việc để không sửa chung một file:
  validate.py          Người 1: đọc lô, luật FILE (missing_image, wrong_hash), luật METADATA
                       (scene_leak, guideline_mismatch), gộp kết quả, ghi errors.json
  rules_detection.py   Người 2: hàm check(L) -> invalid_class, bbox_out_of_bounds, duplicate_box
  rules_panoptic.py    Người 2: hàm check(L) -> invalid_category_id, thing_overlap
  candidates.py        Người 3: hàm build(L, skip_images) -> danh sách ứng viên lỗi gán nhãn
File nào chưa có thì validate.py bỏ qua và báo, nên chạy được ngay từ đầu.

Luật của đề (RULES.md): chỉ đọc thư mục delivery được truyền vào; chỉ thư viện chuẩn
(+ scripts/pngutil.py); không đọc _private/, data_source/; không mạng; không nhúng tên ảnh/đáp án.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # để import rules_detection, rules_panoptic, candidates
sys.path.insert(0, str(HERE.parent / "scripts"))  # để Người 2/3 import pngutil


# ============================================================ 1. ĐỌC LÔ
def load_delivery(d: Path) -> dict:
    """Đọc mọi file mô tả lô một lần, trả về dict L dùng chung cho mọi luật.

    L["man"]     : manifest.json  {tên ảnh: {image, label, mask, image_sha256, ...}}
    L["meta"]    : metadata.json  {tên ảnh: {scene_id, split, job_id, guideline_version}}
    L["jobs"]    : jobs.csv       {job_id: {annotator, reviewer, guideline}}
    L["classes"] : classes.txt    {0: "person", ...}
    L["pan"]     : panoptic.json  (nguyên văn)
    """
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["items"]
    meta = {m["image"]: m for m in json.loads((d / "metadata.json").read_text(encoding="utf-8"))["items"]}
    with (d / "jobs.csv").open(encoding="utf-8", newline="") as f:
        jobs = {row["job_id"].strip(): {k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)}
    classes = {}
    for line in (d / "classes.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            idx, name = line.split(" ", 1)
            classes[int(idx)] = name.strip()
    pan = json.loads((d / "panoptic.json").read_text(encoding="utf-8"))
    return {"dir": d, "man": man, "meta": meta, "jobs": jobs, "classes": classes, "pan": pan}


def sha256_file(path: Path) -> str:
    """sha256 tính trên BYTES của file (mở 'rb', đọc từng khúc 1 MB).

    Không được chỉ kiểm chuỗi trong manifest có đủ 64 ký tự hex: lỗi gài là hash HỢP LỆ
    nhưng của một ảnh khác, nên phải tự tính lại rồi so.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================ 2. LUẬT FILE
def check_files(L: dict) -> list[dict]:
    """missing_image: manifest có ảnh mà không có file.
    wrong_hash   : có file nhưng sha256 tính lại khác image_sha256 trong manifest.
    Ảnh thiếu thì không tính hash (không có gì để tính) => một ảnh không bị báo hai lỗi.
    """
    out = []
    for name, it in sorted(L["man"].items()):
        path = L["dir"] / it["image"]
        if not path.is_file():
            out.append({"type": "missing_image", "target": name, "why": f"không thấy {it['image']}"})
        elif sha256_file(path) != it["image_sha256"].strip().lower():
            out.append({"type": "wrong_hash", "target": name, "why": "sha256 tính lại khác manifest"})
    return out


# ============================================================ 3. LUẬT METADATA
def check_metadata(L: dict) -> tuple[list[dict], dict]:
    """guideline_mismatch: guideline_version của ẢNH khác guideline của JOB (jobs.csv).
        So với job chứ không so với các ảnh khác: phiên bản guideline là thứ được giao
        cùng job; job mới là nguồn đúng.
    scene_leak: một cảnh có ảnh ở cả train và val. TARGET LÀ scene_id, không phải tên ảnh.
        Kèm thông tin split đa số + ảnh lệch để Người 5 sửa split trong v2.
    """
    out = []
    scenes: dict[str, dict[str, list[str]]] = {}
    for name, m in sorted(L["meta"].items()):
        scenes.setdefault(m["scene_id"], {}).setdefault(m["split"], []).append(name)
        job = L["jobs"].get(m["job_id"])
        if job is None:
            print(f"  ! {name}: job {m['job_id']} không có trong jobs.csv (không tự kết luận)")
            continue
        if m.get("guideline_version", "").strip() != job["guideline"]:
            out.append({"type": "guideline_mismatch", "target": name,
                        "why": f"ảnh ghi {m.get('guideline_version')}, job {m['job_id']} ghi {job['guideline']}"})
    leaks = {}
    for scene, splits in sorted(scenes.items()):
        if len(splits) > 1:
            major = max(splits, key=lambda s: len(splits[s]))
            odd = sorted(n for s, names in splits.items() if s != major for n in names)
            leaks[scene] = {"majority_split": major, "odd_images": odd}
            out.append({"type": "scene_leak", "target": scene, "why": f"cảnh nằm ở {sorted(splits)}; ảnh lệch: {odd}"})
    return out, leaks


# ============================================================ 4. GỘP + GHI
def optional(module: str):
    """Import phần của người khác nếu file đã có; chưa có thì trả None."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name == module:
            return None
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description="Validator lô bàn giao")
    ap.add_argument("delivery")
    ap.add_argument("--out", required=True, help="errors.json: chỉ lỗi luật")
    ap.add_argument("--candidates", help="candidates.json: ứng viên lỗi gán nhãn (phải xem ảnh)")
    args = ap.parse_args()
    d = Path(args.delivery)
    L = load_delivery(d)

    found = check_files(L)
    meta_errs, leaks = check_metadata(L)
    found += meta_errs
    for mod in ("rules_detection", "rules_panoptic"):
        m = optional(mod)
        if m is None:
            print(f"  (chưa có validator/{mod}.py: bỏ qua)")
        else:
            found += m.check(L)

    # mỗi cặp (type, target) một lần; type chỉ thuộc 9 lỗi luật
    rule_types = {"missing_image", "wrong_hash", "invalid_class", "bbox_out_of_bounds", "duplicate_box",
                  "scene_leak", "guideline_mismatch", "invalid_category_id", "thing_overlap"}
    pairs = sorted({(e["type"], e["target"]) for e in found if e["type"] in rule_types})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"errors": [{"type": t, "target": g} for t, g in pairs]}, indent=1) + "\n",
                              encoding="utf-8")
    counts: dict[str, int] = {}
    for t, _ in pairs:
        counts[t] = counts.get(t, 0) + 1
    print(f"Lỗi luật: {len(pairs)} -> {args.out}")
    for t in sorted(counts):
        print(f"  {t:<20} {counts[t]}")

    if args.candidates:
        # Mỗi ảnh mang nhiều nhất một lỗi: ảnh đã có lỗi luật (kể cả ảnh lệch của scene_leak)
        # thì không sinh ứng viên gán nhãn.
        skip = {g for _, g in pairs} | {n for v in leaks.values() for n in v["odd_images"]}
        c = optional("candidates")
        cands = c.build(L, skip) if c else []
        if c is None:
            print("  (chưa có validator/candidates.py: candidates rỗng)")
        doc = {"note": "ỨNG VIÊN, chưa phải lỗi: mở ảnh xem rồi ghi review_log.csv",
               "rules_detail": found, "scene_leak": leaks, "candidates": cands}
        Path(args.candidates).parent.mkdir(parents=True, exist_ok=True)
        Path(args.candidates).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Ứng viên: {len(cands)} -> {args.candidates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
