import argparse, json, os
from PIL import Image, ImageDraw

def load_classes(path):
    classes = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                classes[int(parts[0])] = parts[1]
    return classes

def load_panoptic(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    id_to_stem = {
        img["id"]: os.path.splitext(img["file_name"])[0]
        for img in data.get("images", [])
        if "id" in img and "file_name" in img
    }
    anns = {}
    for ann in data.get("annotations", []):
        if "file_name" in ann:
            stem = os.path.splitext(ann["file_name"])[0]
        else:
            stem = id_to_stem.get(ann.get("image_id"), str(ann.get("image_id", "")))
        anns[stem] = ann.get("segments_info", [])
    return cats, anns

def find_image_path(delivery, img_stem):
    for name in (f"{img_stem}.jpg", f"{img_stem}.jpeg", f"{img_stem}.png", img_stem):
        p = os.path.join(delivery, "images", name)
        if os.path.exists(p):
            return p
    return None

def draw_image(delivery, img_stem, out_dir, classes, cats, anns, cand_info=None):
    img_path = find_image_path(delivery, img_stem)
    if not img_path:
        print(f"Khong tim thay anh: {img_stem}")
        return
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    thing_names = set(classes.values())

    # 1. Ve Panoptic bbox (Mau DO - nhan o goc duoi)
    for seg in anns.get(img_stem, []):
        if seg.get("iscrowd", 0) == 1:
            continue
        cname = cats.get(seg["category_id"], str(seg["category_id"]))
        if cname not in thing_names:
            continue
        x, y, bw, bh = seg["bbox"]
        draw.rectangle([x, y, x + bw, y + bh], outline="red", width=2)
        draw.text((x + 2, max(0, y + bh - 12)), f"PAN:{cname}", fill="red")

    # 2. Ve Detection YOLO box (Mau XANH LA - nhan o goc tren)
    lbl_path = os.path.join(delivery, "labels", f"{img_stem}.txt")
    if os.path.exists(lbl_path):
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cid, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
                x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
                cname = classes.get(cid, str(cid))
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                draw.text((x1 + 2, max(0, y1)), f"DET:{cname}", fill="lime")

    # 3. Ve khung VANG DAM khoanh dung ung vien tu candidates.json
    if cand_info and "bbox" in cand_info:
        bx1, by1, bx2, by2 = cand_info["bbox"]
        draw.rectangle([bx1 - 3, by1 - 3, bx2 + 3, by2 + 3], outline="yellow", width=3)
        tag = f"[{cand_info.get('type','')}] DET:{cand_info.get('det_class','')} | PAN:{cand_info.get('panoptic_class','')}"
        draw.text((max(0, bx1 - 3), max(0, by1 - 16)), tag, fill="yellow")

    os.makedirs(out_dir, exist_ok=True)
    im.save(os.path.join(out_dir, f"{img_stem}.jpg"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivery", required=True)
    ap.add_argument("--candidates", help="File candidates.json")
    ap.add_argument("--images", nargs="*")
    ap.add_argument("--out", default="runs/tmp_view")
    args = ap.parse_args()

    classes = load_classes(os.path.join(args.delivery, "classes.txt"))
    cats, anns = load_panoptic(os.path.join(args.delivery, "panoptic.json"))

    cand_map = {}
    targets = list(args.images) if args.images else []
    if args.candidates and os.path.exists(args.candidates):
        with open(args.candidates, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = raw.get("candidates", raw) if isinstance(raw, dict) else raw
        for it in items:
            raw_name = it if isinstance(it, str) else (it.get("image") or it.get("target"))
            stem = os.path.splitext(raw_name)[0]
            targets.append(stem)
            if isinstance(it, dict):
                cand_map[stem] = it

    for raw_stem in dict.fromkeys(targets):
        stem = os.path.splitext(raw_stem)[0]
        draw_image(args.delivery, stem, args.out, classes, cats, anns, cand_map.get(stem))
    print(f"Da ve {len(set(targets))} anh vao {args.out}")