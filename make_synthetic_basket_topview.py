
import os
import re
import cv2
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# =============================
# Config (EDIT THESE PATHS)
# =============================
BG_DIR = "C:/Users/serah/Downloads/backgrounds"     # folder with empty basket images (jpg/png)
OBJ_DIR = "C:/Users/serah/Downloads/objects"        # objects/<class>/*.png (RGBA) or flat: objects/class_*.png
OUT_DIR = "C:/Users/serah/Downloads/synthetic"  # output root; will create images/ and labels/

IMAGES_PER_BG = 30         # composites per background image
OBJS_PER_IMAGE = (1, 3)    # min, max objects per composite

# Top-view friendly placement & looks
MARGIN = 0.12              # keep away from walls/edges
ANGLE_RANGE = (-15, 15)    # gentle rotation (deg)
OVERLAP_PROB = 0.45        # how often to try overlapping
TARGET_IOU_RANGE = (0.2, 0.4)

# Global photometric jitter at the end
BRIGHTNESS_RANGE = (0.75, 1.25)
CONTRAST_RANGE   = (0.8, 1.2)
GAUSSIAN_BLUR_P  = 0.30    # probability of applying slight blur

# Reproducibility
SEED = 1337  # set None for non-deterministic

# =============================
# Helpers
# =============================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

def discover_classes_and_files(obj_dir: Path) -> Dict[str, List[Path]]:
    # \""\"Support subfolders per class OR flat files prefixed by class name.\"\"\"
    classes: Dict[str, List[Path]] = {}
    subdirs = [d for d in obj_dir.iterdir() if d.is_dir()]
    if subdirs:
        for sd in subdirs:
            files = [p for p in sd.iterdir() if p.suffix.lower() == ".png"]
            if files:
                classes[sd.name] = sorted(files)
        return classes

    # Flat
    pngs = [p for p in obj_dir.iterdir() if p.suffix.lower() == ".png"]
    for p in pngs:
        m = re.match(r"([A-Za-z0-9\-]+)[_\-].*\.png$", p.name)
        cname = m.group(1) if m else p.stem
        classes.setdefault(cname, []).append(p)
    return classes

def rotate_rgba(img_rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img_rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW/2) - w/2
    M[1, 2] += (nH/2) - h/2
    return cv2.warpAffine(img_rgba, M, (nW, nH),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0,0,0,0))

def paste_rgba(dst_bgr: np.ndarray, src_rgba: np.ndarray, x: int, y: int):
    # \"\"\"Alpha blend RGBA onto BGR at top-left (x,y). Returns alpha crop used.\"\"\"
    h, w = src_rgba.shape[:2]
    H, W = dst_bgr.shape[:2]
    if x >= W or y >= H:
        return None
    x_end = min(W, x + w)
    y_end = min(H, y + h)
    src_crop = src_rgba[0:(y_end-y), 0:(x_end-x), :]
    if src_crop.size == 0: return None
    alpha = src_crop[:, :, 3:4] / 255.0
    inv = 1.0 - alpha
    dst_crop = dst_bgr[y:y_end, x:x_end, :]
    dst_bgr[y:y_end, x:x_end, :] = (alpha * src_crop[:, :, :3] + inv * dst_crop).astype(np.uint8)
    return src_crop[:, :, 3]

def bbox_from_alpha_at(x, y, alpha: np.ndarray):
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0: return None
    x1 = x + int(xs.min()); x2 = x + int(xs.max())
    y1 = y + int(ys.min()); y2 = y + int(ys.max())
    return (x1, y1, x2, y2)

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = max(0, (ax2 - ax1 + 1)) * max(0, (ay2 - ay1 + 1))
    area_b = max(0, (bx2 - bx1 + 1)) * max(0, (by2 - by1 + 1))
    return inter / (area_a + area_b - inter + 1e-6)

def resize_to_target_area(rgba, W, H, n_objs, min_side_ratio=0.08, max_side_ratio=0.60):
    # \"\"\"Resize RGBA so that its area ~ target % of frame, with side-ratio clamps.\"\"\"
    if   n_objs <= 1: target = random.uniform(0.18, 0.28)
    elif n_objs == 2: target = random.uniform(0.12, 0.20)
    else:             target = random.uniform(0.08, 0.14)

    frame_area = W * H
    tgt_pixels = frame_area * target
    h, w = rgba.shape[:2]
    if w*h <= 0: return rgba
    scale = (tgt_pixels / (w*h)) ** 0.5

    min_side = int(min(W, H) * min_side_ratio)
    max_side = int(min(W, H) * max_side_ratio)
    nw = max(min_side, min(max_side, int(w * scale)))
    nh = max(min_side, min(max_side, int(h * scale)))
    nw, nh = max(4, nw), max(4, nh)
    return cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_LINEAR)

# =============================
# Main
# =============================
def main():
    if SEED is not None:
        set_seed(SEED)

    bg_dir = Path(BG_DIR)
    obj_dir = Path(OBJ_DIR)
    out_dir = Path(OUT_DIR)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    bgs = [p for p in bg_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
    if not bgs:
        raise FileNotFoundError(f"No backgrounds in {BG_DIR}")

    classes = discover_classes_and_files(obj_dir)
    if not classes:
        raise FileNotFoundError(f"No objects found in {OBJ_DIR} (need RGBA PNGs)")

    # 원하는 순서로 클래스 ID 고정
    custom_order = ['downy', 'homestar', 'pepper']
    class_names = [n for n in custom_order if n in classes]

    # 누락된 클래스(혹시 모를 추가 객체)가 있으면 뒤에 자동 추가
    for n in classes.keys():
        if n not in class_names:
            class_names.append(n)

    # 고정된 ID 매핑
    name_to_id = {n:i for i,n in enumerate(class_names)}


    with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
        for n in class_names:
            f.write(n + "\n")

    print(f"[INFO] Backgrounds: {len(bgs)}")
    print(f"[INFO] Classes: {name_to_id}")

    total = 0
    for bg_path in bgs:
        bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if bg is None:
            print(f"[WARN] skip bg: {bg_path}")
            continue
        H, W = bg.shape[:2]

        for k in range(IMAGES_PER_BG):
            canvas = bg.copy()
            labels = []
            xyxys  = []

            n_objs = random.randint(OBJS_PER_IMAGE[0], OBJS_PER_IMAGE[1])
            for oi in range(n_objs):
                cname = random.choice(class_names)
                cid = name_to_id[cname]
                src_path = random.choice(classes[cname])
                rgba = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
                if rgba is None or rgba.shape[2] != 4:
                    print(f"[WARN] skip non-RGBA: {src_path}")
                    continue

                # gentle rotation + size by target area
                angle = random.uniform(ANGLE_RANGE[0], ANGLE_RANGE[1])
                rgba = rotate_rgba(rgba, angle)
                rgba = resize_to_target_area(rgba, W, H, n_objs)

                attempts = 25
                placed = False
                last_xyxy = xyxys[-1] if (xyxys and random.random() < OVERLAP_PROB) else None

                for _ in range(attempts):
                    if last_xyxy is None:
                        xmin = int(W * MARGIN); xmax = int(W * (1 - MARGIN) - rgba.shape[1])
                        ymin = int(H * MARGIN); ymax = int(H * (1 - MARGIN) - rgba.shape[0])

                        if xmax < xmin or ymax < ymin:
                            xmin, ymin = 0, 0
                            xmax = max(0, W - rgba.shape[1])
                            ymax = max(0, H - rgba.shape[0])

                        x = random.randint(xmin, max(xmin, xmax)) if xmax >= xmin else 0
                        y = random.randint(ymin, max(ymin, ymax)) if ymax >= ymin else 0
                    else:
                        cx = (last_xyxy[0] + last_xyxy[2]) // 2
                        cy = (last_xyxy[1] + last_xyxy[3]) // 2
                        jitter = max(8, int(min(W, H) * 0.08))
                        x = int(max(0, min(W - rgba.shape[1], cx + random.randint(-jitter, jitter) - rgba.shape[1] // 2)))
                        y = int(max(0, min(H - rgba.shape[0], cy + random.randint(-jitter, jitter) - rgba.shape[0] // 2)))

                    alpha_used = paste_rgba(canvas, rgba, x, y)
                    if alpha_used is None: 
                        continue
                    xyxy = bbox_from_alpha_at(x, y, alpha_used)
                    if xyxy is None: 
                        continue

                    if last_xyxy is not None:
                        iou = iou_xyxy(xyxy, last_xyxy)
                        if not (TARGET_IOU_RANGE[0] <= iou <= TARGET_IOU_RANGE[1]):
                            canvas[y:y+rgba.shape[0], x:x+rgba.shape[1]] = bg[y:y+rgba.shape[0], x:x+rgba.shape[1]]
                            continue

                    placed = True
                    xyxys.append(xyxy)
                    cxn = (xyxy[0] + xyxy[2]) / 2 / W
                    cyn = (xyxy[1] + xyxy[3]) / 2 / H
                    wn  = (xyxy[2] - xyxy[0] + 1) / W
                    hn  = (xyxy[3] - xyxy[1] + 1) / H
                    labels.append((cid, cxn, cyn, wn, hn))
                    break

                if not placed:
                    x = random.randint(0, max(0, W - rgba.shape[1]))
                    y = random.randint(0, max(0, H - rgba.shape[0]))
                    alpha_used = paste_rgba(canvas, rgba, x, y)
                    if alpha_used is not None:
                        xyxy = bbox_from_alpha_at(x, y, alpha_used)
                        if xyxy is not None:
                            cxn = (xyxy[0] + xyxy[2]) / 2 / W
                            cyn = (xyxy[1] + xyxy[3]) / 2 / H
                            wn  = (xyxy[2] - xyxy[0] + 1) / W
                            hn  = (xyxy[3] - xyxy[1] + 1) / H
                            labels.append((cid, cxn, cyn, wn, hn))

            # global photometric jitter
            img = canvas.astype(np.float32) * random.uniform(CONTRAST_RANGE[0], CONTRAST_RANGE[1])
            img = np.clip(img, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1]), 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            if random.random() < GAUSSIAN_BLUR_P:
                img = cv2.GaussianBlur(img, (5,5), 0)

            # save
            stem = f"{bg_path.stem}_{k:03d}"
            out_img = Path(OUT_DIR) / "images" / f"{stem}.jpg"
            out_lbl = Path(OUT_DIR) / "labels" / f"{stem}.txt"
            cv2.imwrite(str(out_img), img)
            with open(out_lbl, "w", encoding="utf-8") as f:
                for (cid, cx, cy, w, h) in labels:
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            print(f"[GEN] {out_img.name}  objs={len(labels)}")

    total = len(list((Path(OUT_DIR)/"images").glob("*.jpg")))
    print(f"[DONE] Generated {total} images to {OUT_DIR}/images with labels in {OUT_DIR}/labels")
    print(f"[INFO] classes.txt saved at {OUT_DIR}/classes.txt")

if __name__ == "__main__":
    main()
