#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from ultralytics import YOLO

# ========= 설정(환경변수 연동) =========
OV_MODEL_DIR = os.environ.get("OV_MODEL_DIR", "/home/pi/Desktop/detect/finetune_my6_es/weights/best_openvino_model")
IMG_SIZE     = int(os.environ.get("IMG_SIZE", "832"))

PRIMARY_CONF = float(os.environ.get("PRIMARY_CONF", "0.55"))  # 모델 내부 conf
IOU_THRESH   = float(os.environ.get("IOU_THRESHOLD", "0.70"))
AGNOSTIC_NMS = os.environ.get("AGNOSTIC_NMS", "1") == "1"

SAVE_DIR     = os.environ.get("SAVE_DIR", "/home/pi/kiosk_captures")  # 출력 폴더

def infer_and_save(model: YOLO, img_path: Path, save_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERR] cannot read: {img_path}")
        return

    results = model(
        img, imgsz=IMG_SIZE,
        conf=PRIMARY_CONF, iou=IOU_THRESH,
        agnostic_nms=AGNOSTIC_NMS, verbose=False
    )
    r = results[0]
    ann = r.plot()  # 원본 크기 기준으로 박스/라벨 그리기

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"{img_path.stem}_pred.jpg"
    if cv2.imwrite(str(out), ann):
        # 간단한 요약 출력
        names = model.names
        counts = {}
        if hasattr(r, "boxes") and hasattr(r.boxes, "cls"):
            for c in r.boxes.cls.cpu().numpy().astype(int):
                counts[names.get(int(c), str(c))] = counts.get(names.get(int(c), str(c)), 0) + 1
        print(f"[DONE] {img_path.name} → {out.name} | {counts}")
    else:
        print(f"[ERR] save failed → {out}")

def build_target_list(image_path: str, dir_path: str):
    if image_path:
        return [Path(image_path)]
    targets = []
    if dir_path:
        p = Path(dir_path)
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for ext in exts:
            targets += [Path(x) for x in glob.glob(str(p / ext))]
    return targets

def main():
    parser = argparse.ArgumentParser(description="YOLO image inference")
    parser.add_argument("--image", type=str, help="단일 이미지 경로")
    parser.add_argument("--dir", type=str, help="이미지 폴더 경로 (jpg/jpeg/png/bmp)")
    args = parser.parse_args()

    if not Path(OV_MODEL_DIR).exists():
        raise FileNotFoundError(f"OpenVINO 모델 경로 없음: {OV_MODEL_DIR}")

    print("[YOLO] loading:", OV_MODEL_DIR)
    model = YOLO(OV_MODEL_DIR)
    # warm-up
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    _ = model(dummy, imgsz=IMG_SIZE, verbose=False)

    targets = build_target_list(args.image, args.dir)
    if not targets:
        print("[WARN] 처리할 이미지가 없습니다. --image 또는 --dir 를 지정하세요.")
        return

    save_dir = Path(SAVE_DIR)
    for t in targets:
        infer_and_save(model, t, save_dir)

if __name__ == "__main__":
    main()
