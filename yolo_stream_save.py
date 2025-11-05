#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, subprocess, threading
from datetime import datetime
import numpy as np
import cv2
from ultralytics import YOLO

# ==== 설정 ====
OV_MODEL_DIR = os.environ.get("OV_MODEL_DIR", "/home/pi/Desktop/detect/finetune_my6_es/weights/best_openvino_model")
IMG_SIZE     = int(os.environ.get("IMG_SIZE", "832"))
PRIMARY_CONF = float(os.environ.get("PRIMARY_CONF", "0.55"))  # 모델 내부 conf
IOU_THRESH   = float(os.environ.get("IOU_THRESHOLD", "0.70"))
CONF_THRESH  = float(os.environ.get("CONF_THRESHOLD", "0.45"))  # 후처리 필터
AGNOSTIC_NMS = os.environ.get("AGNOSTIC_NMS", "1") == "1"

SAVE_DIR     = os.environ.get("SAVE_DIR", "/home/pi/kiosk_captures")  # 저장 폴더
SAVE_EVERY_S = float(os.environ.get("SAVE_EVERY_S", "1.0"))           # 저장 주기(초)

# 카메라 파라미터 (rpicam-vid)
CAM_W       = int(os.environ.get("CAM_W", "640"))
CAM_H       = int(os.environ.get("CAM_H", "480"))
CAM_FPS     = int(os.environ.get("CAM_FPS", "25"))
CAM_SHUTTER = int(os.environ.get("CAM_SHUTTER", "20000"))
CAM_GAIN    = float(os.environ.get("CAM_GAIN", "1.0"))
CAM_DENOISE = os.environ.get("CAM_DENOISE", "off")

def start_camera_yuv420_stream():
    """rpicam-vid로 YUV420(raw) 스트림 받아 BGR 프레임로 변환해서 yield"""
    cmd = [
        "rpicam-vid",
        "-t", "0",
        "-n",
        "--width", str(CAM_W),
        "--height", str(CAM_H),
        "--framerate", str(CAM_FPS),
        "--codec", "yuv420",
        "--shutter", str(CAM_SHUTTER),
        "--gain", str(CAM_GAIN),
        "--denoise", str(CAM_DENOISE),
        "-o", "-"
    ]
    print("[CAM] exec:", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

    frame_size = CAM_W * CAM_H * 3 // 2  # YUV420(I420) 1.5 bytes/pixel
    stash = bytearray()
    try:
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                time.sleep(0.005)
                continue
            stash.extend(chunk)
            while len(stash) >= frame_size:
                fb = stash[:frame_size]
                del stash[:frame_size]
                yuv = np.frombuffer(fb, dtype=np.uint8).reshape((CAM_H * 3 // 2, CAM_W))
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                yield bgr
    finally:
        try:
            proc.terminate()
        except Exception:
            pass

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # YOLO 모델 로드
    if not os.path.exists(OV_MODEL_DIR):
        raise FileNotFoundError(f"OpenVINO 모델 경로가 없습니다: {OV_MODEL_DIR}")
    print("[YOLO] loading:", OV_MODEL_DIR)
    model = YOLO(OV_MODEL_DIR)

    # 워밍업
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    _ = model(dummy, imgsz=IMG_SIZE, verbose=False)

    last_saved = 0.0
    for bgr in start_camera_yuv420_stream():
        # 크기 맞춰 입력
        inp = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))

        # (선택) 약한 조명 보정
        # inp = cv2.GaussianBlur(inp, (0,0), 1.0)
        # inp = cv2.addWeighted(inp, 1.6, inp, -0.6, 0)

        # 추론
        results = model(
            inp,
            imgsz=IMG_SIZE,
            conf=PRIMARY_CONF,
            iou=IOU_THRESH,
            agnostic_nms=AGNOSTIC_NMS,
            verbose=False
        )
        r = results[0]

        # 후처리: CONF_THRESH로 한 번 더 거르기
        boxes = getattr(r, "boxes", None)
        if boxes is None or not hasattr(boxes, "cls") or len(boxes.cls) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        clses  = boxes.cls.cpu().numpy().astype(int)

        mask = scores >= CONF_THRESH
        if not mask.any():
            continue

        # 주기적으로 저장(기본 1초에 1장)
        now = time.time()
        if now - last_saved >= SAVE_EVERY_S:
            ann = r.plot()  # r.orig_img(=inp)에 박스 그려줌
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out_path = os.path.join(SAVE_DIR, f"{ts}.jpg")
            ok = cv2.imwrite(out_path, ann)
            if ok:
                print(f"[IMG] saved → {out_path}", flush=True)
            last_saved = now

if __name__ == "__main__":
    main()
