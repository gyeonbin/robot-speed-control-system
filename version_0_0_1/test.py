# infer_speed_conv3d_webcam.py
# -------------------------------------------------------------
# 실시간 웹캠 추론 (VideoCapture(1) 기본)
# - 학습과 동일 전처리: RGB 정규화 + Gray 차분(Δt) 정규화 → 4채널
# - WINDOW, TEMP_STRIDE, IMG_SIZE 등은 meta_speed_conv3d.json에서 읽음
# - 가중치는 models/speed_conv3d_best.pt 우선, 없으면 speed_conv3d_last.pt
# - 키: q=종료, t=스레시홀드 0.05씩 토글, r=EMA 리셋
# -------------------------------------------------------------

import os
import json
import time
import argparse
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

# -------- 모델 정의 (학습 코드와 동일) --------
class Small3DCNN(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=(3,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2)),

            nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2,2,2)),

            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1)),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        z = self.backbone(x)
        z = z.flatten(1)
        logit = self.fc(z)
        return logit

# -------- 유틸: 전처리 (학습 코드와 동일) --------
def preprocess_rgb(frame_rgb, img_size, mean, std):
    img = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    return np.transpose(img, (2, 0, 1))  # (3,H,W)

def preprocess_gray(frame_rgb, img_size):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0  # (H,W)

def preprocess_diff(diff, dif_mean, dif_std):
    return (diff - dif_mean) / max(1e-6, dif_std)

def load_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # 안전장치: 필수 키 확인
    needed = ["window", "temp_stride", "img_size", "mean", "std", "diff_norm", "in_channels"]
    for k in needed:
        if k not in meta:
            raise KeyError(f"meta json에 '{k}'가 없습니다: {meta_path}")
    return meta

def draw_overlay(frame_bgr, text_lines, color=(255,255,255)):
    y = 24
    for line in text_lines:
        cv2.putText(frame_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
        y += 26

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=1, help="OpenCV VideoCapture index (default: 1)")
    parser.add_argument("--meta", type=str, default="models/meta_speed_conv3d.json")
    parser.add_argument("--best", type=str, default="models/speed_conv3d_best.pt")
    parser.add_argument("--last", type=str, default="models/speed_conv3d_last.pt")
    parser.add_argument("--thr", type=float, default=0.5, help="classification threshold for 'fast'")
    parser.add_argument("--ema", type=float, default=0.7, help="EMA smoothing for prob (0=off)")
    parser.add_argument("--win_name", type=str, default="Speed Conv3D - Webcam")
    args = parser.parse_args()

    meta = load_meta(args.meta)
    WINDOW = int(meta["window"])
    STRIDE = int(meta["temp_stride"])
    IMG_SIZE = int(meta["img_size"])
    MEAN = np.array(meta["mean"], dtype=np.float32)
    STD  = np.array(meta["std"], dtype=np.float32)
    DIF_MEAN = float(meta["diff_norm"]["mean"])
    DIF_STD  = float(meta["diff_norm"]["std"])
    IN_CH = int(meta["in_channels"])
    assert IN_CH == 4, f"in_channels expected 4, got {IN_CH}"

    need = (WINDOW - 1) * STRIDE + 1  # 필요한 원본 프레임 수

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()

    model = Small3DCNN(in_ch=IN_CH).to(device)
    weight_path = args.best if os.path.exists(args.best) else args.last
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"모델 가중치가 없습니다: {args.best} 또는 {args.last}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 캡처 열기
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({args.cam})을 열 수 없습니다.")

    cv2.namedWindow(args.win_name, cv2.WINDOW_NORMAL)

    # 버퍼: 원본 프레임 전처리 결과를 매 프레임 저장
    # 마지막 need 개만 유지해서 [::STRIDE]로 샘플링
    rgb_buf = deque(maxlen=need)  # 각 원소 shape: (3,H,W)
    dif_buf = deque(maxlen=need)  # 각 원소 shape: (1,H,W)
    last_gray = None

    prob_ema = None
    last_t = time.time()
    fps_disp = 0.0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 전처리
        proc_rgb = preprocess_rgb(frame_rgb, IMG_SIZE, MEAN, STD)  # (3,H,W)
        gray = preprocess_gray(frame_rgb, IMG_SIZE)                # (H,W)
        if last_gray is None:
            diff = np.zeros_like(gray, dtype=np.float32)
        else:
            diff = np.abs(gray - last_gray)
        proc_dif = preprocess_diff(diff, DIF_MEAN, DIF_STD)[None, ...]  # (1,H,W)

        rgb_buf.append(proc_rgb)
        dif_buf.append(proc_dif)
        last_gray = gray

        pred_text = "Warming up..."
        color = (0, 255, 255)
        prob = None

        # 충족 시 샘플 구성 → 추론
        if len(rgb_buf) == need:
            # STRIDE 간격으로 샘플 선택
            rgb_clip = np.stack(list(rgb_buf)[::STRIDE], axis=1)  # (3,T,H,W), T==WINDOW
            dif_clip = np.stack(list(dif_buf)[::STRIDE], axis=1)  # (1,T,H,W)
            x = np.concatenate([rgb_clip, dif_clip], axis=0)[None, ...]  # (1,4,T,H,W)
            x = torch.from_numpy(x).to(device)

            t0 = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
                logit = model(x).squeeze(1)  # (1,)
                prob = torch.sigmoid(logit)[0].item()  # fast 확률
            t1 = time.time()

            # EMA 스무딩
            if args.ema > 0.0:
                prob_ema = prob if prob_ema is None else (args.ema * prob_ema + (1 - args.ema) * prob)
                p_show = prob_ema
            else:
                p_show = prob

            label = "FAST" if p_show >= args.thr else "SLOW"
            color = (0, 255, 0) if label == "FAST" else (0, 165, 255)
            pred_text = f"{label}  p_fast={p_show:.3f}  thr={args.thr:.2f}  (raw={prob:.3f})"

            # FPS 측정(간단 평균)
            dt = t1 - t0
            fps_disp = 1.0 / max(1e-6, dt)

        # 오버레이
        hud = [
            pred_text,
            f"WINDOW={WINDOW}  STRIDE={STRIDE}  need={need}",
            f"INF FPS≈{fps_disp:.1f}  Device={'CUDA' if torch.cuda.is_available() else 'CPU'}",
            "keys: [q]=quit  [t]=thr±0.05  [r]=reset EMA"
        ]
        draw_overlay(frame_bgr, hud, color=color)

        cv2.imshow(args.win_name, frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('t'):
            # 스레시홀드 토글: 0.35 → 0.50 → 0.65 → ...
            args.thr += 0.05
            if args.thr > 0.95:
                args.thr = 0.35
        elif key == ord('r'):
            prob_ema = None  # EMA 리셋

    cap.release()
    if cv2.getWindowProperty(args.win_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(args.win_name)

if __name__ == "__main__":
    main()
