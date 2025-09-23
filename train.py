# train_speed_conv3d.py
# -------------------------------------------------------------
# 원본 영상 프레임을 그대로 사용하여 5프레임 윈도우에 대해 3D Convolution으로
# "빠름/느림" 이진 분류를 학습하는 스크립트. ROI/스켈레톤 사용 없음.
# - 데이터: data/*.mp4 (파일명에 fast → 1, slow → 0 자동 라벨)
# - 입력 크기: (C=3, T=5, H=112, W=112)
# - 모델: 간단한 3D CNN → GlobalAvgPool → Linear → Sigmoid
# - 저장: models/speed_conv3d.pt, models/meta_speed_conv3d.json
# - 메모리 절약: On-the-fly Dataset (프레임 캐시 제거)
# -------------------------------------------------------------

import os
import json
import glob
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------ 설정 ------------------
WINDOW = 5              # 5프레임 윈도우
IMG_SIZE = 112          # 112x112로 리사이즈 (가볍고 빠름)
BATCH_SIZE = 8          # 메모리 절약
LR = 1e-4
EPOCHS = 8
NUM_WORKERS = 0         # Windows 권장 0

# 채널 정규화 (간단히 0~1 스케일 + 평균/표준편차 정규화)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ------------------ 유틸 ------------------

def preprocess_frame(frame_rgb: np.ndarray) -> np.ndarray:
    # 입력: RGB uint8 (H,W,3) → 출력: float32 (3,H,W)
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))


class OnTheFlyVideoDataset(Dataset):
    """프레임 캐시 없이 디스크에서 바로 읽는 메모리 친화적 Dataset"""
    def __init__(self, video_files: List[str], window: int = WINDOW):
        self.window = window
        self.samples: List[Tuple[str, int, int]] = []  # (video_path, start_idx, label)

        for vp in video_files:
            base = os.path.basename(vp).lower()
            if   'fast' in base: label = 1
            elif 'slow' in base: label = 0
            else: continue

            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total >= window:
                for s in range(0, total - window + 1):
                    self.samples.append((vp, s, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vp, s, label = self.samples[idx]
        cap = cv2.VideoCapture(vp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)

        clip = []
        last = None
        for _ in range(self.window):
            ok, frame = cap.read()
            if not ok:
                # 말미에서 실패하면 마지막 프레임 반복
                if last is None:
                    # 완전 실패 시 제로 클립
                    x = np.zeros((3, self.window, IMG_SIZE, IMG_SIZE), dtype=np.float32)
                    y = np.array([label], dtype=np.float32)
                    cap.release()
                    return torch.from_numpy(x), torch.from_numpy(y)
                clip.append(last)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            proc = preprocess_frame(rgb)
            clip.append(proc)
            last = proc
        cap.release()

        x = np.stack(clip, axis=1)  # (3,T,H,W)
        y = np.array([label], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


# ------------------ 3D CNN 모델 ------------------
class Small3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2)),   # 공간만 1/2

            nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2,2,2)),   # 시간/공간 모두 1/2 (T:5→2)

            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1)),       # (C,1,1,1)
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B,3,T=5,H,W)
        z = self.backbone(x)     # (B,64,1,1,1)
        z = z.flatten(1)         # (B,64)
        logit = self.fc(z)       # (B,1)
        return logit


# ------------------ 학습 루프 ------------------

def train_one_epoch(model, loader, optim, device, amp=False):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logit = model(x)
            loss = loss_fn(logit, y)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
    return total_loss / max(1, total_n)


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prob = torch.sigmoid(model(x))
            pred = (prob >= 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    video_files = sorted(glob.glob(os.path.join("data", "*.mp4")))
    if len(video_files) == 0:
        raise FileNotFoundError("data/*.mp4 에 학습용 mp4를 넣어주세요 (파일명에 fast/slow 포함)")

    # 8:2 split
    n = len(video_files)
    n_tr = max(1, int(n * 0.8))
    tr_files = video_files[:n_tr]
    va_files = video_files[n_tr:]

    tr_ds = OnTheFlyVideoDataset(tr_files, window=WINDOW)
    va_ds = OnTheFlyVideoDataset(va_files, window=WINDOW) if len(va_files) > 0 else None

    if len(tr_ds) == 0:
        raise RuntimeError("학습 윈도우가 0개입니다. 동영상 길이를 확인하세요.")

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False) if va_ds else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()  # GPU면 AMP 사용

    model = Small3DCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"[INFO] train windows: {len(tr_ds)} | val windows: {len(va_ds) if va_ds else 0}")
    for e in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, tr_loader, optim, device, amp=amp)
        if va_loader:
            acc = evaluate(model, va_loader, device)
            print(f"Epoch {e:02d} | loss={loss:.4f} | val_acc={acc:.3f}")
        else:
            print(f"Epoch {e:02d} | loss={loss:.4f}")

    torch.save(model.state_dict(), os.path.join("models", "speed_conv3d.pt"))
    meta = {
        "window": WINDOW,
        "img_size": IMG_SIZE,
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "label_def": {"slow": 0, "fast": 1},
    }
    with open(os.path.join("models", "meta_speed_conv3d.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[DONE] Saved models/speed_conv3d.pt and models/meta_speed_conv3d.json")
