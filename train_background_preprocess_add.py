# train_speed_conv3d_balanced.py
# -------------------------------------------------------------
# - 라벨: 파일명에 fast → 1, slow → 0 (breakfast 오탐 방지)
# - 입력: RGB(3) + Gray-Δt(1) = 4채널, 고정길이 윈도우 (≈ 3초)
# - 폴더: clips/*.mp4 (예: fast_1.mp4, slow_2.mp4 ...)
# - 분할: 영상 단위 stratified split (fast/slow 비율 유지)
# - 불균형: BCEWithLogitsLoss(pos_weight)만 사용 (샘플러 제거)
# - 안정화: SAMPLE_STRIDE로 시작 인덱스 간격, GradClip, AMP
# - 기타: 혼동행렬 프린트, sanity prints, 메타 저장
# -------------------------------------------------------------

import os, re, json, glob, random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------ 설정 ------------------
# 카메라 FPS가 고정(동일 캠)이라고 했으니 30으로 두고 필요시 바꿔도 됨.
FPS = 30
WINDOW_SEC = 3.0       # "3초 윈도우" 요구사항
TEMP_STRIDE = 2        # 프레임 간 간격(샘플링 스텝). 2면 15Hz 샘플링 느낌.
# WINDOW는 WINDOW_SEC, FPS, TEMP_STRIDE로 자동계산
WINDOW = max(2, int(round(WINDOW_SEC * FPS / TEMP_STRIDE)))  # ≈ 45 (3s @30fps, stride=2)

SAMPLE_STRIDE = 8      # 윈도우 시작 인덱스 간격(과적합/중복 완화)
IMG_SIZE = 112
BATCH_SIZE = 32
LR = 2e-3
EPOCHS = 10
NUM_WORKERS = 0        # Windows면 0 권장
WEIGHT_DECAY = 1e-4
SEED = 42

# 채널 정규화
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Δt(차영상) 정규화 스케일 (간단히 0.0/0.5로 시작 → 데이터에 맞춰 조정 가능)
DIF_MEAN, DIF_STD = 0.0, 0.5

# 라벨 파싱(단어 경계) - 'breakfast' 오탐 방지
FAST_RE = re.compile(r'(^|[^a-z])fast([^a-z]|$)')
SLOW_RE = re.compile(r'(^|[^a-z])slow([^a-z]|$)')

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------ 유틸 ------------------
def label_from_name(path: str) -> int:
    base = os.path.basename(path).lower()
    if FAST_RE.search(base) and "breakfast" not in base:
        return 1
    if SLOW_RE.search(base):
        return 0
    return -1

def preprocess_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return np.transpose(img, (2, 0, 1))  # (3,H,W)

def preprocess_gray(frame_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return (gray.astype(np.float32) / 255.0)  # (H,W)

def preprocess_diff(diff: np.ndarray) -> np.ndarray:
    # diff: 0..1 -> 간단 정규화
    return (diff - DIF_MEAN) / max(1e-6, DIF_STD)

# ------------------ Dataset ------------------
class OnTheFlyVideoDataset(Dataset):
    """
    - 디스크에서 바로 읽음(프레임 캐시 X)
    - 입력: (4,T,H,W) = RGB(3) + Gray-Δt(1)
    - WINDOW ≈ WINDOW_SEC(초) 길이가 되도록 TEMP_STRIDE로 등간격 샘플
    """
    def __init__(self, video_files: List[str], window: int = WINDOW,
                 stride: int = TEMP_STRIDE, sample_stride: int = SAMPLE_STRIDE):
        self.window = window
        self.stride = stride
        self.sample_stride = sample_stride
        self.samples: List[Tuple[str, int, int]] = []  # (video_path, start_idx, label)

        need = (window - 1) * stride + 1  # 필요한 원본 프레임 수
        for vp in video_files:
            lb = label_from_name(vp)
            if lb not in (0, 1):
                continue

            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total >= need:
                # 시작 인덱스 간격을 sample_stride로 띄워서 중복·상관을 낮춤
                for s in range(0, total - need + 1, self.sample_stride):
                    self.samples.append((vp, s, lb))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vp, s, label = self.samples[idx]
        cap = cv2.VideoCapture(vp)

        clip_rgb = []
        clip_dif = []
        last_gray = None

        for k in range(self.window):
            target_f = s + k * self.stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            ok, frame = cap.read()

            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                proc_rgb = preprocess_rgb(rgb)         # (3,H,W)
                gray = preprocess_gray(rgb)            # (H,W) in 0..1
            else:
                # fallback: 이전 프레임 재사용 (버그 수정: gray는 last_gray 사용)
                proc_rgb = clip_rgb[-1] if clip_rgb else np.zeros((3, IMG_SIZE, IMG_SIZE), np.float32)
                gray = last_gray if last_gray is not None else np.zeros((IMG_SIZE, IMG_SIZE), np.float32)

            if last_gray is None:
                diff = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
            else:
                diff = np.abs(gray - last_gray)        # 0..1
            diff = preprocess_diff(diff)               # 정규화

            clip_rgb.append(proc_rgb)
            clip_dif.append(diff[None, ...])           # (1,H,W)
            last_gray = gray

        cap.release()

        x_rgb = np.stack(clip_rgb, axis=1)   # (3,T,H,W)
        x_dif = np.stack(clip_dif, axis=1)   # (1,T,H,W)
        x = np.concatenate([x_rgb, x_dif], axis=0)  # (4,T,H,W)
        y = np.array([label], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

# ------------------ 모델 ------------------
class Small3DCNN(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=(3,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,2,2)),  # 공간만 다운

            nn.Conv3d(16, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2,2,2)),  # 시간/공간 다운

            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1,1,1)),      # 가변 T 안정화
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        z = self.backbone(x)     # (B,64,1,1,1)
        z = z.flatten(1)         # (B,64)
        logit = self.fc(z)       # (B,1)
        return logit

# ------------------ 평가(정확도/혼동행렬) ------------------
def evaluate(model, loader, device, verbose=False, thr=0.5):
    model.eval()
    total, correct = 0, 0
    tp=tn=fp=fn=0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().squeeze(1)
            prob = torch.sigmoid(model(x)).squeeze(1)
            pred = (prob >= thr).float()
            correct += (pred == y).sum().item()
            tp += ((pred==1)&(y==1)).sum().item()
            tn += ((pred==0)&(y==0)).sum().item()
            fp += ((pred==1)&(y==0)).sum().item()
            fn += ((pred==0)&(y==1)).sum().item()
            total += y.numel()
    if verbose:
        print(f"[VAL] TP:{tp} TN:{tn} FP:{fp} FN:{fn}")
    return correct / max(1, total)

# ------------------ 메인 ------------------
if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs("models", exist_ok=True)

    # 1) 비디오 목록 + 라벨 (clips 폴더 사용)
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join("clips", ext)))
    video_files = sorted(video_files)

    labeled = [(vp, label_from_name(vp)) for vp in video_files]
    labeled = [(vp, lb) for vp, lb in labeled if lb in (0, 1)]
    if not labeled:
        raise FileNotFoundError("clips/*.mp4 등에 fast_*, slow_* 라벨 파일이 필요합니다.")

    # 2) stratified split (영상 단위)
    fast = [vp for vp, lb in labeled if lb == 1]
    slow = [vp for vp, lb in labeled if lb == 0]
    random.shuffle(fast); random.shuffle(slow)

    def split_ratio(lst, r=0.8):
        n = max(1, int(len(lst) * r))
        return lst[:n], lst[n:]

    tr_fast, va_fast = split_ratio(fast, 0.8)
    tr_slow, va_slow = split_ratio(slow, 0.8)
    tr_files = tr_fast + tr_slow
    va_files = va_fast + va_slow
    random.shuffle(tr_files); random.shuffle(va_files)

    print("[INFO] split:",
          f"train fast/slow = {len(tr_fast)}/{len(tr_slow)} |",
          f"val fast/slow = {len(va_fast)}/{len(va_slow)}")

    # 3) Dataset/DataLoader
    tr_ds = OnTheFlyVideoDataset(tr_files, window=WINDOW, stride=TEMP_STRIDE, sample_stride=SAMPLE_STRIDE)
    va_ds = OnTheFlyVideoDataset(va_files, window=WINDOW, stride=TEMP_STRIDE, sample_stride=SAMPLE_STRIDE) if len(va_files) > 0 else None
    if len(tr_ds) == 0:
        raise RuntimeError("학습 윈도우가 0개입니다. 영상 길이 또는 WINDOW/STRIDE를 확인하세요.")

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=False)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=False) if va_ds else None

    print(f"[INFO] WINDOW={WINDOW} (~{WINDOW_SEC:.1f}s @FPS={FPS}, stride={TEMP_STRIDE}) | "
          f"train windows: {len(tr_ds)} | val windows: {len(va_ds) if va_ds else 0}")

    # pos_weight (BCE) — 샘플러 대신 이것만 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()

    labels = [lb for _, _, lb in tr_ds.samples]
    n_pos = sum(1 for lb in labels if lb == 1)
    n_neg = len(labels) - n_pos
    print(f"[INFO] train label counts (window-level): slow={n_neg} / fast={n_pos}")

    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device, dtype=torch.float32)

    model = Small3DCNN(in_ch=4).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train_one_epoch():
        model.train()
        total_loss, total_n = 0.0, 0
        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logit = model(x)
                loss = loss_fn(logit, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optim)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)
        return total_loss / max(1, total_n)

    # 4) Sanity prints: 첫 배치 범위
    x0, y0 = next(iter(tr_loader))
    print("[INFO] first batch:", x0.shape, "min/max:", x0.min().item(), x0.max().item())

    # 5) 학습 루프
    best_acc = -1.0
    for e in range(1, EPOCHS + 1):
        loss = train_one_epoch()
        if va_loader:
            acc = evaluate(model, va_loader, device, verbose=(e % 5 == 0))
            print(f"Epoch {e:02d} | loss={loss:.4f} | val_acc={acc:.3f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join("models", "speed_conv3d_best.pt"))
        else:
            print(f"Epoch {e:02d} | loss={loss:.4f}")

    # 6) 마지막 저장 + 메타
    torch.save(model.state_dict(), os.path.join("models", "speed_conv3d_last.pt"))
    meta = {
        "fps": FPS,
        "window_sec": WINDOW_SEC,
        "window": WINDOW,
        "temp_stride": TEMP_STRIDE,
        "sample_stride": SAMPLE_STRIDE,
        "img_size": IMG_SIZE,
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "diff_norm": {"mean": DIF_MEAN, "std": DIF_STD},
        "label_def": {"slow": 0, "fast": 1},
        "in_channels": 4,
    }
    with open(os.path.join("models", "meta_speed_conv3d.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE] Saved models/speed_conv3d_best.pt (if better) & speed_conv3d_last.pt, meta json")
