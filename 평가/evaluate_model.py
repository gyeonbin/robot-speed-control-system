# evaluate_conv3d_folder.py
# -------------------------------------------------------------
# - models/speed_conv3d_best.pt 로드 (state_dict)
# - meta_speed_conv3d.json 로 파라미터 동기화(FPS, WINDOW, STRIDE, IMG_SIZE, 구조 옵션, 정규화 등)
# - 입력 폴더 내 *.mp4 전부 평가 (파일명 fast/slow로 라벨 결정, breakfast 예외)
# - 각 파일: 통짜 디코딩 → stride로 WINDOW개 프레임 선택 → (read 직후 1600x900 1회 리사이즈)
#            → 4채널 텐서(RGB정규화3 + Gray-Δt 1) → 모델 추론
# - 출력: 터미널(혼동행렬+리포트+시간통계), PNG(confusion_matrix.png), CSV(eval_conv3d_results.csv)
# -------------------------------------------------------------

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import csv

# ---------- 라벨 규칙 ----------
FAST_RE = re.compile(r'(^|[^a-z])fast([^a-z]|$)', re.IGNORECASE)
SLOW_RE = re.compile(r'(^|[^a-z])slow([^a-z]|$)', re.IGNORECASE)

def label_from_name(name: str) -> int:
    n = name.lower()
    if FAST_RE.search(n) and "breakfast" not in n:
        return 1
    if SLOW_RE.search(n):
        return 0
    return -1

# ---------- 활성함수 ----------
def get_act(name: str):
    return nn.ReLU(inplace=True) if name == "relu" else nn.SiLU(inplace=True)

# ---------- 모델 (학습 스크립트와 동일 구조) ----------
class Small3DCNN(nn.Module):
    def __init__(
        self,
        in_ch=4,
        width_mult=1.0,
        img_size=112,
        n_blocks=3,                 # {2,3}
        pool_t_pattern="b1",        # {"none","b1","b1b2"}
        kt_mode="3x3x3",            # {"3x3x3","3x3x1+stride_t2"}
        dropout_p=0.0,              # {0.0,0.1,0.3}
        act_name="relu"             # {"relu","silu"}
    ):
        super().__init__()
        self.img_size = img_size
        self.act = get_act(act_name)

        c1 = int(round(16 * width_mult))
        c2 = int(round(32 * width_mult))
        c3 = int(round(64 * width_mult))
        chans = [c1, c2, c3][:n_blocks]

        blocks = []
        in_c = in_ch
        for i, out_c in enumerate(chans):
            # 시간 커널/스트라이드
            if kt_mode == "3x3x3":
                k = (3,3,3); s = (1,1,1)
            else:  # "3x3x1+stride_t2"
                k = (3,3,1); s = (2,1,1) if i > 0 else (1,1,1)

            blocks += [
                nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=(k[0]//2,1,1), bias=False),
                nn.BatchNorm3d(out_c),
                self.act,
            ]

            # 공간/시간 풀링
            do_pool_t = (pool_t_pattern == "b1" and i == 0) or (pool_t_pattern == "b1b2" and i <= 1)
            pt = 2 if do_pool_t else 1
            blocks += [nn.MaxPool3d(kernel_size=(pt, 2, 2))]

            # 드롭아웃
            if dropout_p > 0:
                blocks += [nn.Dropout3d(p=dropout_p)]

            in_c = out_c

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(1),
            nn.Linear(chans[-1], 1)
        )

    def forward(self, x):
        # (B, C=4, T, H=?, W=?) → interpolate로 (img_size, img_size)
        if self.img_size is not None:
            B, C, T, H, W = x.shape
            x2 = x.reshape(B, C*T, H, W)
            x2 = F.interpolate(x2, size=(self.img_size, self.img_size), mode="area", align_corners=None)
            x = x2.reshape(B, C, T, self.img_size, self.img_size)
        z = self.backbone(x)
        z = self.head(z)
        return z  # (B,1)

# ---------- 전처리 (학습과 동일 정책) ----------
RESIZE_W, RESIZE_H = 1600, 900  # read 직후 단 1회 리사이즈

def normalize_rgb(rgb_float01: np.ndarray, mean, std) -> np.ndarray:
    img = (rgb_float01 - mean) / std
    return np.transpose(img.astype(np.float32), (2, 0, 1))  # (3,H,W)

def to_gray01(rgb_uint8_resized: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_uint8_resized, cv2.COLOR_RGB2GRAY)
    return (gray.astype(np.float32) / 255.0)  # (H,W)

def norm_diff(diff: np.ndarray, dif_mean, dif_std) -> np.ndarray:
    return (diff - dif_mean) / max(1e-6, dif_std)

# ---------- 윈도우 구성 후 텐서 만들기 ----------
def build_clip_tensor_from_video(path, window, stride, mean, std, dif_mean, dif_std):
    """
    path의 전체 프레임을 디코딩한 뒤,
    indices = [0, 0+stride, ..., 0+(window-1)*stride] 프레임을 뽑아 4채널(3+1) 시퀀스 텐서로 만든다.
    - 프레임은 read 직후 1600x900으로 '단 1회' 리사이즈 (학습 정책과 동일)
    반환: x (torch.FloatTensor, shape=(1,4,T,H,W)), preprocess_time_ms
    부족하면 None, None
    """
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need = (window - 1) * stride + 1
    if total < need:
        cap.release()
        return None, None

    frames_rgb = []
    idxs = [0 + k * stride for k in range(window)]
    for fidx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame_bgr = cap.read()
        if not ok:
            cap.release()
            return None, None
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # 단 1회 리사이즈 (1600x900)
        frame_rgb = cv2.resize(frame_rgb, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_AREA)
        frames_rgb.append(frame_rgb)
    cap.release()

    clip_rgb = []
    clip_dif = []
    last_gray = None
    for rgb in frames_rgb:
        rgb01    = (rgb.astype(np.float32) / 255.0)
        proc_rgb = normalize_rgb(rgb01, mean, std)             # (3,H,W)
        gray     = to_gray01(rgb)                               # (H,W)

        if last_gray is None:
            diff = np.zeros_like(gray, dtype=np.float32)
        else:
            diff = np.abs(gray - last_gray)
        diff = norm_diff(diff, dif_mean, dif_std)               # (H,W)
        last_gray = gray

        clip_rgb.append(proc_rgb)               # list of (3,H,W)
        clip_dif.append(diff[None, ...])        # list of (1,H,W)

    x_rgb = np.stack(clip_rgb, axis=1)          # (3,T,H,W)
    x_dif = np.stack(clip_dif, axis=1)          # (1,T,H,W)
    x = np.concatenate([x_rgb, x_dif], axis=0)  # (4,T,H,W)
    x = torch.from_numpy(x).unsqueeze(0).float()# (1,4,T,H,W)

    t1 = time.perf_counter()
    pre_ms = (t1 - t0) * 1000.0
    return x, pre_ms

# ---------- 메인 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="평가할 폴더 경로 (*.mp4)")
    parser.add_argument("--model", type=str, default="models/speed_conv3d_best.pt")
    parser.add_argument("--meta",  type=str, default="models/meta_speed_conv3d.json")
    parser.add_argument("--csv",   type=str, default="eval_conv3d_results.csv")
    parser.add_argument("--cm_png",type=str, default="confusion_matrix_conv3d.png")
    args = parser.parse_args()

    folder = Path(args.folder)
    files = sorted(folder.glob("*.mp4"))
    if not files:
        print(f"[ERR] 폴더에 mp4 없음: {folder}")
        return

    # 메타 로드(학습 파라미터와 동기화)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    WINDOW   = int(meta["window"])                 # 예: 15
    STRIDE   = int(meta["temp_stride"])            # 예: 2
    IMG_SIZE = int(meta["img_size"])               # 예: 112
    MEAN     = np.array(meta["mean"], dtype=np.float32)
    STD      = np.array(meta["std"], dtype=np.float32)
    DIF_MEAN = float(meta["diff_norm"]["mean"])
    DIF_STD  = float(meta["diff_norm"]["std"])
    in_channels = int(meta.get("in_channels", 4))

    # 구조 하이퍼파라미터 (없으면 기본값 사용)
    model_cfg = meta.get("model", {})
    width_mult     = float(model_cfg.get("width_mult", 1.0))
    n_blocks       = int(model_cfg.get("n_blocks", 3))
    pool_t_pattern = model_cfg.get("pool_t_pattern", "b1")
    kt_mode        = model_cfg.get("kt_mode", "3x3x3")
    dropout_p      = float(model_cfg.get("dropout_p", 0.0))
    act_name       = model_cfg.get("act", "relu")

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Small3DCNN(
        in_ch=in_channels,
        width_mult=width_mult,
        img_size=IMG_SIZE,
        n_blocks=n_blocks,
        pool_t_pattern=pool_t_pattern,
        kt_mode=kt_mode,
        dropout_p=dropout_p,
        act_name=act_name
    ).to(device)

    ckpt = torch.load(args.model, map_location=device)
    # state_dict만 저장되어 있으니 바로 로드
    model.load_state_dict(ckpt)
    model.eval()

    y_true, y_pred, probs = [], [], []
    pre_times, inf_times = [], []
    rows = []

    print(f"[INFO] WINDOW={WINDOW}, STRIDE={STRIDE}, IMG_SIZE={IMG_SIZE}, device={device}")
    print(f"[INFO] Evaluating {len(files)} files in {folder} ...")

    with torch.no_grad():
        for vp in files:
            gt = label_from_name(vp.name)
            if gt not in (0,1):
                print(f"[SKIP] 라벨 파악 불가: {vp.name}")
                continue

            # 전처리(윈도우 텐서 생성)
            x, pre_ms = build_clip_tensor_from_video(
                vp, WINDOW, STRIDE, MEAN, STD, DIF_MEAN, DIF_STD
            )
            if x is None:
                print(f"[SKIP] need frames not met: {vp.name}")
                continue

            # 추론
            t0 = time.perf_counter()
            logits = model(x.to(device))
            prob = float(torch.sigmoid(logits).item())
            t1 = time.perf_counter()
            infer_ms = (t1 - t0) * 1000.0

            pred = 1 if prob >= 0.49 else 0

            y_true.append(gt)
            y_pred.append(pred)
            probs.append(prob)
            pre_times.append(pre_ms)
            inf_times.append(infer_ms)

            rows.append([vp.name, gt, pred, prob, pre_ms, infer_ms])
            print(f"{vp.name}: GT={gt} Pred={pred} Prob={prob:.4f} | pre={pre_ms:.1f}ms infer={infer_ms:.1f}ms")

    if len(y_true) == 0:
        print("[ERR] 평가 가능한 샘플이 없음.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs  = np.array(probs)
    pre_times = np.array(pre_times)
    inf_times = np.array(inf_times)

    # 혼동행렬/지표
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    acc = accuracy_score(y_true, y_pred)
    print("\n=== Confusion Matrix (rows=True, cols=Pred) ===")
    print("labels: 0=slow, 1=fast")
    print(cm)
    print("\n=== Accuracy ===")
    print(f"{acc*100:.2f}%")
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["slow","fast"], digits=4))

    # 시간 통계
    def stats(a):
        return float(np.mean(a)), float(np.median(a)), float(np.min(a)), float(np.max(a))
    p_mean, p_med, p_min, p_max = stats(pre_times)
    i_mean, i_med, i_min, i_max = stats(inf_times)
    print("\n=== Preprocess time (ms) ===")
    print(f"mean={p_mean:.2f}, median={p_med:.2f}, min={p_min:.2f}, max={p_max:.2f}")
    print("=== Inference time (ms) ===")
    print(f"mean={i_mean:.2f}, median={i_med:.2f}, min={i_min:.2f}, max={i_max:.2f}")

    # PNG 저장
    fig, ax = plt.subplots(figsize=(4.8,4.2))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix (counts)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["slow(0)","fast(1)"])
    ax.set_yticklabels(["slow(0)","fast(1)"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(args.cm_png, dpi=150)
    plt.close(fig)
    print(f"\n[INFO] Saved confusion matrix -> {args.cm_png}")

    # CSV 저장
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filename","label_true","label_pred","prob_fast","preprocess_ms","inference_ms"])
        wr.writerows(rows)
        wr.writerow([])
        wr.writerow(["# preprocess_ms_stats", f"mean={p_mean}", f"median={p_med}", f"min={p_min}", f"max={p_max}"])
        wr.writerow(["# inference_ms_stats",  f"mean={i_mean}", f"median={i_med}", f"in={i_min}", f"max={i_max}"])
        wr.writerow(["# overall_accuracy", acc])
    print(f"[INFO] Saved CSV -> {args.csv}")

if __name__ == "__main__":
    main()
