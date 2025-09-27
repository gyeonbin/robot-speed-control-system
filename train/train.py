# speed_conv3d_autorun_es_struct_memlite_tqdm.py
# -------------------------------------------------------------
# - 라벨: 파일명에 fast → 1, slow → 0 (breakfast 오탐 방지)
# - 입력: RGB(3) + Gray-Δt(1) = 4채널, 고정길이 윈도우
# - 프레임 전처리: 디코딩 직후 **img_size×img_size로 즉시 리사이즈** (중간 1600×900 제거)
# - 분할: 앞 60초 → train, 이어지는 20초 → val (시간 분리)
# - 탐색: Optuna (F1 기준, EarlyStop, trial=30, epoch=5)
# - 모델: 소형 3D CNN
#         구조 탐색 포함: N_BLOCKS{1,2,3}, POOL_T_PATTERN{none,b1,b1b2},
#                       KT_MODE{3x3x3, 3x3x1+stride_t2}, DROPOUT{0,0.1,0.3},
#                       WIDTH_MULT{1.0,1.25,1.5}
#         (※ TEMP_STRIDE=2, SAMPLE_STRIDE=8, ACT=ReLU 고정)
# - EarlyStopping: patience=3, monitor=val F1
# - 실행: python speed_conv3d_autorun_es_struct_memlite_tqdm.py
# - 출력: models/speed_conv3d_best.pt, models/meta_speed_conv3d.json, optuna_best_f1.json
# -------------------------------------------------------------

import os, re, json, glob, random
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from tqdm.auto import tqdm

# ===== 기본 설정 =====
FPS = 30
WINDOW_SEC = 1.0
TEMP_STRIDE = 2              # (탐색 제외, 고정)
WINDOW = max(2, int(round(WINDOW_SEC * FPS / TEMP_STRIDE)))

SAMPLE_STRIDE = 8            # (탐색 제외, 고정)
IMG_SIZE = 112               # (Optuna가 바꾸는 값)
BATCH_SIZE = 32
LR = 4e-3
EPOCHS = 5
NUM_WORKERS = 4              # 요청 반영
WEIGHT_DECAY = 1e-4
SEED = 42
N_TRIALS = 30                # 요청: 30회

# 시간 분할
TRAIN_SEC = 60
VAL_SEC   = 20
MAX_DURATION_SEC = 80

# normalize
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DIF_MEAN, DIF_STD = 0.0, 0.5

# 파일명 라벨 정규식
FAST_RE = re.compile(r'(^|[^a-z])fast([^a-z]|$)')
SLOW_RE = re.compile(r'(^|[^a-z])slow([^a-z]|$)')

# EarlyStopping
ES_PATIENCE = 3
ES_MIN_DELTA = 0.0  # F1이 이 값보다 더 좋아져야 '개선'으로 판단

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===== 라벨 판정 =====
def label_from_name(path: str) -> int:
    base = os.path.basename(path).lower()
    if FAST_RE.search(base) and "breakfast" not in base:
        return 1
    if SLOW_RE.search(base):
        return 0
    return -1

# ===== 전처리 =====
def normalize_rgb(rgb_float01: np.ndarray) -> np.ndarray:
    img = (rgb_float01 - MEAN) / STD
    return np.transpose(img.astype(np.float32), (2, 0, 1))  # (3,H,W)

def to_gray01(rgb_uint8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return (gray.astype(np.float32) / 255.0)

def norm_diff(diff: np.ndarray) -> np.ndarray:
    return (diff - DIF_MEAN) / max(1e-6, DIF_STD)

# ===== 윈도우 샘플 생성 (시간 분할) =====
def build_window_samples_by_time(
    video_files: List[str],
    window: int,
    stride: int,
    sample_stride: int,
    train_sec: int = TRAIN_SEC,
    val_sec: int   = VAL_SEC,
    max_duration_sec: int = MAX_DURATION_SEC
):
    train_samples, val_samples = [], []
    need = (window - 1) * stride + 1

    for vp in video_files:
        lb = label_from_name(vp)
        if lb not in (0, 1):
            continue

        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if not fps or fps <= 1e-6 or np.isnan(fps):
            fps = FPS

        max_frames = int(round(fps * max_duration_sec))
        eff_total = min(total, max_frames)

        train_end_f = min(eff_total, int(round(train_sec * fps)))
        val_start_f = train_end_f
        val_end_f   = min(eff_total, val_start_f + int(round(val_sec * fps)))

        # train
        if train_end_f >= need:
            for s in range(0, train_end_f - need + 1, sample_stride):
                train_samples.append((vp, s, lb))
        # val
        if val_end_f - val_start_f >= need:
            for s in range(val_start_f, val_end_f - need + 1, sample_stride):
                val_samples.append((vp, s, lb))

    return train_samples, val_samples

# ===== Dataset =====
class OnTheFlyVideoDataset(Dataset):
    """샘플: (video_path, start_idx, label)
       접근 시 디스크에서 읽고 **즉시 (IMG_SIZE, IMG_SIZE)로 리사이즈** → 전처리"""
    def __init__(self, samples: List[Tuple[str,int,int]], window: int, stride: int, img_size: int):
        self.samples = samples
        self.window = window
        self.stride = stride
        self.img_size = img_size

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
            ok, frame_bgr = cap.read()

            if ok:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                rgb01 = (rgb.astype(np.float32) / 255.0)
                gray01 = to_gray01(rgb)
                proc_rgb = normalize_rgb(rgb01)
            else:
                proc_rgb = clip_rgb[-1] if clip_rgb else np.zeros((3, self.img_size, self.img_size), np.float32)
                gray01   = last_gray if last_gray is not None else np.zeros((self.img_size, self.img_size), np.float32)

            if last_gray is None:
                diff = np.zeros_like(gray01, dtype=np.float32)
            else:
                diff = np.abs(gray01 - last_gray)
            diff = norm_diff(diff)

            clip_rgb.append(proc_rgb)            # (3,H,W)
            clip_dif.append(diff[None, ...])     # (1,H,W)
            last_gray = gray01

        cap.release()

        x_rgb = np.stack(clip_rgb, axis=1)   # (3,T,H,W)
        x_dif = np.stack(clip_dif, axis=1)   # (1,T,H,W)
        x = np.concatenate([x_rgb, x_dif], axis=0)  # (4,T,H,W)
        y = np.array([label], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

# ===== 활성함수: 고정(ReLU) =====
def get_act(name: str):
    return nn.ReLU(inplace=True)

# ===== 모델 (구조 탐색 포함) =====
class Small3DCNN(nn.Module):
    def __init__(
        self,
        in_ch=4,
        width_mult=1.0,
        img_size=None,             # 내부 interpolate 비활성화 (프레임이 이미 img_size)
        n_blocks=3,                # {1,2,3}
        pool_t_pattern="b1",       # {"none","b1","b1b2"}
        kt_mode="3x3x3",           # {"3x3x3","3x3x1+stride_t2"}
        dropout_p=0.0,             # {0.0,0.1,0.3}
        act_name="relu"            # 고정
    ):
        super().__init__()
        self.img_size = img_size
        self.act = get_act(act_name)

        c1 = int(round(16 * width_mult))
        c2 = int(round(32 * width_mult))
        c3 = int(round(64 * width_mult))
        chans_all = [c1, c2, c3]
        chans = chans_all[:max(1, min(3, n_blocks))]

        blocks = []
        in_c = in_ch
        for i, out_c in enumerate(chans):
            if kt_mode == "3x3x3":
                k = (3,3,3); s = (1,1,1)
            else:  # "3x3x1+stride_t2"
                k = (3,3,1); s = (2,1,1) if i > 0 else (1,1,1)

            blocks += [
                nn.Conv3d(in_c, out_c, kernel_size=k, stride=s, padding=(k[0]//2,1,1), bias=False),
                nn.BatchNorm3d(out_c),
                self.act,
            ]

            do_pool_t = (pool_t_pattern == "b1" and i == 0) or (pool_t_pattern == "b1b2" and i <= 1)
            pt = 2 if do_pool_t else 1
            blocks += [nn.MaxPool3d(kernel_size=(pt, 2, 2))]

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
        z = self.backbone(x)
        z = self.head(z)
        return z  # (B,1)

# ===== 평가(F1) =====
@torch.no_grad()
def eval_metrics_f1(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.squeeze(1).float().to(device)
        p = torch.sigmoid(model(x)).squeeze(1)
        y_true.append(y.cpu().numpy())
        y_prob.append(p.cpu().numpy())

    if not y_true:
        return {"f1":0.0, "best_thr":0.5, "pr_auc":0.0, "auroc":0.5, "acc":0.0}

    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = (2*precisions*recalls) / (precisions+recalls+1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_f1 = float(f1s[best_idx])
    best_thr = float(thresholds[best_idx-1]) if best_idx>0 and best_idx-1 < len(thresholds) else 0.5

    try: pr_auc = average_precision_score(y_true, y_prob)
    except Exception: pr_auc = 0.0
    try: auroc = roc_auc_score(y_true, y_prob)
    except Exception: auroc = 0.5
    acc = float(((y_prob>=0.5)==(y_true>0.5)).mean())
    return {"f1":best_f1, "best_thr":best_thr, "pr_auc":float(pr_auc), "auroc":float(auroc), "acc":acc}

# ===== Early Stopper =====
class EarlyStopper:
    def __init__(self, patience=ES_PATIENCE, min_delta=ES_MIN_DELTA, mode='max'):
        assert mode in ('max', 'min')
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -np.inf if mode == 'max' else np.inf
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        if self.mode == 'max':
            improved = metric > (self.best + self.min_delta)
        else:
            improved = metric < (self.best - self.min_delta)
        if improved:
            self.best = metric
            self.bad_epochs = 0
            return False
        else:
            self.bad_epochs += 1
            return self.bad_epochs > self.patience

# ===== 실행 1회 (학습+검증, EarlyStop) =====
def run_once(cfg: Dict) -> Dict:
    global WINDOW_SEC, TEMP_STRIDE, WINDOW, SAMPLE_STRIDE, IMG_SIZE
    global BATCH_SIZE, LR, EPOCHS, NUM_WORKERS, WEIGHT_DECAY
    global TRAIN_SEC, VAL_SEC, MAX_DURATION_SEC

    WINDOW_SEC     = cfg.get("WINDOW_SEC", WINDOW_SEC)
    TEMP_STRIDE    = cfg.get("TEMP_STRIDE", TEMP_STRIDE)     # 고정으로 유지
    WINDOW         = max(2, int(round(WINDOW_SEC * FPS / TEMP_STRIDE)))
    SAMPLE_STRIDE  = cfg.get("SAMPLE_STRIDE", SAMPLE_STRIDE) # 고정으로 유지
    IMG_SIZE       = cfg.get("IMG_SIZE", IMG_SIZE)
    BATCH_SIZE     = cfg.get("BATCH_SIZE", BATCH_SIZE)
    LR             = cfg.get("LR", LR)
    EPOCHS         = cfg.get("EPOCHS", EPOCHS)
    WEIGHT_DECAY   = cfg.get("WEIGHT_DECAY", WEIGHT_DECAY)
    TRAIN_SEC      = cfg.get("TRAIN_SEC", TRAIN_SEC)
    VAL_SEC        = cfg.get("VAL_SEC", VAL_SEC)
    MAX_DURATION_SEC = cfg.get("MAX_DURATION_SEC", MAX_DURATION_SEC)

    tqdm_desc = cfg.get("TQDM_DESC", "")

    set_seed(cfg.get("SEED", SEED))

    exts = ("*.mp4","*.avi","*.mov","*.mkv")
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join("../clips", ext)))
    video_files = sorted(video_files)
    if not video_files:
        raise FileNotFoundError("clips 폴더에 영상이 없습니다. (fast_*.mp4, slow_*.mp4 등)")

    tr_samps, va_samps = build_window_samples_by_time(
        video_files, WINDOW, TEMP_STRIDE, SAMPLE_STRIDE,
        TRAIN_SEC, VAL_SEC, MAX_DURATION_SEC
    )
    if len(tr_samps)==0 or len(va_samps)==0:
        raise RuntimeError(f"윈도우 부족: train={len(tr_samps)}, val={len(va_samps)}")

    random.shuffle(tr_samps); random.shuffle(va_samps)

    tr_ds = OnTheFlyVideoDataset(tr_samps, window=WINDOW, stride=TEMP_STRIDE, img_size=IMG_SIZE)
    va_ds = OnTheFlyVideoDataset(va_samps, window=WINDOW, stride=TEMP_STRIDE, img_size=IMG_SIZE)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=False)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.cuda.is_available()

    labels = [lb for _,_,lb in tr_ds.samples]
    n_pos = sum(1 for lb in labels if lb==1)
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([(n_neg/max(1,n_pos))], device=device, dtype=torch.float32)

    model = Small3DCNN(
        in_ch=4,
        width_mult=cfg.get("WIDTH_MULT", 1.0),
        img_size=None,  # 내부 보간 비활성화 (이미 데이터셋에서 img_size 맞춤)
        n_blocks=cfg.get("N_BLOCKS", 3),
        pool_t_pattern=cfg.get("POOL_T_PATTERN", "b1"),
        kt_mode=cfg.get("KT_MODE", "3x3x3"),
        dropout_p=cfg.get("DROPOUT_P", 0.0),
        act_name="relu"  # 고정
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state = None
    es = EarlyStopper(patience=ES_PATIENCE, min_delta=ES_MIN_DELTA, mode='max')

    epoch_iter = tqdm(range(1, EPOCHS+1), desc=tqdm_desc, leave=False)
    for e in epoch_iter:
        model.train()
        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logit = model(x)
                loss = loss_fn(logit, y)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()

        # ---- Validation (F1) & EarlyStopping ----
        m = eval_metrics_f1(model, va_loader, device)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        epoch_iter.set_postfix({"val_f1": f"{m['f1']:.4f}"})

        if es.step(m["f1"]):
            break

    epoch_iter.close()
    return {"best_f1": best_f1, "state_dict": best_state}

# ===== Optuna 튜닝 & 바로 학습 (tqdm 포함) =====
def autorun():
    import optuna
    from optuna.samplers import TPESampler

    def objective(trial: optuna.Trial):
        tdesc = f"trial {trial.number+1}/{N_TRIALS} (5 epochs)"
        cfg = {
            "SEED": 42,
            "EPOCHS": 5,  # 고정

            # 데이터/학습 (탐색 축소: img_size, batch, lr, wd만)
            "IMG_SIZE": trial.suggest_categorical("img_size", [80,96,112,128,144]),
            "BATCH_SIZE": trial.suggest_categorical("batch_size", [8,16,32]),
            "LR": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "WEIGHT_DECAY": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),

            # 고정값
            "TEMP_STRIDE": TEMP_STRIDE,
            "SAMPLE_STRIDE": SAMPLE_STRIDE,
            "WINDOW_SEC": 1.0,
            "MAX_DURATION_SEC": 80,
            "TRAIN_SEC": 60,
            "VAL_SEC": 20,

            # 구조 탐색 유지 (n_blocks 1~3 확장)
            "WIDTH_MULT": trial.suggest_categorical("width_mult", [1.0, 1.25, 1.5]),
            "N_BLOCKS": trial.suggest_categorical("n_blocks", [1, 2, 3]),
            "POOL_T_PATTERN": trial.suggest_categorical("pool_t_pat", ["none","b1","b1b2"]),
            "KT_MODE": trial.suggest_categorical("kt_mode", ["3x3x3","3x3x1+stride_t2"]),
            "DROPOUT_P": trial.suggest_categorical("dropout_p", [0.0, 0.1, 0.3]),

            # 진행률 라벨
            "TQDM_DESC": tdesc,
        }
        out = run_once(cfg)
        return out["best_f1"]

    set_seed(42)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42, multivariate=True, group=True),
        study_name="speed_conv3d_f1_time_split_es_struct_memlite_tqdm",
    )

    pbar_trials = tqdm(total=N_TRIALS, desc="optuna trials (30)")
    def _cb(study_, trial_):
        pbar_trials.update(1)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_cb])
    pbar_trials.close()

    print("\n[OPTUNA] Best F1:", study.best_value)
    print("[OPTUNA] Best params:", study.best_params)

    with open("../models/optuna_best_f1.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)

    # 최적 파라미터로 학습 & 저장 (진행률 바 포함)
    params = study.best_params
    cfg = {
        "SEED": 42,
        "EPOCHS": 5,
        "IMG_SIZE": params["img_size"],
        "BATCH_SIZE": params["batch_size"],
        "LR": params["lr"],
        "WEIGHT_DECAY": params["weight_decay"],

        "TEMP_STRIDE": TEMP_STRIDE,
        "SAMPLE_STRIDE": SAMPLE_STRIDE,
        "WINDOW_SEC": 1.0,
        "MAX_DURATION_SEC": 80,
        "TRAIN_SEC": 60,
        "VAL_SEC": 20,

        "WIDTH_MULT": params["width_mult"],
        "N_BLOCKS": params["n_blocks"],
        "POOL_T_PATTERN": params["pool_t_pat"],
        "KT_MODE": params["kt_mode"],
        "DROPOUT_P": params["dropout_p"],

        "TQDM_DESC": "final train (5 epochs)",
    }
    out = run_once(cfg)

    os.makedirs("../models", exist_ok=True)
    torch.save(out["state_dict"], os.path.join("../models", "speed_conv3d_best.pt"))
    meta = {
        "fps": FPS,
        "window_sec": cfg["WINDOW_SEC"],
        "window": max(2, int(round(cfg["WINDOW_SEC"]*FPS/cfg["TEMP_STRIDE"]))),
        "temp_stride": cfg["TEMP_STRIDE"],
        "sample_stride": cfg["SAMPLE_STRIDE"],
        "img_size": cfg["IMG_SIZE"],
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "diff_norm": {"mean": DIF_MEAN, "std": DIF_STD},
        "label_def": {"slow": 0, "fast": 1},
        "in_channels": 4,
        "time_split": {"train_sec": TRAIN_SEC, "val_sec": VAL_SEC, "max_duration_sec": MAX_DURATION_SEC},
        "model": {
            "width_mult": cfg["WIDTH_MULT"],
            "n_blocks": cfg["N_BLOCKS"],
            "pool_t_pattern": cfg["POOL_T_PATTERN"],
            "kt_mode": cfg["KT_MODE"],
            "dropout_p": cfg["DROPOUT_P"],
            "act": "relu",
        },
        "best_val_f1": out["best_f1"],
        "early_stopping": {"patience": ES_PATIENCE, "min_delta": ES_MIN_DELTA, "monitor": "val_f1"},
        "note": "Frames resized directly to (img_size,img_size) at decode time for memory saving.",
    }
    with open(os.path.join("../models", "meta_speed_conv3d.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved models/speed_conv3d_best.pt and meta_speed_conv3d.json | best F1 = {out['best_f1']:.4f}")

if __name__ == "__main__":
    autorun()
