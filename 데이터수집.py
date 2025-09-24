# requirements:
#   pip install opencv-python
# 표준 라이브러리: tkinter, threading, time, datetime, pathlib

import cv2
import time
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

# ===== 설정 =====
OUTPUT_DIR = Path("clips")         # 클립 저장 폴더
CAM_INDEX = 1                      # 웹캠 인덱스
TARGET_FPS = 30                    # 파일 저장 FPS(웹캠 fps 불확실 대비)
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')  # .mp4

# ===== 전역 상태 =====
running = False                    # 캡처 스레드 실행 여부
recording = False                  # 실제 녹화 중 여부
cap = None
writer = None                      # 현재 라벨 파일에 쓰는 VideoWriter
current_label = None               # 현재 연속 라벨('fast' or 'slow')
clip_idx = 0                       # 파일 인덱스(라벨 바뀔 때만 증가)

frame_size = None                  # (w, h)
buffer_frames = []                 # "이전 체크포인트 이후" 프레임 버퍼
last_checkpoint_ts = None          # 마지막 체크포인트 시각 (time.time)
start_ts = None                    # 녹화 시작 시각

# UI
root = tk.Tk()
root.title("Webcam Recorder (fast/slow segmentation)")
root.geometry("380x160")

status_var = tk.StringVar(value="대기")
elapsed_var = tk.StringVar(value="경과: 0.00s")

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def open_camera():
    global cap, frame_size
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # 재시도(플랫폼별 드라이버 지연 대응)
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없음")

    # 해상도/FPS는 환경별로 다름. 기본 값 사용.
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("첫 프레임 캡처 실패")
    h, w = frame.shape[:2]
    frame_size = (w, h)

def close_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def open_new_writer(label):
    """라벨이 바뀌면 새 파일로 전환"""
    global writer, clip_idx, current_label
    clip_idx += 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = OUTPUT_DIR / f"{ts}_clip{clip_idx:03d}_{label}.mp4"
    writer = cv2.VideoWriter(str(fname), FOURCC, TARGET_FPS, frame_size)
    if not writer.isOpened():
        raise RuntimeError("VideoWriter 열기 실패")
    current_label = label

def close_writer():
    global writer
    if writer is not None:
        writer.release()
        writer = None

def reset_elapsed():
    global last_checkpoint_ts
    last_checkpoint_ts = time.time()

def on_start():
    global running, recording, buffer_frames, start_ts, last_checkpoint_ts, current_label, clip_idx
    if recording:
        return
    try:
        ensure_output_dir()
        open_camera()
    except Exception as e:
        messagebox.showerror("오류", str(e))
        return

    running = True
    recording = True
    buffer_frames = []
    current_label = None
    clip_idx = 0
    start_ts = time.time()
    reset_elapsed()
    status_var.set("녹화 중")
    btn_start.config(state="disabled")
    btn_checkpoint.config(state="normal")
    btn_stop.config(state="normal")

    threading.Thread(target=capture_loop, daemon=True).start()

def on_checkpoint():
    """버퍼를 라벨링해서 파일에 flush. 라벨은 '직전 체크포인트→지금' 간격으로 판정."""
    global buffer_frames, current_label, writer

    now = time.time()
    if last_checkpoint_ts is None:
        # 이론상 on_start에서 설정됨
        reset_elapsed()
        return

    interval = now - last_checkpoint_ts
    label = "fast" if interval >= 3.0 else "slow"

    # 같은 라벨이 연속되면 기존 파일에 이어쓰기
    if current_label != label:
        # 라벨 바뀌므로 이전 writer 닫고 새로
        if writer is not None:
            writer.release()
        open_new_writer(label)

    # 버퍼 flush
    if writer is not None and buffer_frames:
        for f in buffer_frames:
            writer.write(f)
    buffer_frames = []

    # 경과시간 리셋
    reset_elapsed()
    status_var.set(f"체크포인트 ({label}, {interval:.2f}s)")

def on_stop():
    """녹화 중지. 마지막 버퍼(마지막 체크포인트→중지)는 라벨 미정이라 폐기."""
    global running, recording, buffer_frames
    if not recording:
        return
    running = False
    recording = False

    # 남은 버퍼는 요구사항상 저장 안 함(라벨 없음)
    buffer_frames = []

    close_writer()
    close_camera()
    status_var.set("중지됨")
    btn_start.config(state="normal")
    btn_checkpoint.config(state="disabled")
    btn_stop.config(state="disabled")

def draw_overlay(frame):
    """화면에 경과시간 텍스트 오버레이"""
    if last_checkpoint_ts is None:
        elapsed = 0.0
    else:
        elapsed = max(0.0, time.time() - last_checkpoint_ts)


def capture_loop():
    """메인 캡처 루프: 프레임을 버퍼에 쌓고, 체크포인트 시 flush."""
    global running, cap, buffer_frames

    win_name = "Recorder"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # 타이밍 보정: TARGET_FPS에 맞춰 슬립(대략)
    frame_interval = 1.0 / TARGET_FPS

    while running and cap is not None and cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            # 카메라 글리치 시 잠깐 대기 후 시도
            time.sleep(0.01)
            continue

        draw_overlay(frame)

        # 현재 구간 버퍼에 쌓기
        buffer_frames.append(frame.copy())

        # 미리보기
        cv2.imshow(win_name, frame)
        # 키보드에서 'q'나 ESC로 비상종료 가능(선택)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            root.after(0, on_stop)
            break

        # FPS 근사 유지
        dt = time.time() - t0
        if dt < frame_interval:
            time.sleep(frame_interval - dt)

    cv2.destroyWindow(win_name)

# ===== Tkinter UI =====
btn_start = tk.Button(root, text="시작", width=12, command=on_start)
btn_checkpoint = tk.Button(root, text="체크포인트", width=12, state="disabled", command=on_checkpoint)
btn_stop = tk.Button(root, text="중지", width=12, state="disabled", command=on_stop)

lbl_status = tk.Label(root, textvariable=status_var, anchor="w")
lbl_elapsed = tk.Label(root, textvariable=elapsed_var, anchor="w")

btn_start.pack(pady=8)
btn_checkpoint.pack(pady=4)
btn_stop.pack(pady=4)
lbl_status.pack(pady=6, anchor="w", padx=10)
lbl_elapsed.pack(pady=2, anchor="w", padx=10)

def on_closing():
    if recording:
        if messagebox.askokcancel("종료", "녹화를 중지하고 종료할까?"):
            on_stop()
            root.destroy()
    else:
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

if __name__ == "__main__":
    root.mainloop()
