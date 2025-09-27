# export_clips_1s_fixed30fps.py
import os
import glob
import cv2

INPUT_DIR = "../test"  # test/fast, test/slow
OUTPUT_DIR = "../test/test_data"  # 결과 저장 폴더
CLASSES = ["fast", "slow"]  # 처리할 라벨 폴더
TARGET_SEC = 100            # 0초 ~ 100초(미포함) → 1초 클립 100개
CLIP_LEN_SEC = 1            # 1초 클립
FIXED_FPS = 30              # 무조건 30fps로 저장
VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_videos(class_dir):
    paths = []
    for ext in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(class_dir, ext)))
    return sorted(paths)

def cut_1s_clips(video_path, label, start_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames = min(total_frames, FIXED_FPS * TARGET_SEC)  # 앞 100초만
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    clips_written = 0
    idx = start_idx

    for s in range(TARGET_SEC):  # 0~99 → 100개 클립
        start_frame = s * FIXED_FPS
        end_frame_excl = (s + 1) * FIXED_FPS
        if end_frame_excl > max_frames:
            break

        out_name = f"{label}_{idx}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        writer = cv2.VideoWriter(out_path, fourcc, FIXED_FPS, (w, h), True)
        if not writer.isOpened():
            print(f"[WARN] cannot open writer: {out_path}")
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cur = start_frame
        while cur < end_frame_excl:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            cur += 1

        writer.release()
        clips_written += 1
        idx += 1

    cap.release()
    print(f"[OK] {video_path} -> {clips_written} clips ({label}_{start_idx} ~ {label}_{idx-1})")
    return clips_written

def main():
    counters = {lbl: 1 for lbl in CLASSES}
    for lbl in CLASSES:
        class_dir = os.path.join(INPUT_DIR, lbl)
        vids = list_videos(class_dir)
        if not vids:
            print(f"[INFO] no videos in {class_dir}")
            continue
        for vp in vids:
            start_idx = counters[lbl]
            n = cut_1s_clips(vp, lbl, start_idx)
            counters[lbl] += n

    print("[DONE]")
    for lbl in CLASSES:
        print(f"  {lbl}: saved {counters[lbl]-1} clips to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
