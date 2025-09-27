# split_video_by_keypress.py
import os
import sys
import cv2
from pathlib import Path

OUTPUT_DIR = "train_data/regression_model_train_data"
USE_FIXED_FPS = True     # True면 30fps로 강제 저장
FIXED_FPS = 30
FOURCC = "mp4v"

def main(video_path_str: str):
    src_path = Path(video_path_str)
    if not src_path.exists():
        print(f"[ERR] not found: {src_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"[ERR] cannot open: {src_path}")
        sys.exit(1)

    # 원본 메타
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 1e-6:
        src_fps = FIXED_FPS  # 읽기 실패 시 30 가정
    fps_out = FIXED_FPS if USE_FIXED_FPS else src_fps

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 준비
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = src_path.stem  # 확장자 제거한 원본 이름

    def open_writer(idx: int):
        out_name = f"{base}_clip_{idx}.mp4"
        out_path = out_dir / out_name
        fourcc = cv2.VideoWriter_fourcc(*FOURCC)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (w, h), True)
        return writer, out_path

    clip_idx = 1
    writer, cur_out_path = open_writer(clip_idx)
    if not writer.isOpened():
        print(f"[ERR] cannot open writer: {cur_out_path}")
        cap.release()
        sys.exit(1)

    frames_in_clip = 0
    written_total = 0

    print(f"[INFO] src='{src_path.name}', size={w}x{h}, src_fps≈{src_fps:.3f}, out_fps={fps_out}")
    print("[INFO] Controls: 'c' = cut & new clip, 'q' = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            # 영상 종료 → 마지막 클립 정리
            if frames_in_clip == 0:
                # 빈 클립 파일이면 삭제
                writer.release()
                try:
                    os.remove(cur_out_path)
                except OSError:
                    pass
            else:
                writer.release()
                print(f"[SAVE] {cur_out_path.name} ({frames_in_clip} frames)")
            break

        # 디스플레이 오버레이
        overlay = frame.copy()
        txt1 = f"{base} | clip #{clip_idx} | this clip frames: {frames_in_clip}"
        txt2 = "Press 'c' to cut, 'q' to quit"
        cv2.rectangle(overlay, (0, 0), (max(420, w//2), 50), (0,0,0), -1)
        cv2.putText(overlay, txt1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(overlay, txt2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
        cv2.imshow("splitter", overlay)

        # 저장
        writer.write(frame)
        frames_in_clip += 1
        written_total += 1

        key = cv2.waitKey(20) & 0xFF
        if key == ord('c'):
            # 현재 클립 종료
            writer.release()
            if frames_in_clip == 0:
                # 이론상 발생하기 어렵지만 방어
                try:
                    os.remove(cur_out_path)
                except OSError:
                    pass
            else:
                print(f"[CUT] saved {cur_out_path.name} ({frames_in_clip} frames)")

            # 다음 클립 시작
            clip_idx += 1
            writer, cur_out_path = open_writer(clip_idx)
            if not writer.isOpened():
                print(f"[ERR] cannot open writer: {cur_out_path}")
                break
            frames_in_clip = 0

        elif key == ord('q'):
            # 강제 종료 → 현재 진행 중인 클립 저장/정리
            writer.release()
            if frames_in_clip == 0:
                try:
                    os.remove(cur_out_path)
                except OSError:
                    pass
            else:
                print(f"[QUIT] saved {cur_out_path.name} ({frames_in_clip} frames)")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] total frames processed: {written_total}, clips created up to index {clip_idx}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_video_by_keypress.py <video_path>")
        sys.exit(1)
    main(sys.argv[1])
