import argparse
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_WEIGHTS = PROJECT_DIR / "yolov8n.pt"
WINDOW_NAME = "YOLOv8 ByteTrack"


def parse_args():
    parser = argparse.ArgumentParser(description="Track objects in a webcam or video file.")
    parser.add_argument("--source", default="0", help="Camera index (default: 0) or path to a video file")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--classes", type=int, nargs="*", help="COCO class IDs to keep, for example: --classes 0")
    parser.add_argument("--device", default=None, help="Inference device, for example: cpu, 0, or mps")
    return parser.parse_args()


def resolve_source(source):
    if source.isdigit():
        return int(source)
    video_path = Path(source)
    if not video_path.is_absolute():
        video_path = PROJECT_DIR / video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    return str(video_path)


def main():
    args = parse_args()
    source = resolve_source(args.source)

    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS}")

    if isinstance(source, int):
        probe = cv2.VideoCapture(source)
        if not probe.isOpened():
            probe.release()
            raise RuntimeError(f"Could not open webcam at index {source}. Try --source 1.")
        probe.release()

    model = YOLO(str(MODEL_WEIGHTS))
    stream = model.track(
        source=source,
        stream=True,
        tracker="bytetrack.yaml",
        persist=True,
        conf=args.confidence,
        iou=args.iou,
        classes=args.classes,
        device=args.device,
        show=False,
        verbose=False,
    )

    previous_time = time.perf_counter()
    print("Press 'q' to quit.")

    try:
        for result in stream:
            frame = result.plot()
            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
