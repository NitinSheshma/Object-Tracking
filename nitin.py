import cv2
from ultralytics import YOLO

# ---- Settings you can tweak ----
MODEL_WEIGHTS = "yolov8n.pt"   # small & fast; try yolov8s.pt for better accuracy
CONF_THRESH   = 0.35           # detection confidence
IOU_THRESH    = 0.45           # NMS IoU threshold
CLASSES_KEEP  = None           # e.g., [0] to track only 'person'; None = all COCO classes
CAM_INDEX     = 0              # webcam index: try 1 or 2 if you have multiple cameras
WINDOW_NAME   = "YOLOv8 ByteTrack — Webcam"
# --------------------------------

# Load model (auto-downloads weights on first run)
# For CPU only, add: YOLO(MODEL_WEIGHTS).to('cpu') or pass device='cpu' in .track()
model = YOLO(MODEL_WEIGHTS)

# Open a quick probe to ensure the camera index is valid before handing to Ultralytics
probe = cv2.VideoCapture(CAM_INDEX)
if not probe.isOpened():
    print(f"Error: Could not open webcam at index {CAM_INDEX}. Try CAM_INDEX = 1 or 2.")
    raise SystemExit(1)
probe.release()

# Ultralytics tracking with ByteTrack. We render frames ourselves, so show=False.
# Notes:
# - 'persist=True' keeps track IDs across frames
# - 'tracker="bytetrack.yaml"' enables ByteTrack (bundled with Ultralytics)
stream = model.track(
    source=CAM_INDEX,
    stream=True,
    tracker="bytetrack.yaml",
    persist=True,
    conf=CONF_THRESH,
    iou=IOU_THRESH,
    classes=CLASSES_KEEP,
    show=False,
    verbose=False,
    # device="cpu",   # uncomment to force CPU
)

# Names for class IDs (COCO)
names = model.model.names if hasattr(model.model, "names") else {}

print("Press 'q' to quit.")
for result in stream:
    # result.plot() returns an annotated BGR frame with boxes, labels, and track IDs
    frame = result.plot()

    # Optionally, overlay simple FPS from Ultralytics timing if available
    if hasattr(result, "speed") and isinstance(result.speed, dict):
        fps = 1000.0 / max(result.speed.get("inference", 1.0), 1e-3)
        cv2.putText(frame, f"FPS ~ {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Optionally, draw your own labels with track IDs (already shown by plot(), but this shows how to access data)
    # boxes = result.boxes
    # if boxes is not None and boxes.id is not None:
    #     ids = boxes.id.int().tolist()
    #     for (xyxy, cls, tid) in zip(boxes.xyxy.cpu().numpy(), boxes.cls.int().tolist(), ids):
    #         x1, y1, x2, y2 = map(int, xyxy)
    #         label = f"{names.get(cls, str(cls))} #{tid}"
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(frame, label, (x1, max(20, y1 - 10)),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
