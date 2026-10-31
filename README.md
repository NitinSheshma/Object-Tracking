# Object Tracking

A real-time object detection and tracking application using Python, OpenCV, YOLOv8, and ByteTrack.

## Features

- Real-time tracking from a webcam
- Tracking from a local video file
- Persistent object IDs with ByteTrack
- Optional class filtering
- Measured display FPS

## Requirements

- Python 3.10â€“3.12
- A working webcam, or a local video file

## Installation

```powershell
cd Object-Tracking
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Track the default webcam:

```powershell
python nitin.py
```

Use another camera:

```powershell
python nitin.py --source 1
```

Track a video file:

```powershell
python nitin.py --source path\to\video.mp4
```

Track only people (COCO class 0):

```powershell
python nitin.py --classes 0
```

Press `q` while the video window is active to stop the program.

## Main options

- `--source`: Camera index or video path (default: `0`)
- `--confidence`: Detection confidence threshold (default: `0.35`)
- `--iou`: NMS IoU threshold (default: `0.45`)
- `--classes`: Space-separated COCO class IDs
- `--device`: Device such as `cpu`, `0`, or `mps`

Run `python nitin.py --help` for all options.

