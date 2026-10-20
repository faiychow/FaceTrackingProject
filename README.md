# Real-Time Face Detection

Python script that detects faces from your webcam in real time using OpenCV's Haar cascade classifier. Each detected face gets a blue bounding box and a green crosshair at its center.

## How it works

1. Loads OpenCV's pretrained `haarcascade_frontalface_default.xml`
2. Grabs frames from the webcam and mirrors them
3. Converts each frame to grayscale (Haar cascades work better on it)
4. Runs `detectMultiScale` to find faces
5. Draws the box and center crosshair, then displays the frame

## Requirements

- Python 3.8+
- A webcam

```bash
pip install opencv-python
```

## Usage

```bash
python face_detection.py
```

Press `q` to quit.

## Tuning

In `detectMultiScale`:
- `scaleFactor` (1.1): smaller is more accurate but slower
- `minNeighbors` (5): higher means fewer false positives
- `minSize` (30, 30): smallest face size to detect, in pixels

## Limitations

Haar cascades work best on front-facing, well-lit faces. Side profiles, heavy shadows, and partial occlusion are often missed.

## Ideas for next steps

- Track multiple faces with IDs
- Swap in a DNN-based detector (e.g. MediaPipe) for better accuracy
