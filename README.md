
# Arvyax - Hand Tracking POC (Classical Computer Vision)

**Objective:** Prototype a real-time hand/fingertip tracking system that detects when a hand approaches a virtual object and displays `DANGER DANGER` on-screen when too close. This solution uses only classical computer vision techniques (no MediaPipe/OpenPose or cloud APIs).

## Features
- Real-time hand segmentation using combined HSV + YCrCb skin color thresholds
- Largest-contour based hand detection
- Fingertip estimated as the contour point farthest from the centroid
- Virtual rectangular boundary in the center of the screen
- Distance-based classification: SAFE / WARNING / DANGER
- Visual overlays including `DANGER DANGER` during danger
- CPU-friendly (frame size 640x480) and optimized operations (aims ≥ 8 FPS)

## Files
- `main.py` - main application script (run with `python main.py`)
- `requirements.txt` - Python dependencies

## How to run
1. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate    # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python main.py
   ```
4. Press `q` or `Esc` to quit.

## Notes & Tips
- Lighting and background affect skin segmentation. Prefer a plain background and stable lighting.
- If segmentation performs poorly, try adjusting HSV and YCrCb ranges in `skin_mask()` in `main.py`.
- For improved robustness you can:
  - Add background subtraction (MOG2) and combine with skin mask
  - Train a small custom classifier on webcam patches (if allowed)
  - Use contour convexity defects to detect fingertips more precisely
- The project avoids MediaPipe/OpenPose per assignment restrictions.

## License
MIT
