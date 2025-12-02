
"""
Arvyax Technologies - Hand Tracking POC (classical CV, CPU-only)

Run:
    python main.py

Requirements:
    - Python 3.8+
    - OpenCV (cv2)
    - numpy

This script:
 - captures webcam frames
 - segments a hand using a combined HSV + YCrCb skin color mask
 - finds the largest contour (assumed hand)
 - computes the fingertip as the contour point furthest from the contour centroid
 - draws a virtual rectangular boundary and computes the minimum distance between the hand contour and that rectangle
 - classifies into SAFE / WARNING / DANGER and overlays "DANGER DANGER" during danger
 - aims for >= 8 FPS on CPU by resizing frame and using optimized operations
"""
import cv2
import numpy as np
import time

# Parameters
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_CONTOUR_AREA = 2000  # ignore small contours
# Virtual object: rectangle (x1,y1)-(x2,y2) as a fraction of frame size (centered)
RECT_W_FRAC = 0.3
RECT_H_FRAC = 0.25
# Distance thresholds (in pixels) -- will be scaled with frame diagonal
SAFE_THRESHOLD_RATIO = 0.35
WARNING_THRESHOLD_RATIO = 0.12

def skin_mask(frame):
    """Return a binary mask where skin-like pixels are white."""
    # Convert to HSV and YCrCb and combine thresholds for robustness
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # HSV skin range
    lower_hsv = np.array([0, 30, 60])
    upper_hsv = np.array([25, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # YCrCb skin range
    lower_ycrcb = np.array([0, 133, 77])
    upper_ycrcb = np.array([255, 173, 127])
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)

    # Morphology to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    return mask

def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # choose largest by area
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
        return None
    return cnt

def contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

def fingertip_point(cnt, centroid):
    # choose contour point with maximum distance from centroid
    pts = cnt.reshape(-1,2)
    dists = np.linalg.norm(pts - np.array(centroid), axis=1)
    idx = np.argmax(dists)
    return tuple(pts[idx].tolist())

def rect_coords(frame_w, frame_h):
    rw = int(frame_w * RECT_W_FRAC)
    rh = int(frame_h * RECT_H_FRAC)
    cx = frame_w // 2
    cy = frame_h // 2
    x1 = cx - rw//2
    y1 = cy - rh//2
    x2 = cx + rw//2
    y2 = cy + rh//2
    return (x1,y1,x2,y2)

def point_rect_distance(px, py, rx1, ry1, rx2, ry2):
    # distance from point to rectangle (0 if inside)
    dx = max(rx1 - px, 0, px - rx2)
    dy = max(ry1 - py, 0, py - ry2)
    return np.hypot(dx, dy)

def contour_rect_min_distance(cnt, rect):
    # compute minimum distance from any contour point to rect
    pts = cnt.reshape(-1,2)
    dists = [point_rect_distance(int(x), int(y), *rect) for (x,y) in pts]
    return min(dists) if dists else float('inf')

def draw_overlay(frame, rect, state, fingertip=None, centroid=None, fps=0.0):
    x1,y1,x2,y2 = rect
    # color by state
    if state == "SAFE":
        color = (0,255,0)
    elif state == "WARNING":
        color = (0,165,255)
    else:
        color = (0,0,255)
    # rectangle thicker when danger
    thickness = 2 if state!="DANGER" else 4
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

    # draw centroid and fingertip
    if centroid:
        cv2.circle(frame, centroid, 5, (255,0,0), -1)
    if fingertip:
        cv2.circle(frame, fingertip, 8, (255,255,255), -1)
        cv2.circle(frame, fingertip, 12, color, 2)

    # state text
    cv2.putText(frame, f"State: {state}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-130,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    # Danger overlay
    if state == "DANGER":
        txt = "DANGER DANGER"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)
        tx = (frame.shape[1] - tw)//2
        ty = 60
        # blinking effect based on time
        if int(time.time()*2) % 2 == 0:
            cv2.putText(frame, txt, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 3)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    prev = time.time()
    fps = 0.0
    # compute thresholds in pixels based on diagonal
    diag = np.hypot(FRAME_WIDTH, FRAME_HEIGHT)
    safe_thresh = SAFE_THRESHOLD_RATIO * diag
    warn_thresh = WARNING_THRESHOLD_RATIO * diag

    rect = rect_coords(FRAME_WIDTH, FRAME_HEIGHT)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        # optionally resize for speed (already set), and blur slightly
        frame_blur = cv2.GaussianBlur(frame, (5,5), 0)
        mask = skin_mask(frame_blur)
        cnt = largest_contour(mask)
        state = "SAFE"
        fingertip = None
        centroid = None
        if cnt is not None:
            centroid = contour_centroid(cnt)
            if centroid is not None:
                fingertip = fingertip_point(cnt, centroid)
                min_dist = contour_rect_min_distance(cnt, rect)
                # classify
                if min_dist <= warn_thresh * 0.3:
                    state = "DANGER"
                elif min_dist <= warn_thresh:
                    state = "WARNING"
                else:
                    state = "SAFE"
                # draw contour
                cv2.drawContours(frame, [cnt], -1, (150,150,150), 1)
        # compute fps smoothing
        t1 = time.time()
        dt = t1 - prev
        prev = t1
        if dt > 0:
            fps = 0.9*fps + 0.1*(1.0/dt)
        draw_overlay(frame, rect, state, fingertip, centroid, fps)
        # show mask in small window
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        small_mask = cv2.resize(mask_rgb, (200,150))
        # compose side-by-side
        h1, w1 = frame.shape[:2]
        frame[0:150, w1-200:w1] = small_mask
        cv2.imshow("Arvyax Hand POC", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
