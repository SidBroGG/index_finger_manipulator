import json
from pathlib import Path

import cv2
import numpy
from hand_tracker import HandTracker

def draw_index_ray(img, p1, p2):
    h, w = img.shape[:2]
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = numpy.hypot(dx, dy)

    if length == 0:
        return img

    diag = int(numpy.hypot(w, h))

    pt_end = (
        int(round(p1[0] + (dx / length) * diag)),
        int(round(p1[1] + (dy / length) * diag))
    )

    cv2.line(img, p1, pt_end, (0, 255, 0), 2)
    return img

def run():
    with open("calibration.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    mtx = numpy.array(data["mtx"], numpy.float32)
    dist = numpy.array(data["dist"], numpy.float32)

    cap = cv2.VideoCapture(0)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, new_mtx, (w, h), cv2.CV_16SC2
    )

    hand_tracker = HandTracker().start()

    while True:
        _, frame = cap.read()

        undisorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        hand_tracker.process_frame_async(undisorted_frame)
        landmarks = hand_tracker.get_landmark_pixel(undisorted_frame.shape)

        if landmarks:
            idx1, idx2 = HandTracker.INDEX_FINGER_LINE
            p1 = landmarks[idx1]
            p2 = landmarks[idx2]
            undisorted_frame = draw_index_ray(undisorted_frame, p1, p2)
            
        cv2.imshow("img", undisorted_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_tracker.close()
    cap.release()
    cv2.destroyAllWindows()
