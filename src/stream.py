import json

import cv2
import numpy


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

    while True:
        _, frame = cap.read()

        undisorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        cv2.imshow("original", frame)
        cv2.imshow("undisorted", undisorted_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
