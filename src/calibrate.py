import json
import math
import time

import cv2
import numpy


def take_images(image_count: int) -> list:
    cap = cv2.VideoCapture(0)
    images = []

    time_left = 3
    next_second = math.floor(time.time()) + 1

    cv2.namedWindow("img")

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        if time.time() >= next_second:
            if time_left == 0:
                images.append(frame.copy())

                if len(images) == image_count:
                    break

                print(f"\n{image_count - len(images)} images left")
                time_left = 3
            else:
                print(f"shot in {time_left}")
                time_left -= 1

            next_second += 1

        cv2.imshow("img", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    return images


def run(chessboard: str, square_size: float, calibration_attempts: int):
    chess_rows, chess_cols = chessboard.split("x")
    chess_cols = int(chess_cols)
    chess_rows = int(chess_rows)

    # Критерии для ограничений по поиску углов
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = numpy.zeros((chess_rows * chess_cols, 3), numpy.float32)
    objp[:, :2] = numpy.mgrid[0:chess_cols, 0:chess_rows].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = take_images(calibration_attempts)
    gray_shape = None

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray,
            (chess_cols, chess_rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            objpoints.append(objp)

            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(refined_corners)

    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None, flags=cv2.CALIB_FIX_K3
    )

    print(f"calibration rms: {ret}")

    calibration_data = {}
    calibration_data["mtx"] = mtx.tolist()
    calibration_data["dist"] = dist.tolist()

    with open("calibration.json", mode="w", encoding="utf-8") as file:
        json.dump(calibration_data, file, ensure_ascii=False, indent=4)

    print("calibration complete")
