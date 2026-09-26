import json
import time

import cv2
import numpy


def take_images(image_count: int) -> list:
    cap = cv2.VideoCapture(0)
    images = []

    for i in range(image_count):
        print(f"\n{image_count - i} images left")

        for j in range(4):
            print(f"camera shot in {3 - j}")
            time.sleep(1)

        # Пробные снимки для экспозиции и баланса белого
        for _ in range(5):
            cap.grab()

        ret, frame = cap.read()

        if ret:
            images.append(frame)

    cap.release()
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

        ret, corners = cv2.findChessboardCorners(gray, (chess_cols, chess_rows), None)

        if ret:
            objpoints.append(objp)

            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(refined_corners)

    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    print(f"calibration rms: {ret}")

    calibration_data = {}
    calibration_data["mtx"] = mtx.tolist()
    calibration_data["dist"] = dist.tolist()

    with open("calibration.json", mode="w", encoding="utf-8") as file:
        json.dump(calibration_data, file, ensure_ascii=False, indent=4)

    print("calibration complete")
