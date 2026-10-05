from pathlib import Path
import time
from typing import Tuple, List, Optional

import cv2
import numpy
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
    INDEX_FINGER_LINE = (0, 8)

    def __init__(self):
        self._last_timestamp_ms: int = 0
        self._landmarker: Optional[vision.HandLandmarker] = None
        self._latest_result: Optional[vision.HandLandmarkerResult] = None


    def _result_callback(self, result: vision.HandLandmarkerResult, output_image: mediapipe.Image, timestamp_ms: int):
        self._latest_result = result

    def start(self):
        options = vision.HandLandmarkerOptions(
            base_options = python.BaseOptions(model_asset_path = str(self.MODEL_PATH)),
            running_mode = vision.RunningMode.LIVE_STREAM,
            result_callback = self._result_callback
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        return self

    def close(self):
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def process_frame_async(self, frame_bgr: numpy.ndarray):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mediapipe.Image(mediapipe.ImageFormat.SRGB, frame_rgb)

        current_timestamp_ms = int(time.time() * 1000)
        if current_timestamp_ms <= self._last_timestamp_ms:
            current_timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_timestamp_ms

        self._landmarker.detect_async(mp_image, current_timestamp_ms)

    def get_landmark_pixel(self, frame_shape: Tuple[int, int, ...]) -> Optional[List[Tuple[int, int]]]:
        h, w = frame_shape[:2]
        if not self._latest_result or not self._latest_result.hand_landmarks:
            return None

        first_hand = self._latest_result.hand_landmarks[0]
        return [(int(pt.x * w), int(pt.y * h)) for pt in first_hand]
            
        



    

    

