import typer

import calibrate as calib


def main(
    calibrate: bool = False,
    chessboard: str = "9x6",
    square_size: float = 15.0,
    calibration_attempts: int = 5,
):
    if calibrate:
        calib.run(chessboard, square_size, calibration_attempts)


if __name__ == "__main__":
    typer.run(main)
