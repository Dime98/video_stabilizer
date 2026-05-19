import cv2
import numpy as np

KERNEL_1X1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
KERNEL_3X3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_5X5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_7X7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
KERNEL_10X10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))


video_extensions = [
    ".mp4",
    ".mov",
    ".wmv",
    ".avi",
    ".avchd",
    ".flv",
    ".f4v",
    ".swf",
    ".mkv",
    ".webm",
    ".html5",
    ".mpeg-2",
    ".gif",
]


def rescale(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    if width <= 0 or height <= 0:
        print(f"[!] rescaled {dim=}")
        return frame
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def image_shape_by_format(format: str):
    format_by_shape = {"rgb": 3, "bgr": 3, "gray": 2}
    return format_by_shape[format]


def convert_image(image: np.ndarray, target_format: str = "gray"):
    target_format = target_format.casefold()
    target_format_shape = image_shape_by_format(target_format)
    if target_format_shape == len(image.shape):
        return image
    elif target_format == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif target_format in ["rbg", "bgr"]:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def get_video_writer(output_video_path, fps, width, height, fourcc="FFV1"):
    return cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )
