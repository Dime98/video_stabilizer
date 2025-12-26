from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    video_path: str
    frames_count: int
    width: int
    height: int
    fps: float

    @classmethod
    def from_path(cls, video_path):
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise ValueError(f"{video_capture.isOpened()=}\n{video_path=}")

        metadata = cls(
            video_path=video_path,
            frames_count=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=video_capture.get(cv2.CAP_PROP_FPS),
        )

        video_capture.release()
        return metadata

    def __repr__(self):
        print(f"wxh: {self.width} {self.height}")
        print(f"fps: {self.fps}")
