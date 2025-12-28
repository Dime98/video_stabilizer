from pathlib import Path

import cv2

from misc_functions.utils import get_video_writer
from video_stabilizer.rendering_ui.display import DisplayManager
from video_stabilizer.trackers.trackers import Tracker
from video_stabilizer.video.video_metadata import VideoMetadata


class VideoRenderer:
    def __init__(
            self,
            video_metadata: VideoMetadata,
            display_manager: DisplayManager,
            fourcc: str = "FFV1",
    ):
        self.display_manager = display_manager
        self.video_metadata = video_metadata

        self.fourcc_mapping = {
            "FFV1": ".avi",
            "XVID": ".mp4",
            "MJPG": ".avi",
        }
        if not self.fourcc_mapping.get(fourcc):
            raise ValueError(
                f"Unsupported {fourcc=}, use one of following {list(self.fourcc_mapping.keys())}"
            )
        self.fourcc = fourcc

    def extension_according_to_fourcc(self, full_path: Path):
        suffix = self.fourcc_mapping.get(self.fourcc.upper())
        return full_path.with_suffix(suffix)

    def render_stabilized(self, output_video_path: Path, tracker: Tracker):
        if not tracker.is_valid:
            print(f"{tracker.is_valid}")
            return

        output_video_path = self.extension_according_to_fourcc(output_video_path)
        video_writer = get_video_writer(
            output_video_path=output_video_path,
            fps=self.video_metadata.fps,
            width=self.video_metadata.width,
            height=self.video_metadata.height,
            fourcc=self.fourcc,
        )

        cap = cv2.VideoCapture(self.video_metadata.video_path)
        for tracker_center in tracker.tracked_coordinates:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.display_manager.warp_affine(frame, tracker_center)
            video_writer.write(frame)

        video_writer.release()
        cap.release()

        print(f"\n-- done rendering to '{output_video_path}' --\n")
