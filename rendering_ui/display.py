from dataclasses import dataclass

import cv2
import numpy as np
from screeninfo import get_monitors

from utils import rescale
from video_stabilizer.trackers.trackers import Tracker
from video_stabilizer.video.video_metadata import VideoMetadata


@dataclass
class DrawingRelatedProperties:
    track_trail_thickness: int
    circle_thickness: int
    draw_cross: int
    lock_axis: int
    pin_coord: np.ndarray | None
    manual_pin: bool
    kernel_win_size: int

    @classmethod
    def from_video_metadate(cls, video_metadata: VideoMetadata):
        return cls(
            track_trail_thickness=int(video_metadata.width * 0.01),
            circle_thickness=int(
                max(video_metadata.width, video_metadata.height) * 0.003
            ),
            draw_cross=False,
            lock_axis=1,
            pin_coord=None,
            manual_pin=False,
            kernel_win_size=1,
        )

    def get_locked_axis(self):
        """
        as slider has 2 position its is mapped like this
        [locker_x, no_lock, locker_y]
        """
        lock_axis_mapping = {0: 0, 1: None, 2: 1}
        return lock_axis_mapping.get(self.lock_axis)

    def get_props_for_output_path(self):
        """creates string to append to rendered video path"""
        lock_axis = self.get_locked_axis()
        lock_axis = f"_locked_axis_{lock_axis}" if lock_axis is not None else ""
        kernel_win_size = self.kernel_win_size
        kernel_win_size = f"_conv_{kernel_win_size}" if kernel_win_size != 1 else ""
        return lock_axis + kernel_win_size


class DisplayManager:
    def __init__(
        self,
        drawing_properties: DrawingRelatedProperties,
        frames: list[np.ndarray],
        video_metadata: VideoMetadata,
        tracker_component: Tracker,
    ):
        self.video_metadata = video_metadata
        self.drawing_properties = drawing_properties
        self.tracker_component = tracker_component

        self.frames = frames
        self.current_frame_index = 0

        self.rescale_increment = 0.5
        self.rescale_factor = 1

        self.windows_name = "Stabilize"
        self.timeline = "timeline"
        self.display_windows = "display"

        self._monitor_width = get_monitors()[0].width
        self._monitor_height = get_monitors()[0].height

        self.set_rescale_factor()

        self._manual_movement = {
            ord("w"): np.array([0, -1]),
            ord("s"): np.array([0, 1]),
            ord("d"): np.array([1, 0]),
            ord("a"): np.array([-1, 0]),
        }

    def on_change(self, value):
        self.current_frame_index = value

    def on_lock_axis(self, value):
        self.drawing_properties.lock_axis = value

    def on_convolve_coordinates(self, value):
        self.drawing_properties.kernel_win_size = value

    def create_sliders(self):
        cv2.namedWindow(self.timeline, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("", self.timeline, 0, 1, self.on_change)
        cv2.setTrackbarMax("", self.timeline, self.video_metadata.frames_count - 1)
        # cv2.resizeWindow(self.timeline, int(self._monitor_width * 0.8), 100)

        cv2.createTrackbar("lock_axis", self.timeline, 1, 1, self.on_lock_axis)
        cv2.setTrackbarMax("lock_axis", self.timeline, 2)

        cv2.createTrackbar(
            "conv_smooth", self.timeline, 1, 1, self.on_convolve_coordinates
        )
        cv2.setTrackbarMax("conv_smooth", self.timeline, 20)
        cv2.setTrackbarMin("conv_smooth", self.timeline, 1)

    def set_rescale_factor(self):
        target_size_height = self._monitor_height * 0.7
        if self.video_metadata.height > target_size_height:
            self.rescale_factor = target_size_height / self.video_metadata.height

    def rescale_larger(self, *args, **kwargs):
        self.rescale_factor += self.rescale_increment

    def rescale_smaller(self, *args, **kwargs):
        self.rescale_factor -= self.rescale_increment

    def select_roi(self):
        rescaled_frame = self.show_frame()

        roi = cv2.selectROI(self.display_windows, rescaled_frame)
        crop_rescale_factor = self.video_metadata.height / rescaled_frame.shape[0]
        return map(int, np.array(roi) * crop_rescale_factor)

    def toggle_manual_pin(self, *args, **kwargs):
        self.drawing_properties.manual_pin = not self.drawing_properties.manual_pin

    def move_pin_manually(self, *args, **kwargs):
        if self.drawing_properties.pin_coord is None:
            return
        _stride = 2
        direction = self._manual_movement.get(kwargs["key"])
        if direction is None:
            return
        self.drawing_properties.pin_coord = (
            self.drawing_properties.pin_coord + direction * _stride
        )

    def set_pin(self, *args, **kwargs):
        coord = kwargs.get("coord")  # on end of track
        if coord is not None:  # on end of track
            self.drawing_properties.pin_coord = kwargs.get("coord")
            return

        key = kwargs.get("key")

        tracker = kwargs["tracker"]
        if not tracker.is_valid:
            return

        coordinate = tracker.tracked_coordinates[self.current_frame_index]

        if key == ord("p"):
            self.drawing_properties.pin_coord = coordinate
        elif key == ord("u"):
            self.drawing_properties.pin_coord = None
        else:
            print(f"{key=} not supported for 'set_pin()'\n{kwargs=}")
        print(f"pin_coord:{self.drawing_properties.pin_coord}")

    def toggle_cross(self, *args, **kwargs):
        self.drawing_properties.draw_cross = not self.drawing_properties.draw_cross

    @staticmethod
    def _draw_cross(frame):
        """x:vertical; y:horizontal"""
        mid_y, mid_x = np.array(np.array(frame.shape[:2]) / 2).astype(np.uint32)
        cross_width = (mid_y * 0.005).astype(np.uint32)

        frame[mid_y - cross_width : mid_y + cross_width, :, :] = [255, 0, 0]
        frame[:, mid_x - cross_width : mid_x + cross_width, :] = [255, 0, 0]

    def draw_tracker_tail(self, image, track_coordinates, length=100):
        # TODO think of what best len is
        length = self.video_metadata.frames_count

        coord_indices = np.arange(
            self.current_frame_index - length, self.current_frame_index
        )
        valid_indices = coord_indices[coord_indices >= 0]

        line_thickness = np.linspace(
            1, self.drawing_properties.track_trail_thickness, length
        )
        color_gradients = np.linspace(100, 255, length).astype(np.uint8)
        _zip = zip(
            valid_indices[1:], valid_indices[:-1], line_thickness, color_gradients
        )
        for before, after, _line_thickness, color_gradient in _zip:
            cv2.line(
                img=image,
                pt1=track_coordinates[after].ravel().astype(np.intp),
                pt2=track_coordinates[before].ravel().astype(np.intp),
                color=(int(color_gradient), 255, int(color_gradient)),
                thickness=int(_line_thickness),
            )

        cv2.circle(
            img=image,
            center=track_coordinates[self.current_frame_index].ravel().astype(np.intp),
            radius=5,
            color=(0, 255, 0),
            thickness=self.drawing_properties.circle_thickness,
        )

    def warp_affine(self, image, current_coord_center):
        if self.drawing_properties.pin_coord is None:
            return image

        delta = (self.drawing_properties.pin_coord - current_coord_center).ravel()

        axis_to_lock = self.drawing_properties.get_locked_axis()
        if axis_to_lock is not None:
            delta[axis_to_lock] = 0
        delta_x, delta_y = delta

        translation_matrix = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
        return cv2.warpAffine(
            image,
            translation_matrix,
            [self.video_metadata.width, self.video_metadata.height],
        )

    def dim_display_window(self):
        """dim display image while doing ops"""
        self.show_frame((self.get_current_frame() * 0.5).astype(np.uint8))
        cv2.waitKey(1)

    def get_current_frame(self):
        return np.copy(self.frames[self.current_frame_index])

    def show_frame(self, frame=None, warp=None) -> np.ndarray:
        """displays rescaled frame"""
        frame = frame if frame is not None else self.get_current_frame()

        if warp:
            self.draw_tracker_tail(frame, warp["tracked_coordinates"])
            frame = self.warp_affine(frame, warp["current_coord_center"])

        scaled_image = rescale(frame, self.rescale_factor)

        if self.drawing_properties.draw_cross:
            self._draw_cross(scaled_image)

        cv2.imshow(self.display_windows, scaled_image)
        return scaled_image
