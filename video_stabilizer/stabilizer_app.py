import cv2
import numpy as np

from helpers.helpers import ViewColorMasking
from rendering_ui.display import (
    DisplayManager,
    DrawingRelatedProperties,
)
from trackers.trackers import Tracker, EmptyTracker
from cv2_utils.utils import convert_image
from video_stabilizer.video.render_video import VideoRenderer
from video_stabilizer.video.video_metadata import VideoMetadata


class InputHandler:
    def __init__(
        self,
        app: "StabilizerApplication",
        display_component: DisplayManager,
    ):
        self.app = app
        self.display_component = display_component

        self.inputs_mapper = {
            27: lambda **kwargs: "exit",
            ord("q"): lambda **kwargs: "break",
            ord("Q"): lambda **kwargs: "break",
            43: display_component.rescale_larger,  # rescale_larger
            45: display_component.rescale_smaller,  # rescale_smaller
            ord("p"): self.display_component.set_pin,
            ord("u"): self.display_component.set_pin,
            ord("m"): self.display_component.toggle_manual_pin,
            ord("C"): display_component.toggle_cross,
            ord("o"): self.app.start_optical_flow_tracking,
            ord("c"): self.app.start_color_masking_tracking,
            ord("t"): self.app.start_tracker_method,
            ord("r"): self.app.render_op,
            ord("R"): self.app.render_all_axis,
        }

    def handle_inputs(self, key):
        input_instruction = self.inputs_mapper.get(key)
        if input_instruction:
            return input_instruction(key=key, tracker=self.app.active_tracker)

    def show_commands(self):
        SPECIAL_KEYS = {
            0: "←",
            8: "Backspace",
            9: "Tab",
            10: "Line Feed",
            13: "Enter",
            27: "Escape",
            32: "Space",
        }
        key_code_max_len = max([len(str(key)) for key in self.inputs_mapper]) + 2
        key_repr_max_len = max([len(str(key)) for key in SPECIAL_KEYS.values()]) + 4
        funct_max_len = max([len(str(function.__name__)) for function in self.inputs_mapper.values()])

        print("-" * 50)
        for key_code, function in self.inputs_mapper.items():
            key_repr = repr(chr(key_code))
            if key_code in SPECIAL_KEYS:
                key_repr = SPECIAL_KEYS[key_code]
            print(
                f"{key_code:<{key_code_max_len}} | \
                {key_repr:<{key_repr_max_len}} | \
                {function.__name__:>{funct_max_len}}|"
            )
        print("-" * 50)


class StabilizerApplication:
    def __init__(self, video_path):
        self.video_metadata = VideoMetadata.from_path(video_path)
        self.video_path = self.video_metadata.video_path

        self.frames = load_video(video_path)

        self.active_tracker: Tracker = EmptyTracker()
        self.debug_display = False

        drawing_properties = DrawingRelatedProperties.from_video_metadate(self.video_metadata)
        self.display = DisplayManager(
            drawing_properties=drawing_properties,
            frames=self.frames,
            video_metadata=self.video_metadata,
            tracker_component=self.active_tracker,
        )

        self.video_renderer = VideoRenderer(
            video_metadata=self.video_metadata,
            display_manager=self.display,
            fourcc="MJPG",
        )

        self.keybindings = InputHandler(
            app=self,
            display_component=self.display,
        )
        self.keybindings.show_commands()

    def __del__(self):
        cv2.destroyAllWindows()

    def render_op(self, **kwargs):
        self.display.dim_display_window()

        stem = self.video_metadata.video_path.stem
        props_as_str = self.display.drawing_properties.get_props_for_output_path()
        output_path = self.video_metadata.video_path.with_stem(stem + "_stb" + props_as_str)
        self.video_renderer.render_stabilized(output_video_path=output_path, tracker=self.active_tracker)

    def render_all_axis(self, **kwargs):
        self.display.dim_display_window()

        current_lock_axis = cv2.getTrackbarPos("lock_axis", self.display.timeline)
        for i in [0, 1, 2]:
            cv2.setTrackbarPos("lock_axis", self.display.timeline, i)
            stem = self.video_metadata.video_path.stem
            props_as_str = self.display.drawing_properties.get_props_for_output_path()
            output_path = self.video_metadata.video_path.with_stem(stem + "_stb" + props_as_str)
            self.video_renderer.render_stabilized(output_video_path=output_path, tracker=self.active_tracker)
        cv2.setTrackbarPos("lock_axis", self.display.timeline, current_lock_axis)

    def start_tracker_method(self, *args, **kwargs):
        print("\n-- cv2 tracker --")
        tracker_class = Tracker.factory("tracker")
        tracker = tracker_class()
        self.run_tracking(tracker)

    def start_optical_flow_tracking(self, *args, **kwargs):
        print("\n-- optical flow --")
        tracker_class = Tracker.factory("optical_flow")
        tracker = tracker_class(feature_params=None)
        self.run_tracking(tracker)

    def start_color_masking_tracking(self, *args, **kwargs):
        print("\n-- color masking --")
        tracker_class = Tracker.factory("color_masking")
        color_mask_tracker = tracker_class()

        color_mask_viewer = ViewColorMasking(
            frames=self.frames,
            video_metadata=self.video_metadata,
            rescale_factor=self.display.rescale_factor,
        )
        returned_value = color_mask_viewer.view(self.display.show_frame())
        if returned_value is None:
            return
        lower, upper = returned_value
        color_mask_tracker.set_bounds(lower=lower, upper=upper)

        self.run_tracking(color_mask_tracker)

    def run_tracking(self, tracker):
        roi = 0, 0, 0, 0  # placeholder for trackers that don't use roi

        if tracker.requires_roi:
            x, y, w, h = self.display.select_roi()
            roi = x, y, w, h
            if not all([x, y, w, h]):
                print("[-] roi not set")
                return

        self.display.dim_display_window()

        forward_range = range(self.display.current_frame_index, self.video_metadata.frames_count)
        backward_range = range(self.display.current_frame_index - 1, -1, -1)

        print("\n-- tracking --")
        print(f"{forward_range =}")
        print(f"{backward_range=}")

        frame = np.copy(self.display.get_current_frame())

        tracker.initialize_tracker(frame=frame, roi=roi)

        debug_window_name = "tracking solution"
        for range_set in [forward_range, backward_range]:
            for index in range_set:
                image_for_tracking = np.copy(self.frames[index])
                debug_display_frame = np.copy(image_for_tracking)
                current_gray = convert_image(debug_display_frame, "gray")

                coordinates = tracker.update_tracker(image=image_for_tracking, gray_image=current_gray, index=index)

                # TODO lost track but still not all frames are passed
                if not tracker.active:
                    break

                if self.debug_display:
                    tracker.display_tracking_solution(debug_window_name, debug_display_frame, coordinates)

        if self.debug_display:
            cv2.destroyWindow(debug_window_name)

        missing_frames = np.setdiff1d(
            np.arange(self.video_metadata.frames_count),
            np.array(list(tracker.tracked_coordinates.keys())),
        )
        if missing_frames.size != 0:
            print(f"[!] missing frames\n {missing_frames}")
            return

        self.active_tracker = tracker
        tracker.sort_tracker_data()

        self.display.set_pin(coord=tracker.tracked_coordinates[self.display.current_frame_index])
        print("done tracking\n")

    def stabilize(self):
        """Main loop"""
        if not self.frames:
            return

        self.display.create_sliders()

        while True:
            warp_affine = False
            if self.active_tracker.is_valid:
                _tracked_coordinates = self.active_tracker.convolve_smooth(
                    self.display.drawing_properties.kernel_win_size
                )
                current_coord_center = _tracked_coordinates[self.display.current_frame_index].flatten()
                warp_affine = {
                    "current_coord_center": current_coord_center,
                    "tracked_coordinates": _tracked_coordinates,
                }

            self.display.show_frame(warp=warp_affine)
            key = cv2.waitKey(10)

            if self.display.drawing_properties.manual_pin:
                self.display.move_pin_manually(key=key)

            ret = self.keybindings.handle_inputs(key)

            if ret == "break":
                break
            if ret == "exit":
                return


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[!] {cap.isOpened()=}")

    print("\n-- loading video into memory --")
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    return all_frames
