"""
TODO
look into sliders disappearing
lost track but still not all frames are passed
color based tracking
smoothing with convolution
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from screeninfo import get_monitors

from video_stabilizer.helpers.helpers import GoodFeaturesViewer
from video_stabilizer.trackers.trackers import Tracker
from misc_functions import rescale, convert_image, get_video_writer


class Stabilizer:
    def __init__(self, video_path):
        self.video_path = video_path

        self._get_monitor_size()

        self.current_frame_index = 0
        self.track_coordinates = None
        self.pin_coord = None
        self.lock_axis_map = {1: None, 0: 0, 2: 1}
        self.lock_axis = False

        self.smooth_coords = False
        self.kernel_win_size = 1
        self.track_coordinates_smoothed = {}

        self.rescale_increment = 0.15
        self.rescale_factor = 1

        self._video_len, self._video_height, self._video_width, self._video_fps = (
            None,
            None,
            None,
            None,
        )
        self._all_frames = self.load_video()

        self._windows_name = "Stabilize"
        self._timeline = "timeline"
        self._display_windows = "display"

        self._show_cross = False
        self._show_good_features_to_track = False
        self._current_frame = None
        self._track_trail_thickness = int(self.video_width * 0.01)
        self._circle_thickness = int(max(self.video_width, self.video_height) * 0.003)

        self._good_features_viewer = GoodFeaturesViewer(int(self._monitor_width * 0.45))
        self._good_features_viewer.create_sliders()
        self._optical_flow_feature_params = None

        self._manual_pin = False
        self._manual_movement = {
            ord("w"): np.array([0, -1]),
            ord("s"): np.array([0, 1]),
            ord("d"): np.array([1, 0]),
            ord("a"): np.array([-1, 0]),
        }

        self.inputs_mapper = {
            27: lambda: "exit",
            ord("q"): lambda: "break",
            ord("Q"): lambda: "break",
            43: self.rescale_larger,  # rescale_larger
            45: self.rescale_smaller,  # rescale_smaller
            ord("c"): self.toggle_cross,
            ord("g"): self.toggle_good_features_to_track,
            ord("o"): self.optical_flow_method,
            # ord("t"): self.tracker_method,
            ord("p"): self.set_pin_coord,
            ord("m"): self.toggle_manual_pin,
            ord("u"): self.unset_pin,
            ord("r"): self.render_op,
            ord("R"): self.render_all_axis,
        }

    def show_commands(self):
        SPECIAL_KEYS = {
            0: "←",
            8: "Backspace",
            # 9: "Tab",
            # 10: "Line Feed",
            13: "Enter",
            27: "Escape",
            32: "Space",
        }
        key_code_max_len = max([len(str(key)) for key in self.inputs_mapper]) + 2
        key_repr_max_len = max([len(str(key)) for key in SPECIAL_KEYS.values()]) + 4
        funct_max_len = max(
            [len(str(function.__name__)) for function in self.inputs_mapper.values()]
        )

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

    def __del__(self):
        cv2.destroyAllWindows()

    @property
    def video_len(self):
        return self._video_len

    @video_len.setter
    def video_len(self, value):
        self._video_len = value

    @property
    def video_height(self):
        return self._video_height

    @video_height.setter
    def video_height(self, value):
        self._video_height = value

    @property
    def video_width(self):
        return self._video_width

    @video_width.setter
    def video_width(self, value):
        self._video_width = value

    @property
    def video_fps(self):
        return self._video_fps

    @video_fps.setter
    def video_fps(self, value):
        self._video_fps = value

    def set_rescale_factor(self):
        target_size_height = self._monitor_height * 0.7
        if self.video_height > target_size_height:
            self.rescale_factor = target_size_height / self.video_height

    def _get_monitor_size(self):
        self._monitor_width = get_monitors()[0].width
        self._monitor_height = get_monitors()[0].height

    def rescale_larger(self):
        self.rescale_factor += self.rescale_increment

    def rescale_smaller(self):
        self.rescale_factor -= self.rescale_increment
        self.rescale_factor = min(0.1, self.rescale_increment)

    def unset_pin(self):
        self.pin_coord = None
        print("pin unset")

    def select_roi(self):
        rescaled_frame = rescale(self._current_frame, self.rescale_factor)

        roi = cv2.selectROI(self._display_windows, rescaled_frame)
        crop_rescale_factor = self.video_height / rescaled_frame.shape[0]
        return map(int, np.array(roi) * crop_rescale_factor)

    def on_change(self, value):
        self.current_frame_index = value

    def on_lock_axis(self, value):
        self.lock_axis = self.lock_axis_map.get(value)

    def on_smooth_coords(self, value):
        self.smooth_coords = value

    def on_conv_smooth_coords(self, value):
        self.kernel_win_size = value

    def create_sliders(self):
        cv2.namedWindow(self._timeline, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("", self._timeline, 0, 1, self.on_change)
        cv2.setTrackbarMax("", self._timeline, self.video_len - 1)
        cv2.resizeWindow(self._timeline, int(self._monitor_width * 0.8), 100)

        cv2.createTrackbar("lock_axis", self._timeline, 1, 1, self.on_lock_axis)
        cv2.setTrackbarMax("lock_axis", self._timeline, 2)

        cv2.createTrackbar("smooth", self._timeline, 0, 1, self.on_smooth_coords)
        cv2.setTrackbarMax("smooth", self._timeline, 1)

        cv2.createTrackbar(
            "conv_smooth", self._timeline, 1, 1, self.on_conv_smooth_coords
        )
        cv2.setTrackbarMax("conv_smooth", self._timeline, 20)
        cv2.setTrackbarMin("conv_smooth", self._timeline, 1)

    ############################### Tracking Operations  ################################

    def optical_flow_method(self, *args, **kwargs):
        print("\n-- optical flow --")
        tracker_class = Tracker.factory("optical_flow")
        tracker = tracker_class(self._optical_flow_feature_params)
        self.run_tracking(tracker)

    def run_tracking(self, tracker):
        self._show_good_features_to_track = False

        x, y, w, h = self.select_roi()

        if not all([x, y, w, h]):
            print("[-] roi not set")
            return

        self.dim_display_window()
        self.track_coordinates = None
        self.raw_track_coordinates = None

        roi_image = self._current_frame[y: y + h, x: x + w]
        roi_image = convert_image(roi_image, "gray")

        forward_range = range(self.current_frame_index, self.video_len)
        backward_range = range(self.current_frame_index - 1, -1, -1)

        print("\n-- tracking --")
        print(f"{forward_range= }")
        print(f"{backward_range=}")

        old_frame = np.copy(self._all_frames[self.current_frame_index])

        tracker.initialize_tracker(roi_image, old_gray=old_frame)
        tracker.fix_roi_offset(x, y, w, h)

        self.track_coordinates = {}
        for range_set in [forward_range, backward_range]:
            for index in range_set:
                debug_display_frame = np.copy(self._all_frames[index])
                current_gray = convert_image(debug_display_frame, "gray")

                coordinates = tracker.update_tracker(current_gray)

                # TODO lost track but still not all frames are passed
                if not tracker.active:
                    break

                self.track_coordinates[index] = np.mean(
                    coordinates, axis=0, dtype=np.int32
                )

                # for i in coordinates:
                #     x, y = map(int, i.ravel())
                #     cv2.circle(debug_display_frame, (x, y), 5, (0, 255, 0), 5)
                #
                # debug_image = rescale(debug_display_frame, self.rescale_factor)
                # cv2.imshow("Tracking", debug_image)
                # cv2.waitKey(1)

        # cv2.destroyWindow("Tracking")
        missing_frames = np.setdiff1d(
            np.arange(self._video_len), np.array(list(self.track_coordinates.keys()))
        )
        if missing_frames.size != 0:
            print(f"[!] missing frames\n {missing_frames}")
            return

        self.track_coordinates = np.array(
            list(self.track_coordinates.values())
        ).squeeze()
        self.raw_track_coordinates = np.copy(self.track_coordinates)
        self.set_pin_coord(index=self.current_frame_index)
        # self.smooth_coordinates()
        print("done tracking\n")

    def smooth_coordinates(self, sigma=3):
        coords = np.array(list(self.track_coordinates.values()))
        coords = coords.reshape(-1, 2)
        smoothed_x = gaussian_filter1d(coords[:, 0], sigma=sigma)
        smoothed_y = gaussian_filter1d(coords[:, 1], sigma=sigma)
        smoothed = np.vstack((smoothed_x, smoothed_y)).T
        for index, coord in enumerate(smoothed):
            self.track_coordinates_smoothed[index] = coord
        return np.vstack((smoothed_x, smoothed_y)).T

    def set_pin_coord(self, index=None, *args, **kwargs):
        self.pin_coord = self.track_coordinates[index]
        # if self.track_coordinates:
        #     set_index = self.current_frame_index
        #     # if index:
        #     #     set_index = index
        #     self.pin_coord = self.track_coordinates[set_index]
        print(f"pin set to {self.pin_coord}")

    ################################## DISPLAY RELATED ##################################

    def toggle_cross(self, *args):
        self._show_cross = not self._show_cross

    def toggle_good_features_to_track(self, *args):
        self._show_good_features_to_track = not self._show_good_features_to_track
        self._good_features_viewer.active = self._show_good_features_to_track

    @staticmethod
    def _draw_cross(frame):
        """x:vertical; y:horizontal"""
        mid_y, mid_x = np.array(np.array(frame.shape[:2]) / 2).astype(np.uint32)
        cross_width = (mid_y * 0.005).astype(np.uint32)

        frame[mid_y - cross_width: mid_y + cross_width, :, :] = [255, 0, 0]
        frame[:, mid_x - cross_width: mid_x + cross_width, :] = [255, 0, 0]

    def _draw_good_features_to_track(self):
        corners = self._good_features_viewer.get_good_features(
            convert_image(self._current_frame, "gray")
        )
        if corners is None:
            return
        self._optical_flow_feature_params = (
            self._good_features_viewer.parses_parameters()
        )

        for feature in corners:
            x, y = map(int, feature.ravel())
            cv2.circle(
                self._current_frame, (x, y), 5, (255, 255, 0), self._circle_thickness
            )

    def draw_tracker_tail(self, length=100):
        # TODO think of what best len is
        length = self.video_len

        _track_coordinates = self.track_coordinates

        coord_indices = np.arange(
            self.current_frame_index - length, self.current_frame_index
        )
        valid_indices = coord_indices[coord_indices >= 0]

        line_thickness = np.linspace(1, self._track_trail_thickness, length)
        color_gradients = np.linspace(100, 255, length).astype(np.uint8)
        _zip = zip(
            valid_indices[1:], valid_indices[:-1], line_thickness, color_gradients
        )
        for before, after, _line_thickness, color_gradient in _zip:
            cv2.line(
                img=self._current_frame,
                pt1=_track_coordinates[after].ravel().astype(np.intp),
                pt2=_track_coordinates[before].ravel().astype(np.intp),
                color=(int(color_gradient), 255, int(color_gradient)),
                thickness=int(_line_thickness),
            )

        cv2.circle(
            img=self._current_frame,
            center=_track_coordinates[self.current_frame_index].ravel().astype(np.intp),
            radius=5,
            color=(0, 255, 0),
            thickness=self._circle_thickness,
        )

    def toggle_manual_pin(self, *args, **kwargs):
        self._manual_pin = not self._manual_pin
        print(f"{self._manual_pin=}")

    def manual_pin(self, key):
        if self.pin_coord is None:
            return
        _stride = 2
        if self._manual_movement.get(key) is None:
            return
        self.pin_coord = self.pin_coord + self._manual_movement[key] * _stride

    def dim_display_window(self):
        """dim display image while doing ops"""
        self._current_frame = (self._current_frame * 0.5).astype(np.uint8)
        cv2.imshow(
            self._display_windows, rescale(self._current_frame, self.rescale_factor)
        )
        cv2.waitKey(1)

    # @cache
    def convolve_smooth(self, window_size, mode="valid"):
        kernel = np.ones(self.kernel_win_size) / self.kernel_win_size

        pad_x = np.pad(
            self.raw_track_coordinates[..., 0], self.kernel_win_size // 2, mode="edge"
        )
        pad_y = np.pad(
            self.raw_track_coordinates[..., 1], self.kernel_win_size // 2, mode="edge"
        )

        smooth_x = np.convolve(pad_x, kernel, mode=mode)
        smooth_y = np.convolve(pad_y, kernel, mode=mode)

        smoothed = np.column_stack([smooth_x, smooth_y])
        print(smoothed.shape)
        self.track_coordinates = smoothed
        return smoothed

    def stabilize(self):
        """main loop"""
        if not self._all_frames:
            return

        self.create_sliders()
        shape = self.video_width, self.video_height

        while True:
            raw_frame = np.copy(self._all_frames[self.current_frame_index])
            self._current_frame = np.copy(raw_frame)

            # display tracker data
            if self.track_coordinates is not None:
                # _track_coordinates = (
                #     self.convolve_smooth()
                #     if self.smooth_coords
                #     else self.track_coordinates
                # )

                _track_coordinates = self.convolve_smooth(
                    window_size=self.kernel_win_size
                )

                current_coord_center = _track_coordinates[
                    self.current_frame_index
                ].flatten()
                self.draw_tracker_tail()

                if self.pin_coord is not None:
                    delta = (self.pin_coord - current_coord_center).ravel()
                    if self.lock_axis is not None:
                        delta[self.lock_axis] = 0
                    delta_x, delta_y = delta

                    translation_matrix = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
                    self._current_frame = cv2.warpAffine(
                        self._current_frame, translation_matrix, shape
                    )
            if self._show_good_features_to_track:
                self._draw_good_features_to_track()
            if self._show_cross:
                Stabilizer._draw_cross(self._current_frame)

            cv2.imshow(
                self._display_windows,
                rescale(self._current_frame, self.rescale_factor),
            )
            key = cv2.waitKey(1)

            # TODO maybe rework this
            if self._manual_pin:
                self.manual_pin(key)

            ret = self.handle_inputs(key)
            if ret == "break":
                break
            if ret == "exit":
                return

    def handle_inputs(self, key):
        input_instruction = self.inputs_mapper.get(key)
        if input_instruction:
            return input_instruction()

    ################################### RENDER RELATED ##################################

    def make_video_writer(self, output_video_path):
        return get_video_writer(
            output_video_path,
            self.video_fps,
            self.video_width,
            self.video_height,
            fourcc="XVID",
        )

    def render_stabilized(self, cap, video_writer, _track_coordinates):
        self.dim_display_window()

        shape = self.video_width, self.video_height
        for tracker_center in _track_coordinates:
            ret, frame = cap.read()
            if not ret:
                break

            current_coord_center = _track_coordinates[tracker_center]
            delta = (self.pin_coord - current_coord_center).ravel()
            if self.lock_axis:
                delta[self.lock_axis] = 0
            delta_x, delta_y = delta

            translation_matrix = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
            frame = cv2.warpAffine(frame, translation_matrix, shape)
            video_writer.write(frame)
        video_writer.release()
        cap.release()

    # FIXME could use rework, copying render_op pretty much
    def render_all_axis(self, *args):
        self.dim_display_window()

        if not self.track_coordinates:
            print("[-] no tracked info")
            return

        _track_coordinates = (
            self.track_coordinates_smoothed
            if self.smooth_coords
            else self.track_coordinates
        )

        if self.pin_coord is None:
            print("[-] Pin not set")
            self.pin_coord = _track_coordinates[0]

        print("\n-- started render_all_axis --")

        prev_lock = cv2.getTrackbarPos("lock_axis", self._timeline)
        for i in [0, 1, 2]:
            cap = cv2.VideoCapture(self.video_path)
            assert self.video_len == int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cv2.setTrackbarPos("lock_axis", self._timeline, i)
            lock = ""
            if self.lock_axis == 1:
                lock = "_horiz"
            if self.lock_axis == 0:
                lock = "_vert"
            smoothed = "_smooth" if self.smooth_coords else ""

            output_video_path = self.video_path.with_stem(
                self.video_path.stem + "_stb" + lock + smoothed
            )
            video_writer = self.make_video_writer(output_video_path)
            self.render_stabilized(cap, video_writer, _track_coordinates)
        cv2.setTrackbarPos("lock_axis", self._timeline, prev_lock)
        print("Done rendering\n")

    def render_op(self, *args):
        if not self.track_coordinates:
            print("[-] no tracked info")
            return

        _track_coordinates = (
            self.track_coordinates_smoothed
            if self.smooth_coords
            else self.track_coordinates
        )

        if self.pin_coord is None:
            print("[-] Pin not set")
            self.pin_coord = _track_coordinates[0]
            # return

        print("\n-- started rendering --")
        cap = cv2.VideoCapture(self.video_path)
        assert self.video_len == int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        lock = ""
        if self.lock_axis == 1:
            lock = "_horiz"
        if self.lock_axis == 0:
            lock = "_vert"
        smoothed = "_smooth" if self.smooth_coords else ""
        output_video_path = self.video_path.with_stem(
            self.video_path.stem + "_stb" + lock + smoothed
        )
        video_writer = self.make_video_writer(output_video_path)
        self.render_stabilized(cap, video_writer, _track_coordinates)
        print("Done rendering\n")

    def load_video(self):
        """return list of all frames and sets scale_factor to fit screen"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[!] {cap.isOpened()=}")

        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f"shape:  {self.video_width} {self.video_height}")
        print(f"fps:    {self.video_fps}")

        self.set_rescale_factor()

        print("\n-- loading video into memory --")
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()

        self.video_len = len(all_frames)
        return all_frames
