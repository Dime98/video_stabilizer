import cv2
import numpy as np
from utils import rescale

from video_stabilizer.video.video_metadata import VideoMetadata


class ViewColorMasking:
    def __init__(
        self,
        video_metadata: VideoMetadata,
        frames: list[np.ndarray] | None = None,
        rescale_factor: float | None = None,
    ):
        self.index = 0
        self.frames = frames
        self.video_metadata = video_metadata

        self.h_min, self.h_max = 0, 179
        self.s_min, self.s_max = 0, 255
        self.v_min, self.v_max = 0, 255

        self.window_name = "ViewColorMasking"
        self.rescale_factor = rescale_factor

    def reset_sliders(self):
        cv2.setTrackbarPos("h_min", self.window_name, 0)
        cv2.setTrackbarPos("h_max", self.window_name, 170)
        cv2.setTrackbarPos("s_min", self.window_name, 0)
        cv2.setTrackbarPos("s_max", self.window_name, 255)
        cv2.setTrackbarPos("v_min", self.window_name, 0)
        cv2.setTrackbarPos("v_max", self.window_name, 255)

    def on_change(self, value):
        self.index = value

    def create_sliders(self):
        time_line_max = (
            0 if self.frames is None else self.video_metadata.frames_count - 1
        )
        cv2.createTrackbar("", self.window_name, 0, 1, self.on_change)
        cv2.setTrackbarMax("", self.window_name, time_line_max)

        cv2.createTrackbar(
            "h_min", self.window_name, self.h_min, 179, lambda *args: None
        )
        cv2.createTrackbar(
            "h_max", self.window_name, self.h_max, 179, lambda *args: None
        )

        cv2.createTrackbar(
            "s_min", self.window_name, self.s_min, 255, lambda *args: None
        )
        cv2.createTrackbar(
            "s_max", self.window_name, self.s_max, 255, lambda *args: None
        )

        cv2.createTrackbar(
            "v_min", self.window_name, self.v_min, 255, lambda *args: None
        )
        cv2.createTrackbar(
            "v_max", self.window_name, self.v_max, 255, lambda *args: None
        )

    def mask(self, image, min_values, max_values):
        """
        min_values  [h_min, s_min, v_min]
        min_values  [h_max, s_max, v_max]
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array(min_values)
        upper = np.array(max_values)

        return cv2.inRange(hsv, lower, upper)

    def view(self, image=None):
        cv2.namedWindow(self.window_name)
        # cv2.resizeWindow(self._timeline, int(self._monitor_width * 0.8), 100)
        self.create_sliders()

        if self.frames is None:
            self.frames = [image]

        return_value = None
        while True:
            image = np.copy(self.frames[self.index])

            self.h_min = cv2.getTrackbarPos("h_min", self.window_name)
            self.s_min = cv2.getTrackbarPos("s_min", self.window_name)
            self.v_min = cv2.getTrackbarPos("v_min", self.window_name)
            self.h_max = cv2.getTrackbarPos("h_max", self.window_name)
            self.s_max = cv2.getTrackbarPos("s_max", self.window_name)
            self.v_max = cv2.getTrackbarPos("v_max", self.window_name)

            lower = np.array([self.h_min, self.s_min, self.v_min])
            upper = np.array([self.h_max, self.s_max, self.v_max])

            mask = self.mask(image, lower, upper)
            image[mask == 0] = (0, 0, 0)

            concatenated_images = cv2.hconcat(
                [image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]
            )
            if self.rescale_factor:
                concatenated_images = rescale(concatenated_images, self.rescale_factor)
            cv2.imshow("color mask viewer", concatenated_images)

            key = cv2.waitKey(10)
            if key in [27, ord("q")]:
                return_value = None
                break
            if key in [13, 32]:  # 13: enter, 32: space
                return_value = lower, upper
                break
            if key in [ord("r"), ord("R")]:
                self.reset_sliders()

        cv2.destroyWindow(self.window_name)
        cv2.destroyWindow("color mask viewer")
        return return_value


class GoodFeaturesViewer:
    def __init__(self, width):
        self.active = False
        self._window_name = "Good Features"
        self._width = width
        self._parameters = {
            "maxCorners": {
                "value": 100,
                "min": 3,
                "max": 1000,
                "onChange": self._on_maxCorners_change,
            },
            "qualityLevel": {
                "value": 1,
                "min": 2,
                "max": 100,
                "onChange": self._on_qualityLevel_change,
            },
            "blockSize": {
                "value": 3,
                "min": 1,
                "max": 30,
                "onChange": self._on_block_change,
            },
            "minDistance": {
                "value": 1,
                "min": 1,
                "max": 100,
                "onChange": self._on_minDistance_change,
            },
            "corners": {"value": 100, "max": 500, "onChange": self._on_corners_change},
            "useHarrisDetector": {
                "value": 0,
                "max": 1,
                "onChange": self._on_useHarrisDetector_change,
            },
            "k": {"min": 0, "max": 500, "value": 0, "onChange": self._on_k_change},
        }
        for param, param_data in self._parameters.items():
            print(param, param_data["value"])

    def set_parameter(self, parameter_name, val):
        if self.active:
            self._parameters[parameter_name]["value"] = val
        else:
            cv2.setTrackbarPos(
                parameter_name,
                self._window_name,
                self._parameters[parameter_name]["value"],
            )

    def _on_maxCorners_change(self, val):
        self.set_parameter("maxCorners", val)

    def _on_qualityLevel_change(self, val):
        self.set_parameter("qualityLevel", val)

    def _on_block_change(self, val):
        self.set_parameter("blockSize", val)

    def _on_minDistance_change(self, val):
        self.set_parameter("minDistance", val)

    def _on_corners_change(self, val):
        self.set_parameter("corners", val)

    def _on_useHarrisDetector_change(self, val):
        self.set_parameter("useHarrisDetector", val)

    def _on_k_change(self, val):
        self.set_parameter("k", val)

    def create_sliders(self):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, self._width, 100)

        for parameter_name, parameter_data in self._parameters.items():
            cv2.createTrackbar(
                parameter_name,
                self._window_name,
                parameter_data["value"],
                parameter_data["max"],
                parameter_data["onChange"],
            )
            if parameter_data.get("min"):
                cv2.setTrackbarMin(
                    parameter_name, self._window_name, parameter_data["min"]
                )

    def parses_parameters(self):
        if not self.active:
            return
        parsed_parameters = {
            param: param_data["value"] for param, param_data in self._parameters.items()
        }
        parsed_parameters["qualityLevel"] /= 100
        return parsed_parameters

    def get_good_features(self, image):
        corners = cv2.goodFeaturesToTrack(image, **self.parses_parameters())
        return corners
