from abc import ABC, abstractmethod
from copy import deepcopy

import cv2
import numpy as np
from misc_functions.utils import convert_image


class Tracker(ABC):
    def __init__(self):
        self._active = True
        self._valid = False
        self.tracked_coordinates = {}
        self.tracked_coordinates_with_frame_index = {}

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, val):
        self._active = val

    @property
    def is_valid(self):
        return len(self.tracked_coordinates) != 0

    @staticmethod
    def factory(method):
        if method == "optical_flow":
            return OpticalFlow
        if method == "tracker":
            raise NotImplementedError
        if method == "empty":
            raise EmptyTracker
        else:
            raise ValueError(f"'{method}' is not an option")

    @abstractmethod
    def initialize_tracker(self, image):
        ...

    @abstractmethod
    def update_tracker(self, image, index):
        ...

    @abstractmethod
    def fix_roi_offset(self, x, y, w, h):
        ...

    def sort_tracker_data(self):
        self.tracked_coordinates_with_frame_index = deepcopy(self.tracked_coordinates)
        self.tracked_coordinates = dict(sorted(self.tracked_coordinates.items()))
        self.tracked_coordinates = np.array(
            list(self.tracked_coordinates.values())
        ).squeeze()

    def get_coordinate_at_index(self, index):
        return self.tracked_coordinates[index]

    def convolve_smooth(self, window_size, mode="valid"):
        kernel = np.ones(window_size) / window_size

        pad_x = np.pad(self.tracked_coordinates[..., 0], window_size // 2, mode="edge")
        pad_y = np.pad(self.tracked_coordinates[..., 1], window_size // 2, mode="edge")

        smooth_x = np.convolve(pad_x, kernel, mode=mode)
        smooth_y = np.convolve(pad_y, kernel, mode=mode)

        smoothed = np.column_stack([smooth_x, smooth_y])
        # self.tracked_coordinates = smoothed
        return smoothed




class EmptyTracker(Tracker):
    def __init__(self):
        super().__init__()

    def initialize_tracker(self, image, **kwargs): ...

    def fix_roi_offset(self, x, y, w, h, **kwargs): ...

    def update_tracker(self, image, index): ...


class ColorMasking(Tracker):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def initialize_tracker(self, image): ...

    @abstractmethod
    def update_tracker(self, image, index): ...

    @abstractmethod
    def fix_roi_offset(self, x, y, w, h): ...


class OpticalFlow(Tracker):
    def __init__(self, feature_params):
        super().__init__()
        if feature_params is None:
            self.feature_params = dict(
                maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
            )
        else:
            self.feature_params = feature_params

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.p0 = None
        self._old_gray = None

    @property
    def old_gray(self):
        return self._old_gray

    @old_gray.setter
    def old_gray(self, val):
        self._old_gray = val

    def initialize_tracker(self, image, **kwargs):
        image = convert_image(image, "gray")
        self.p0 = cv2.goodFeaturesToTrack(image, mask=None, **self.feature_params)
        if self.p0 is None:
            print("cannot find goodFeaturesToTrack in this roi")
            return
        self.old_gray = convert_image(kwargs["old_gray"], "gray")

    def fix_roi_offset(self, x, y, w, h, **kwargs):
        self.p0 = self.p0 + np.array([x, y])

    def update_tracker(self, image, index):
        image = convert_image(image, "gray")
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray,
            image,
            self.p0.astype(np.float32),
            None,
            **self.lk_params,
        )

        if np.all(st == 0):
            print("[-] All trackers are status 0")
            self.active = False
            return

        # good_points = p1[st]
        good_points = p1[st.flatten() == 1]

        self.old_gray = image
        self.p0 = p1

        good_points_mean = np.mean(good_points, axis=0, dtype=np.int32)
        self.tracked_coordinates[index] = good_points_mean
        return good_points_mean  # return for debugging purposes
