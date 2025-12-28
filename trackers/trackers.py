from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np
from misc_functions.utils import convert_image


class Tracker(ABC):
    def __init__(self):
        self._active = True
        self._valid = False
        self.tracked_coordinates = {}
        self.requires_roi = True

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
        if method == "color_masking":
            return ColorMasking
        if method == "empty":
            raise EmptyTracker
        else:
            raise ValueError(f"'{method}' is not an option")

    @abstractmethod
    def initialize_tracker(self, **kwargs):
        ...

    @abstractmethod
    def update_tracker(self, image, index):
        ...

    @abstractmethod
    def display_tracking_solution(
            self, window_name, image: np.ndarray, solution: Any, **kwargs
    ):
        ...

    def sort_tracker_data(self):
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
        return smoothed


class EmptyTracker(Tracker):
    def __init__(self):
        super().__init__()

    def initialize_tracker(self, **kwargs): ...

    def fix_roi_offset(self, x, y, w, h, **kwargs): ...

    def update_tracker(self, image, index): ...

    def display_tracking_solution(self, window_name, image, solution, **kwargs): ...


class ColorMasking(Tracker):
    def __init__(self):
        super().__init__()
        self.requires_roi = False
        self.lower, self.upper = [], []

    def set_bounds(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def initialize_tracker(self, **kwargs): ...

    def get_mask(self, image):
        # if self.lower == [] or self.upper == []:
        #     raise ValueError(f"lower or upped bounds are not set\n{self.lower=} {self.upper=} ")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower = np.array(self.lower)
        upper = np.array(self.upper)
        return cv2.inRange(hsv, lower, upper)

    def get_contours(self, bw_image: np.ndarray, min_area: int = 30):
        contours, hierarchy = cv2.findContours(
            bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        return contours

    def get_contour_center(self, contours):
        mean_coordinates = []
        for contour in contours:
            points = contour.reshape(-1, 2)
            mean_x = int(np.mean(points[:, 0]))
            mean_y = int(np.mean(points[:, 1]))
            mean_coordinates.append([mean_x, mean_y])
        return np.mean(mean_coordinates, axis=0)

    def update_tracker(self, index, **kwargs):
        mask = self.get_mask(kwargs["image"])

        contours = self.get_contours(mask)
        contour_center = self.get_contour_center(contours)
        self.tracked_coordinates[index] = contour_center
        return contour_center

    def fix_roi_offset(self, x, y, w, h): ...

    def display_tracking_solution(
            self, window_name, image: np.ndarray, solution: Any, **kwargs
    ):
        mask = self.get_mask(image)

        contours = self.get_contours(mask)
        cv2.drawContours(image, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)

        cx, cy = int(solution[0]), int(solution[1])
        cv2.circle(image, (cx, cy), 40, (0, 0, 255), 3)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -3)

        image = cv2.hconcat([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(window_name, image)
        cv2.waitKey(1)


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
            winSize=(25, 25),
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

    def initialize_tracker(self, **kwargs):
        x, y, w, h = kwargs['roi']
        frame = kwargs['frame']

        roi_image = frame[y: y + h, x: x + w]
        roi_image = convert_image(roi_image, "gray")

        init_tracker_image = convert_image(roi_image, "gray")
        self.p0 = cv2.goodFeaturesToTrack(init_tracker_image, mask=None, **self.feature_params)
        if self.p0 is None:
            print("cannot find goodFeaturesToTrack in this roi")
            return
        self.old_gray = convert_image(frame, "gray")

        self.p0 = self.p0 + np.array([x, y])

    def update_tracker(self, index, **kwargs):
        image = kwargs["gray_image"]
        assert len(image.shape) == 2

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

    def display_tracking_solution(
            self, window_name, image: np.ndarray, solution: Any, **kwargs
    ):
        for coordinate in solution:
            x, y = map(int, coordinate.ravel())
            cv2.circle(image, (x, y), 5, (0, 255, 0), 5)

        cv2.imshow(window_name, image)
        cv2.waitKey(1)

        cv2.imshow(window_name, image)
        cv2.waitKey(1)
