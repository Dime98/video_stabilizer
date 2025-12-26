from abc import ABC, abstractmethod
import cv2
import numpy as np
from misc_functions import convert_image


class Tracker(ABC):
    def __init__(self):
        self._active = True

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, val):
        self._active = val

    @staticmethod
    def factory(method):
        if method == "optical_flow":
            return OpticalFlow
        if method == "tracker":
            raise NotImplementedError
        else:
            raise ValueError(f"'{method}' is not an option")

    @abstractmethod
    def initialize_tracker(self, image): ...

    @abstractmethod
    def update_tracker(self, image): ...

    @abstractmethod
    def fix_roi_offset(self, x, y, w, h): ...


class ColorMasking(Tracker):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def initialize_tracker(self, image): ...

    @abstractmethod
    def update_tracker(self, image): ...

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

    def update_tracker(self, image):
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
        return good_points
