from typing import Any, Dict

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector

from src.detection.interfaces.drift_detector import DriftDetector
from src.utils import compare_labels


class LiteratureDetector(DriftDetector):
    """
    Classe de detecção de drift Wrapper para os detectores da
    literatura.
    """
    def __init__(
        self,
        drift_detector: BaseDriftDetector,
        **params: Dict[str, Any]
    ):
        super().__init__(params.get('detector_type'))

        if issubclass(drift_detector, BaseDriftDetector):
            self.drift_detector = drift_detector()
            self.__name__ = self.drift_detector.__class__.__name__
        else:
            raise TypeError(
                f"Missing object type:\n"
                f"Expected 'BaseDriftDetector' in obj hierarchy,"
                f"however found {type(self.drift_detector).__name__}"
            )

    def detect(self, chunk, y_pred) -> bool:
        options = ['ADWIN', 'DDM', 'EDDM', 'HDDM_A', 'HDDM_W', 'PageHinkley']

        if self.__name__ in options:
            return self._labels_detect(chunk, y_pred)

        if self.__name__ == 'KSWIN':
            return self._kswin(chunk)

        return False


    def _labels_detect(self, chunk, y_pred):
        labels = []

        if self.__name__ in ('ADWIN', 'PageHinkley'):
            labels = compare_labels(chunk, y_pred)
        elif self.__name__ in ('DDM', 'EDDM', 'HDDM_A', 'HDDM_W'):
            labels = compare_labels(chunk, y_pred, False)

        for i in labels:
            self.drift_detector.add_element(i)

            if self.drift_detector.detected_change():
                self.drift = True

                return self.drift

        self.drift = False

        return self.drift

    def _kswin(self, chunk):
        self.drift_detector.add_element(chunk)

        if self.drift_detector.detected_change():
            self.drift = True

            return self.drift

        self.drift = False

        return self.drift
