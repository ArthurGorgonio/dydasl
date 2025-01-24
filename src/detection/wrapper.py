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
        labels: list = []

        if self.__name__ in ('ADWIN', 'KSWIN', 'PageHinkley'):
            labels = compare_labels(chunk, y_pred)
        elif self.__name__ in ('DDM', 'EDDM', 'HDDM_A', 'HDDM_W'):
            labels = compare_labels(chunk, y_pred, False)

        return self._labels_detect(labels)

    def _labels_detect(self, labels):

        for i in labels:
            self.drift_detector.add_element(i)

            if self.drift_detector.detected_change():
                self.drift = True

                return self.drift

        self.drift = False

        return self.drift
