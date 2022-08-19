from typing import Any, Dict

from src.detection.interfaces.ichunk import IChunk


class FixedThreshold(IChunk):
    """Classe de detecção de drift com treshold fixo"""
    def __init__(self, **params: Dict[str, Any]):
        super().__init__(**params)

    def detect(self, chunk: float) -> bool:
        if chunk < self.detection_threshold:
            self.increase_counter()
            self.drift = True

            return self.drift
        self.drift = False
        return self.drift

    def reset_params(self):
        ...

    def __reset_all(self):
        self.drift_counter = 0
