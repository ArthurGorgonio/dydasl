from numpy import ndarray
from skmultiflow.trees import HoeffdingTreeClassifier as HT

from src.reaction.interfaces.reactor import Reactor
from src.ssl.ensemble import Ensemble


class VolatileExchange(Reactor):
    """
    Módulo de reação ao drift por troca todos os classificadores que
    não atingem o limiar.
    """
    def __init__(self, **params):
        self.classifier = params.get("classifier", HT)
        self.thr = params.get("thr", 0.8)
        self.retrain_classifier = params.get("retrain_classifier", True)

    def react(
        self,
        ensemble: Ensemble,
        instances: ndarray,
        labels: ndarray,
    ) -> Ensemble:
        y_pred_classifier = ensemble.measure_ensemble(instances, labels)
        pos = [p for p, acc in enumerate(y_pred_classifier) if acc < self.thr]
        ensemble.swap(
            [self.classifier()],
            pos,
            instances,
            labels,
            self.retrain_classifier
        )

        return ensemble
