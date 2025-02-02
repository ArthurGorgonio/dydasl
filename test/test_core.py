import sys
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from numpy import array

from src.dydasl.dydasl_core import Dydasl


class SelfFlexConMock(Mock):
    def __init__(self, cr, threshold):
        super().__init__()
        self.cr = cr
        self.threshold = threshold


class EnsembleMock(Mock):
    def __init__(self, ssl_alg=None, **kwargs):
        super().__init__()

        self.ensemble = []
        self.is_weights = False
        self.ssl_params = SelfFlexConMock(
            **kwargs or {
                'cr': 0.1,
                'threshold': 0.9,
            }
        )

    def fit_single_classifier(self, pos, x, y):
        ...

    def add_classifier(self, classifier, need_train):
        self.ensemble.append(classifier)

    def predict_ensemble(self, chunk):
        ...


class DetectorMock(Mock):
    def __init__(self, drift: bool = False):
        super().__init__()
        self.drift = drift
        self.detection_threshold = 0.8
        self.detector_type = 'threshold'

    def detect(self, y_pred, second):
        return self.drift


class ReactorMock(Mock):
    def react(self, ensemble, X, y):
        ...


class MetricsMock(Mock):
    def accuracy_score(self, y_true, y_pred, nclasses=''):
        return 1.0

    def f1_score(
        self,
        y_true,
        y_pred,
        average='',
        nclasses='',
        zero_division=1.0,
    ):
        return 1.0

    def kappa(self, y_true, y_pred, nclasses=''):
        return 1.0


class DateStreamMock(MagicMock):
    def __init__(self, instances = [True, True, False]):
        super().__init__()
        self.instances = instances
        self.i = -1

    def next_sample(self, size):
        return [], []

    def has_more_samples(self):
        self.i += 1

        return self.instances[self.i]


class TestDydasl(TestCase):
    @patch('src.dydasl.dydasl_core.issubclass')
    def setUp(self, subclass):
        subclass.return_value = True
        self.core = Dydasl(EnsembleMock, DetectorMock, ReactorMock)
        self.core.configure_params(SelfFlexConMock)

    @patch('src.dydasl.dydasl_core.issubclass')
    def test_init_should_return_valid_class_instance_when_args_are_valid(
        self,
        subclass,
    ):
        subclass.return_value = True
        core = Dydasl(EnsembleMock, DetectorMock, ReactorMock)
        self.assertEqual(
            [
                core.ensemble,
                core.detector,
                core.reactor,
                core.chunk_size,
                core.metrics,
                core.metrics_calls,
            ],
            [
                EnsembleMock,
                DetectorMock,
                ReactorMock,
                500,
                {},
                {}
            ]
        )

    @patch('src.dydasl.dydasl_core.Ensemble')
    def test_configure_param_should_return_valid_configurations_when_args_are_valid(  # NOQA
        self,
        custom_ensemble,
    ):
        expected_output = {
            'ssl': {
                'cr': 0.05,
                'threshold': 0.95
            },
            'detector': {
                'threshold': 0.2
            },
        }

        ssl_params = {
            'cr': 0.05,
            'threshold': 0.95
        }
        detector_params = {
            'threshold': 0.2
        }
        reactor_params = {}
        custom_ensemble.return_value = EnsembleMock(
            SelfFlexConMock,
            **ssl_params
        )
        core = Dydasl(custom_ensemble)
        core.configure_params(SelfFlexConMock, ssl_params, detector_params, reactor_params)

        self.assertEqual(
            {
                'ssl': {
                    'cr': core.ensemble.ssl_params.cr,
                    'threshold': core.ensemble.ssl_params.threshold
                },
                'detector': {
                    'threshold': core.detector.detection_threshold
                },
            },
            expected_output
        )

    @patch('src.dydasl.dydasl_core.Ensemble')
    def test_configure_ensemble_should_create_ensemble_when_all_in_configured_with_four_classifiers(  # NOQA
        self,
        ensemble
    ):
        base_classifiers = [1, 2, 3, 4]
        self.core = Dydasl(ensemble)
        self.core.configure_params(ensemble, {}, {}, {})
        self.core.configure_classifier(base_classifiers)

        self.assertEqual(self.core.ensemble.add_classifier.call_count, 4)

    def test_add_metrics_should_return_dict_with_metric_callable(self):
        metrics = MetricsMock()
        self.core.add_metrics('acc', metrics.accuracy_score)
        self.core.add_metrics('f1', metrics.f1_score)

        self.assertEqual(
            self.core.metrics_calls,
            {
                'acc': metrics.accuracy_score,
                'f1': metrics.f1_score
            }
        )

        self.assertEqual(
            self.core.metrics,
            {
                'acc': [],
                'f1': []
            }
        )

    def test_evaluate_metrics_should_return_two_values_when_two_metrics_are_used(  # NOQA
        self,
    ):
        evaluate = MetricsMock()
        self.core.add_metrics('acc', evaluate.accuracy_score)
        #self.core.add_metrics('f1', evaluate.f1_score)

        y_true = [1, 1, 0, 0, 0, 1]
        y_pred = [1, 1, 0, 0, 0, 1]

        self.core._evaluate_metrics(y_true, y_pred, 2)

        self.assertEqual(self.core.metrics, {'acc': [1.0]})

    @patch('src.dydasl.dydasl_core.accuracy_score')
    @patch('src.dydasl.dydasl_core.confusion_matrix')
    @patch('src.dydasl.dydasl_core.Dydasl.run_first_it')
    @patch('src.dydasl.dydasl_core.Dydasl._log_iteration_info')
    def test_run_should_call_log_iteration_info_each_iteration(
        self,
        logger,
        first_it,
        cm,
        acc,
    ):
        cm.return_value = array([[50, 0], [0, 50]])
        acc.return_value = 1.0
        self.core.detector.drift = True
        stream_data = DateStreamMock()
        stream_data.sample_idx = []
        self.core.run(stream_data)

        self.assertEqual(first_it.call_count, 3)
        self.assertEqual(acc.call_count, 2)
        self.assertEqual(cm.call_count, 2)
        self.assertEqual(logger.call_count, 2)

    @patch('src.dydasl.dydasl_core.accuracy_score')
    @patch('src.dydasl.dydasl_core.confusion_matrix')
    @patch('src.ssl.Ensemble.fit_single_classifier')
    @patch('src.dydasl.dydasl_core.Dydasl._log_iteration_info')
    def test_run_should_call_classifier_addition_when_drift_is_detected(
        self,
        logger,
        first_it,
        cm,
        acc,
    ):
        cm.return_value = array([[50, 0], [0, 50]])
        acc.return_value = 1.0
        self.core.detector.drift = False

        stream_data = DateStreamMock([True for _ in range(9)] + [False])
        stream_data.sample_idx = []
        self.core.run(stream_data)

        self.assertEqual(len(self.core.ensemble.ensemble), 10)
        self.assertEqual(acc.call_count, 9)
        self.assertEqual(cm.call_count, 9)
        self.assertEqual(logger.call_count, 9)

    @patch('src.dydasl.dydasl_core.accuracy_score')
    @patch('src.dydasl.dydasl_core.confusion_matrix')
    @patch('src.ssl.Ensemble.fit_single_classifier')
    @patch('src.dydasl.dydasl_core.Dydasl._log_iteration_info')
    def test_run_should_not_add_classifier_when_drift_is_not_detected(
        self,
        logger,
        first_it,
        cm,
        acc,
    ):
        cm.return_value = array([[50, 0], [0, 50]])
        acc.return_value = 1.0

        self.core.detector.drift = False

        stream_data = DateStreamMock([True for _ in range(9)] + [False])
        stream_data.sample_idx = []
        self.core.run(stream_data, 'drift')

        self.assertEqual(len(self.core.ensemble.ensemble), 1)
        self.assertEqual(acc.call_count, 9)
        self.assertEqual(cm.call_count, 9)
        self.assertEqual(logger.call_count, 9)

    @patch('src.utils.Log.write_archive_output')
    def test_log(self, logger):
        metrics = MetricsMock()
        self.core.add_metrics('acc', metrics.accuracy_score)
        self.core.add_metrics('f1', metrics.f1_score)
        self.core.add_metrics('kappa', metrics.kappa)

        y_true = array([1, 1, 0, 0, 0, 1])
        y_pred = array([1, 1, 0, 0, 0, 1])

        self.core._evaluate_metrics(y_true, y_pred)

        self.core._log_iteration_info(600, 1000, 0.2321)

        logger.assert_called_once()
