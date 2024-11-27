import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import f1_score
from skmultiflow.data import DataStream
from skmultiflow.drift_detection import (
    ADWIN,
    DDM,
    EDDM,
    HDDM_A,
    HDDM_W,
    KSWIN,
    PageHinkley,
)

from src.detection import (
    FixedThreshold,
    LiteratureDetector,
    Normal,
    Statistical,
)
from src.dydasl.dydasl_core import Dydasl
from src.reaction import (
    Exchange,
    Pareto,
    VolatileExchange,
)
from src.ssl import (
    Ensemble,
    SelfFlexCon,
)
from src.utils import Log

if __name__ == '__main__':
    detectors = [
        Statistical,
        Normal,
        FixedThreshold,
        ADWIN,
        DDM,
        EDDM,
        HDDM_A,
        HDDM_W,
        KSWIN,
        PageHinkley,
    ]
    reactors = [
        Exchange,
        Pareto,
        VolatileExchange,
    ]

    # datasets = glob('datasets/*.csv')
    # datasets.sort()
    datasets = [
        'bank-full_converted.csv',
        'iot-network_converted.csv'
    ]

    training_modules = [
        "drift",
        "simple",
        # "full",
    ]

    os.makedirs('running_current', exist_ok=True)

    for tr_module in training_modules:
        for dataset in datasets:
            for reactor in reactors:
                for detector in detectors:
                    dydasl = Dydasl(Ensemble, detector, reactor)
                    dydasl.configure_params(
                        ssl_algorithm=SelfFlexCon,
                        params_training={
                            'is_weight': True,
                        },
                        params_detector={},
                        params_reactor={
                            "pareto_strategy": "classifier_ensemble",
                            "q_measure": {
                                "absolute": True,
                                "average": True,
                            },
                        },
                    )
                    dydasl.add_metrics("acc", accuracy_score)
                    dydasl.add_metrics("f1", f1_score)
                    dydasl.add_metrics("kappa", kappa)

                    dydasl.reset()
                    print(dataset)
                    dataframe = pd.read_csv(
                        dataset

                        if "datasets/" in dataset
                        else "datasets/" + dataset
                    )
                    dt_name = dataset.split(".", maxsplit=1)[0].split("/")[-1]

                    if isinstance(dydasl.detector, LiteratureDetector):
                        detection_name = f"-{dydasl.detector.drift_detector.__class__.__name__}"
                    else:
                        detection_name = f"-{dydasl.detector.__class__.__name__}"

                    Log().filename = {
                        "data_name": dt_name,
                        "method_name": f"{tr_module.upper()[0]}"
                        f"W-{type(dydasl.reactor).__name__[0]}",
                    }
                    # depende do dataset
                    dim = dataframe.shape
                    array = dataframe.values
                    instances = array[: dim[0], : dim[1] - 1]
                    target = array[: dim[0], dim[1] - 1]
                    class_set = np.unique(target)
                    class_count = np.unique(target).shape[0]
                    stream = DataStream(
                        instances,
                        target,
                        target_idx=-1,
                        n_targets=class_count,
                        cat_features=None,  # Categorical features?
                        name=None,
                        allow_nan=True,
                    )
                    Log().write_archive_header()
                    dydasl.run(
                        stream,
                        tr_module,
                        std=True if tr_module.lower() == 'full' else False
                    )
