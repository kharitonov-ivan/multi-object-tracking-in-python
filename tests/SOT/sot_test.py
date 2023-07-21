import os
from dataclasses import asdict

import numpy as np
import pytest

from src.common import Gaussian
from src.configs import GroundTruthConfig, SensorModelConfig
from src.measurement_models import (
    ConstantVelocityMeasurementModel,
    RangeBearingMeasurementModel,
)
from src.motion_models import ConstantVelocityMotionModel, CoordinateTurnMotionModel
from src.run import animate, get_gospa, get_motmetrics, track, visulaize
from src.scenarios.scenario_configs import linear_sot, nonlinear_sot
from src.simulator import MeasurementData, ObjectData
from src.trackers.single_object_trackers import (
    GaussSumTracker,
    NearestNeighbourTracker,
    ProbabilisticDataAssociationTracker,
)
from src.utils.get_path import delete_images_dir, get_images_dir


@pytest.fixture(scope="session", autouse=True)
def do_something_before_all_tests():
    # prepare something ahead of all tests
    delete_images_dir(__file__)


@pytest.mark.parametrize(
    "config, motion_model, meas_model, name, tracker_initial_state",
    [
        (
            linear_sot,
            ConstantVelocityMotionModel,
            ConstantVelocityMeasurementModel,
            "SOT-linear-case-(CV)",
            Gaussian(x=np.array([-40, -40, 15.0, 5.0]), P=100.0 * np.eye(4)),
        ),
        (
            nonlinear_sot,
            CoordinateTurnMotionModel,
            RangeBearingMeasurementModel,
            "SOT-non-linear-case-(CT)",
            Gaussian(
                x=np.array([0, 0, 10, 0, np.pi / 180]),
                P=np.power(np.diag([1, 1, 1, 1 * np.pi / 180, 1 * np.pi / 180]), 2),
            ),
        ),
    ],
)
@pytest.mark.parametrize("tracker", [NearestNeighbourTracker, GaussSumTracker, ProbabilisticDataAssociationTracker])  # noqa
def test_tracker(config, motion_model, meas_model, name, tracker, tracker_initial_state):
    config = asdict(config)
    ground_truth = GroundTruthConfig(**config)
    motion_model = motion_model(**config)
    sensor_model = SensorModelConfig(**config)
    meas_model = meas_model(**config)

    object_data = ObjectData(ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False)
    meas_data_gen = MeasurementData(object_data=object_data, sensor_model=sensor_model, meas_model=meas_model)
    meas_data = [next(meas_data_gen) for _ in range(99)]
    # Single object tracker parameter setting
    P_G = 0.999  # gating size in percentage

    tracker_sot = tracker(
        meas_model=meas_model,
        sensor_model=sensor_model,
        motion_model=motion_model,
        gating_size=P_G,
        initial_state=tracker_initial_state,
    )
    filepath = get_images_dir(__file__) + "/" + tracker_sot.__class__.__name__ + "-" + name

    tracker_estimations = track(object_data, meas_data, tracker_sot)
    visulaize(object_data, meas_data, tracker_estimations, filepath)
    gospa = get_gospa(object_data, tracker_estimations)
    motmetrics = get_motmetrics(object_data, tracker_estimations)  # noqa F841
    assert np.mean(gospa) < 100
    if os.getenv("ANIMATE", "False") == "True":
        animate(object_data, meas_data, tracker_estimations, filepath)
