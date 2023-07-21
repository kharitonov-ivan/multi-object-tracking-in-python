import os
from dataclasses import asdict

import numpy as np
import pytest

from src.configs import GroundTruthConfig, SensorModelConfig
from src.measurement_models import ConstantVelocityMeasurementModel
from src.motion_models import ConstantVelocityMotionModel
from src.run import animate, get_gospa, get_motmetrics, track, visulaize
from src.scenarios.initial_conditions import linear_n_mot_object_life_params
from src.scenarios.scenario_configs import linear_n_mot
from src.simulator import MeasurementData, ObjectData
from src.trackers.n_object_trackers import GlobalNearestNeighboursTracker
from src.utils.get_path import delete_images_dir, get_images_dir


@pytest.fixture(scope="session", autouse=True)
def do_something_before_all_tests():
    # prepare something ahead of all tests
    delete_images_dir(__file__)


@pytest.mark.parametrize(
    "config, motion_model, meas_model, name, tracker_initial_states",
    [
        (
            linear_n_mot,
            ConstantVelocityMotionModel,
            ConstantVelocityMeasurementModel,
            "n MOT linear (CV)",
            [x.initial_state for x in linear_n_mot_object_life_params],
        ),
    ],
)
@pytest.mark.parametrize("tracker", [(GlobalNearestNeighboursTracker)])
def test_tracker(config, motion_model, meas_model, name, tracker, tracker_initial_states):
    config = asdict(config)
    ground_truth = GroundTruthConfig(**config)
    motion_model = motion_model(**config)
    sensor_model = SensorModelConfig(**config)
    meas_model = meas_model(**config)

    object_data = ObjectData(ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False)
    meas_data_gen = MeasurementData(object_data=object_data, sensor_model=sensor_model, meas_model=meas_model)
    meas_data = [next(meas_data_gen) for _ in range(99)]
    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT

    tracker_mot = GlobalNearestNeighboursTracker(
        meas_model=meas_model,
        sensor_model=sensor_model,
        motion_model=motion_model,
        P_G=P_G,
        w_min=w_minw,
        merging_threshold=merging_threshold,
        M=M,
        initial_state=tracker_initial_states,
    )
    filepath = get_images_dir(__file__) + "/" + tracker_mot.__class__.__name__ + "-" + name

    tracker_estimations = track(object_data, meas_data, tracker_mot)
    visulaize(object_data, meas_data, tracker_estimations, filepath)
    gospa = get_gospa(object_data, tracker_estimations)
    motmetrics = get_motmetrics(object_data, tracker_estimations)  # noqa F841
    assert np.mean(gospa) < 300

    if os.getenv("ANIMATE", "False") == "True":
        animate(object_data, meas_data, tracker_estimations, filepath)
