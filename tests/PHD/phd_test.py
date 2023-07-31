import os
from collections import namedtuple
from dataclasses import asdict

import numpy as np
import pytest

from src.common import Gaussian, GaussianMixture, WeightedGaussian
from src.configs import GroundTruthConfig, SensorModelConfig
from src.measurement_models import ConstantVelocityMeasurementModel
from src.motion_models import ConstantVelocityMotionModel
from src.run import animate, get_gospa, get_motmetrics, track, visulaize
from src.scenarios.scenario_configs import linear_full_mot
from src.simulator import MeasurementData, ObjectData
from src.trackers.multiple_object_trackers.PHD import GMPHD
from src.utils.get_path import delete_images_dir, get_images_dir


test_env_cases = [
    (
        linear_full_mot,
        ConstantVelocityMotionModel,
        ConstantVelocityMeasurementModel,
        "n_MOT_linear_CV",
    ),
]


@pytest.fixture(scope="session", autouse=True)
def do_something_before_all_tests():
    # prepare something ahead of all tests
    delete_images_dir(__file__)


def generate_environment(config, motion_model, meas_model, *args, **kwargs):
    config = asdict(config)
    ground_truth = GroundTruthConfig(**config)
    motion_model = motion_model(**config)
    sensor_model = SensorModelConfig(**config)
    meas_model = meas_model(**config)
    object_data = ObjectData(ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False)
    meas_data_gen = MeasurementData(object_data=object_data, sensor_model=sensor_model, meas_model=meas_model)
    meas_data = [next(meas_data_gen) for _ in range(len(object_data))]
    env = namedtuple(
        "env",
        [
            "congig",
            "ground_truth",
            "motion_model",
            "sensor_model",
            "meas_model",
            "object_data",
            "meas_data",
        ],
    )
    return env(
        config,
        ground_truth,
        motion_model,
        sensor_model,
        meas_model,
        object_data,
        meas_data,
    )


@pytest.mark.parametrize("config,motion_model,meas_model,name", test_env_cases)
def test_generate_environment(config, motion_model, meas_model, name, *args, **kwargs):
    try:
        env = generate_environment(config, motion_model, meas_model)  # noqa F841
        pass
    except Exception:
        raise AssertionError


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_components_list_operations(config, motion_model, meas_model, name, *args, **kwargs):
    birth_model = GaussianMixture(
        [
            WeightedGaussian(log_weight=np.log(0.03), gaussian=Gaussian(x=pos, P=400 * np.eye(4)))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    )
    gmm_components = GaussianMixture([])
    gmm_components.extend(birth_model)


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_predict_step(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT

    birth_model = GaussianMixture(
        [
            WeightedGaussian(log_weight=np.log(0.03), gaussian=Gaussian(x=pos, P=400 * np.eye(4)))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    )
    tracker = GMPHD(
        meas_model=env.meas_model,
        sensor_model=env.sensor_model,
        motion_model=env.motion_model,
        P_S=config.P_S,
        birth_model=birth_model,
        P_G=P_G,
        w_min=w_minw,
        merging_threshold=merging_threshold,
        M=M,
        P_D=env.sensor_model.P_D,
    )

    # One step predict
    tracker.predict_step(1.0)


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_predict_and_update_step(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-3  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT

    birth_model = GaussianMixture(
        [
            WeightedGaussian(log_weight=np.log(0.03), gaussian=Gaussian(x=pos, P=400 * np.eye(4)))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    )
    tracker = GMPHD(
        meas_model=env.meas_model,
        sensor_model=env.sensor_model,
        motion_model=env.motion_model,
        P_S=config.P_S,
        birth_model=birth_model,
        P_G=P_G,
        P_D=0.6,
        w_min=w_minw,
        merging_threshold=merging_threshold,
        M=M,
    )

    # One step predict
    tracker.predict_step(dt=1.0)

    test_measurements = np.array([config.object_configs[idx].initial_state.x[0:2] for idx in range(3)])
    # One step update
    tracker.update(z=test_measurements)


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_estimate(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT
    P_D = 0.9

    birth_model = GaussianMixture(
        [
            WeightedGaussian(log_weight=np.log(0.03), gaussian=Gaussian(x=pos, P=400 * np.eye(4)))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    )
    tracker = GMPHD(
        meas_model=env.meas_model,
        sensor_model=env.sensor_model,
        motion_model=env.motion_model,
        P_S=config.P_S,
        birth_model=birth_model,
        P_G=P_G,
        P_D=P_D,
        w_min=w_minw,
        merging_threshold=merging_threshold,
        M=M,
    )

    # One step predict
    tracker.predict_step(dt=1.0)

    test_measurements = np.array([config.object_configs[idx].initial_state.x[0:2] for idx in range(3)])
    # One step update
    tracker.update(z=test_measurements)

    # One step estimate
    estimates = tracker.PHD_estimator()  # noqa F841


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_normal(config, motion_model, meas_model, name, *args, **kwargs):
    birth_model = GaussianMixture(
        [
            WeightedGaussian(log_weight=np.log(0.03), gaussian=Gaussian(x=pos, P=400 * np.eye(4)))
            for pos in [
                np.array([0, 0, 0, 0]),
                np.array([400, -600, 0, 0]),
                np.array([-800, -200, 0, 0]),
                np.array([-200, 800, 0, 0]),
            ]
        ]
    )
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.9  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 4  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT
    P_D = 0.99

    tracker_phd = GMPHD(
        meas_model=env.meas_model,
        sensor_model=env.sensor_model,
        motion_model=env.motion_model,
        P_S=config.P_S,
        birth_model=birth_model,
        P_G=P_G,
        P_D=P_D,
        w_min=w_minw,
        merging_threshold=merging_threshold,
        M=M,
    )
    filepath = get_images_dir(__file__) + "/" + tracker_phd.__class__.__name__ + "-" + name
    tracker_estimations = track(env.object_data, env.meas_data, tracker_phd)
    visulaize(env.object_data, env.meas_data, tracker_estimations, filepath)
    gospa = get_gospa(env.object_data, tracker_estimations)
    motmetrics = get_motmetrics(env.object_data, tracker_estimations)  # noqa F841
    assert np.mean(gospa) < 2000
    if os.getenv("ANIMATE", "False") == "True":
        animate(env.object_data, env.meas_data, tracker_estimations, filepath)
