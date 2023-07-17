from collections import namedtuple
from dataclasses import asdict

import numpy as np
import pytest

import mot
from mot.common.gaussian_density import GaussianDensity
from mot.configs import GroundTruthConfig, SensorModelConfig
from mot.measurement_models import ConstantVelocityMeasurementModel
from mot.motion_models import ConstantVelocityMotionModel
from mot.scenarios.scenario_configs import linear_full_mot
from mot.simulator import MeasurementData, ObjectData
from mot.utils.get_path import get_images_dir
from mot.utils.visualizer import Plotter
from mot.trackers.multiple_object_trackers.PHD.gm_phd import GMPHD

test_env_cases = [
    (
        linear_full_mot,
        ConstantVelocityMotionModel,
        ConstantVelocityMeasurementModel,
        "n MOT linear (CV)",
    ),
]


def generate_environment(config, motion_model, meas_model, *args, **kwargs):
    config = asdict(config)
    ground_truth = GroundTruthConfig(**config)
    motion_model = motion_model(**config)
    sensor_model = SensorModelConfig(**config)
    meas_model = meas_model(**config)
    object_data = ObjectData(ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False)
    meas_data = MeasurementData(object_data=object_data, sensor_model=sensor_model, meas_model=meas_model)
    meas_data = [next(meas_data) for _ in range(ground_truth.total_time)]

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
    birth_model = GaussianDensity(
        np.array([[0, 0, 0, 0], [400, -600, 0, 0], [-800, -200, 0, 0], [-200, 800, 0, 0]]),
        np.full((4, 4, 4), 400 * np.eye(4)),
        np.full(4, np.log(0.03)),
    )
    return birth_model


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_predict_step(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT

    birth_model = GaussianDensity(
        np.array([[0, 0, 0, 0], [400, -600, 0, 0], [-800, -200, 0, 0], [-200, 800, 0, 0]]),
        np.full((4, 4, 4), 400 * np.eye(4)),
        np.full(4, np.log(0.03)),
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
        P_D=0.6,
    )

    # One step predict
    tracker.predict_step(dt=1.0)


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_predict_and_update_step(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-3  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT

    birth_model = GaussianDensity(
        np.array([[0, 0, 0, 0], [400, -600, 0, 0], [-800, -200, 0, 0], [-200, 800, 0, 0]]),
        np.full((4, 4, 4), 400 * np.eye(4)),
        np.full(4, np.log(0.03)),
    )
    tracker = mot.trackers.multiple_object_trackers.PHD.gm_phd.GMPHD(
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

    test_measurements = env.meas_data[0][1]
    # One step update
    tracker.update(test_measurements)


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_estimate(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT
    P_D = 0.9

    birth_model = GaussianDensity(
        np.array([[0, 0, 0, 0], [400, -600, 0, 0], [-800, -200, 0, 0], [-200, 800, 0, 0]]),
        np.full((4, 4, 4), 400 * np.eye(4)),
        np.full(4, np.log(0.03)),
    )
    tracker = mot.trackers.multiple_object_trackers.PHD.gm_phd.GMPHD(
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
    test_measurements = env.meas_data[0][1]
    # One step update
    tracker.update(test_measurements)

    # One step estimate
    estimates = tracker.estimate()  # noqa F841


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_tracker_normal(config, motion_model, meas_model, name, *args, **kwargs):
    birth_model = GaussianDensity(
        np.array([[0, 0, 0, 0], [400, -600, 0, 0], [-800, -200, 0, 0], [-200, 800, 0, 0]]),
        np.full((4, 4, 4), 400 * np.eye(4)),
        np.full(4, np.log(0.03)),
    )
    env = generate_environment(config, motion_model, meas_model)

    # Single object tracker parameter setting
    P_G = 0.9  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 4  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT
    P_D = 0.99

    tracker = mot.trackers.multiple_object_trackers.PHD.gm_phd.GMPHD(
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
    a = birth_model + birth_model + birth_model
    c = a[:2]
    tracker_estimations = []
    # meas_data = [next(meas_data_gen) for _ in range(ground_truth.total_time)]
    for measurement in env.meas_data:
        estimations = tracker.step(measurement[1])
        if estimations:
            for idx in range(len(estimations)):
                tracker_estimations.append(estimations[idx])
        else:
            tracker_estimations.append(None)

    # Animator.animate(
    #     [env.meas_data, env.object_data, list(estimations)],
    #     title=name,
    #     filename=get_images_dir(__file__) + "/" + "meas_data_and_obj_data" + ".gif",
    # )

    # Plotter.plot_several(
    #     [env.meas_data, env.object_data, list(estimations)],
    #     out_path=get_images_dir(__file__) + "/" + "meas_data_and_obj_data_and_estimations" + ".png",
    # )
