from collections import namedtuple
from dataclasses import asdict

import numpy as np
import pytest

from src.common.state import Gaussian
from src.configs import GroundTruthConfig, SensorModelConfig
from src.measurement_models import ConstantVelocityMeasurementModel
from src.motion_models import ConstantVelocityMotionModel
from src.run import animate, visulaize
from src.scenarios.scenario_configs import linear_full_mot
from src.simulator import MeasurementData, ObjectData
from src.utils.get_path import delete_images_dir, get_images_dir


test_env_cases = [
    (
        linear_full_mot,
        ConstantVelocityMotionModel,
        ConstantVelocityMeasurementModel,
        "n MOT linear (CV)",
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
    estimations = [
        {
            idx: Gaussian(x=pos, P=400 * np.eye(4))
            for idx, pos in enumerate(
                [
                    np.array([0, 0, 0, 0]),
                    np.array([400, -600, 0, 0]),
                    np.array([-800, -200, 0, 0]),
                    np.array([-200, 800, 0, 0]),
                ]
            )
        }
    ] * 10
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
            "estimations",
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
        estimations,
    )


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_plot_object_data(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)
    visulaize(env.object_data, None, None, get_images_dir(__file__) + "/" + "obj_data" + ".png")


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_plot_meas_data(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)
    visulaize(None, env.meas_data, None, get_images_dir(__file__) + "/" + "meas_data" + ".png")


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_plot_object_meas_data(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)
    visulaize(env.object_data, env.meas_data, None, get_images_dir(__file__) + "/" + "obj_data_and_meas_data" + ".png")


def test_plot_one_gaussian(*args, **kwargs):
    gaussian = Gaussian(x=np.array([0, 0, 10, 10]), P=np.diag([400, 200, 0, 0]))
    visulaize(None, None, [{0: gaussian}], get_images_dir(__file__) + "/" + "one_gaussian" + ".png")


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_plot_gaussians(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)
    visulaize(env.object_data, env.meas_data, env.estimations, get_images_dir(__file__) + "/" + "gaussians" + ".png")


@pytest.mark.parametrize("config, motion_model, meas_model, name", test_env_cases)
def test_animate_input_data(config, motion_model, meas_model, name, *args, **kwargs):
    env = generate_environment(config, motion_model, meas_model)
    animate(env.object_data, env.meas_data, env.estimations, get_images_dir(__file__) + "/" + "animate" + ".gif")
