import os
from dataclasses import asdict

import pytest

from src.configs import GroundTruthConfig, SensorModelConfig
from src.measurement_models import (
    ConstantVelocityMeasurementModel,
    RangeBearingMeasurementModel,
)
from src.motion_models import ConstantVelocityMotionModel, CoordinateTurnMotionModel
from src.run import animate, visulaize
from src.scenarios.scenario_configs import (
    linear_n_mot,
    linear_sot,
    nonlinear_n_mot,
    nonlinear_sot,
)
from src.simulator import MeasurementData
from src.simulator.object_data_generator import ObjectData
from src.utils.get_path import delete_images_dir, get_images_dir


dir_path = os.path.dirname(os.path.realpath(__file__))

test_data = [
    (
        linear_sot,
        ConstantVelocityMotionModel,
        "linear_sot_CV.png",
        ConstantVelocityMeasurementModel,
    ),
    (
        nonlinear_sot,
        CoordinateTurnMotionModel,
        "nonlinear_sot_CT.png",
        RangeBearingMeasurementModel,
    ),
    (
        linear_n_mot,
        ConstantVelocityMotionModel,
        "linear_n_mot_CV.png",
        ConstantVelocityMeasurementModel,
    ),
    (
        nonlinear_n_mot,
        CoordinateTurnMotionModel,
        "nonlinear_n_mot_CT.png",
        RangeBearingMeasurementModel,
    ),
]


@pytest.fixture(scope="session", autouse=True)
def do_something_before_all_tests():
    # prepare something ahead of all tests
    delete_images_dir(__file__)


@pytest.mark.parametrize("config, motion_model, output_image_name, meas_model", test_data)
def test_linear_model_without_noise(config, motion_model, output_image_name, meas_model):
    config = asdict(config)
    ground_truth = GroundTruthConfig(config["object_configs"], config["total_time"])
    motion_model = motion_model(**config)

    object_data = ObjectData(ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False)

    sensor_model = SensorModelConfig(**config)

    meas_model = meas_model(**config)
    meas_data_gen = MeasurementData(object_data=object_data, sensor_model=sensor_model, meas_model=meas_model)
    meas_data = [next(meas_data_gen) for _ in range(len(object_data))]
    output_image_name = get_images_dir(__file__) + "/" + output_image_name
    visulaize(object_data, meas_data, None, output_image_name)
    if os.getenv("ANIMATE", "False") == "True":
        animate(object_data, meas_data, None, output_image_name)
