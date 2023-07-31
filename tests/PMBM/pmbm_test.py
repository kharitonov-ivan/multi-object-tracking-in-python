import logging
import os
import pprint

import numpy as np
import pytest

from src.common import GaussianDensity
from src.configs import GroundTruthConfig, SensorModelConfig
from src.measurement_models import ConstantVelocityMeasurementModel
from src.motion_models import ConstantVelocityMotionModel
from src.run import animate, get_gospa, get_motmetrics, track, visulaize
from src.scenarios.object_motion_scenarious import (
    many_objects_linear_motion_delayed,
    single_object_linear_motion,
    single_static_object,
    two_objects_linear_motion_delayed,
    two_static_objects,
)
from src.simulator import MeasurementData, ObjectData
from src.trackers.multiple_object_trackers.PMBM.common.birth_model import (
    StaticBirthModel,
)
from src.trackers.multiple_object_trackers.PMBM.pmbm import PMBM
from src.utils import delete_images_dir
from src.utils.get_path import get_images_dir

from .params.birth_model import birth_model_params


@pytest.fixture(params=[0.9])
def scenario_detection_probability(request):
    yield request.param


@pytest.fixture(
    params=[
        0.1,
        10.0,
    ]
)
def scenario_clutter_rate(request):
    yield request.param


@pytest.fixture(
    params=[
        ("single_static_object", single_static_object),
        ("two_static_objects", two_static_objects),
        ("single_object_linear_motion", single_object_linear_motion),
        ("two_objects_linear_motion_delayed", two_objects_linear_motion_delayed),
        ("many_objects_linear_motion_delayed", many_objects_linear_motion_delayed),
    ]
)
def object_motion_fixture(request):
    yield request.param


@pytest.fixture(params=[birth_model_params])
def birth_model(request):
    yield request.param


@pytest.fixture(scope="session", autouse=True)
def do_something_before_all_tests():
    # prepare something ahead of all tests
    delete_images_dir(__file__)


@pytest.fixture(
    params=[
        0.99,
    ]
)
def scenario_survival_probability(request):
    yield request.param


def test_synthetic_scenario(
    object_motion_fixture,
    scenario_detection_probability,
    scenario_clutter_rate,
    birth_model,
    scenario_survival_probability,
):
    name, object_motion_fixture = object_motion_fixture
    # Create linear motion model
    dt = 1.0
    sigma_q = 10.0
    motion_model = ConstantVelocityMotionModel(dt, sigma_q)

    # Create linear measurement model
    sigma_r = 10.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r)

    # Create ground truth model
    simulation_steps = 100

    # Create sensor model - range/bearing measurement
    range_c = np.array([[-1000, 1000], [-1000, 1000]])
    sensor_model = SensorModelConfig(P_D=scenario_detection_probability, lambda_c=scenario_clutter_rate, range_c=range_c)

    # Generate true object data (noisy or noiseless) and measurement data
    ground_truth = GroundTruthConfig(object_motion_fixture, total_time=simulation_steps)
    object_data = ObjectData(ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False)
    meas_data_gen = MeasurementData(object_data=object_data, sensor_model=sensor_model, meas_model=meas_model)
    meas_data = [next(meas_data_gen) for _ in range(simulation_steps)]
    logging.debug(f"object motion config {pprint.pformat(object_motion_fixture)}")

    # Object tracker parameter setting
    gating_percentage = 1.0  # gating size in percentage
    max_hypothesis_kept = 100  # maximum number of hypotheses kept
    existense_probability_threshold = 0.8

    tracker_pmbm = PMBM(
        meas_model=meas_model,
        sensor_model=sensor_model,
        motion_model=motion_model,
        birth_model=StaticBirthModel(birth_model),
        max_number_of_hypotheses=max_hypothesis_kept,
        gating_percentage=gating_percentage,
        detection_probability=scenario_detection_probability,
        survival_probability=scenario_survival_probability,
        existense_probability_threshold=existense_probability_threshold,
        track_history_length_threshold=5,
        density=GaussianDensity,
        initial_PPP_intensity=birth_model,
    )

    filepath = (
        get_images_dir(__file__)
        + "/"
        + tracker_pmbm.__class__.__name__
        + "-"
        + name
        + "-"
        + f"P_S={scenario_survival_probability}-P_D={scenario_detection_probability}-lambda_c={scenario_clutter_rate}"
    )
    tracker_estimations = track(object_data, meas_data, tracker_pmbm)
    visulaize(object_data, meas_data, tracker_estimations, filepath)
    gospa = get_gospa(object_data, tracker_estimations)
    motmetrics = get_motmetrics(object_data, tracker_estimations)  # noqa F841
    assert np.mean(gospa) < 2000
    if os.getenv("ANIMATE", "False") == "True":
        animate(object_data, meas_data, tracker_estimations, filepath)
