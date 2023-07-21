import os

import pytest

from src.measurement_models import ConstantVelocityMeasurementModel
from src.motion_models import ConstantVelocityMotionModel
from src.scenarios.initial_conditions import all_object_scenarios


dir_path = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(params=[x for x in all_object_scenarios if x.motion_model == ConstantVelocityMotionModel])
def object_motion_fixture(request):
    yield request.param


@pytest.fixture(params=[ConstantVelocityMeasurementModel(sigma_r=10.0)])
def env_measurement_model(request):
    yield request.param


@pytest.fixture
def env_clutter_rate(request):
    yield 0.0


@pytest.fixture
def env_detection_probability(request):
    yield 1.0


@pytest.fixture
def tracker():
    yield None, None
