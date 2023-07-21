import os

import pytest

from src.common import GaussianDensity
from src.motion_models import ConstantVelocityMotionModel
from src.scenarios.initial_conditions import all_object_scenarios
from src.trackers.multiple_object_trackers.PMBM.common.birth_model import (
    StaticBirthModel,
)
from src.trackers.multiple_object_trackers.PMBM.pmbm import PMBM
from tests.testing_trackers import test_synthetic_scenario  # noqa: F401

from .params.birth_model import birth_model_params


@pytest.fixture()
def _file_():
    yield os.path.abspath(__file__)


@pytest.fixture(params=[x for x in all_object_scenarios if x.motion_model == ConstantVelocityMotionModel])
def object_motion_fixture(request):
    yield request.param


@pytest.fixture()
def tracker(request):
    tracker_hyperparams = {
        "survival_probability": 0.99,
        "existense_probability_threshold": 0.8,
        "track_history_length_threshold": 5,
        "density": GaussianDensity,
        "initial_PPP_intensity": birth_model_params,
        "birth_model": StaticBirthModel(birth_model_params),
        "max_number_of_hypotheses": 100,
        "gating_percentage": 0.999,
        "detection_probability": 0.9,
    }
    yield (PMBM, tracker_hyperparams)
