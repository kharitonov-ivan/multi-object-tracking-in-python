import os

import pytest

from src.motion_models import ConstantVelocityMotionModel
from src.scenarios.initial_conditions import all_object_scenarios
from src.trackers.n_object_trackers import GlobalNearestNeighboursTracker
from tests.testing_trackers import test_synthetic_scenario  # noqa: F401


@pytest.fixture()
def _file_():
    yield os.path.abspath(__file__)


@pytest.fixture(params=[x for x in all_object_scenarios if x.motion_model == ConstantVelocityMotionModel])  # and
def object_motion_fixture(request):
    yield request.param


@pytest.fixture()
def tracker(request, object_motion_fixture):
    tracker_hyperparams = {
        "P_G": 0.99,  # gating size in percentage
        "w_min": 1e-4,  # hypothesis pruning threshold
        "merging_threshold": 2,  # hypothesis merging threshold
        "M": 100,  # maximum number of hypotheses kept in MHT
        "initial_state": [x.initial_state for x in object_motion_fixture.object_configs],
        "n": len(object_motion_fixture.object_configs),
    }
    yield (GlobalNearestNeighboursTracker, tracker_hyperparams)
