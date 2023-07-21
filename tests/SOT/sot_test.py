import os

import numpy as np
import pytest

from src.common import Gaussian
from src.motion_models import ConstantVelocityMotionModel
from src.scenarios.initial_conditions import all_object_scenarios
from src.trackers.single_object_trackers import (
    GaussSumTracker,
    NearestNeighbourTracker,
    ProbabilisticDataAssociationTracker,
)
from tests.testing_trackers import test_synthetic_scenario  # noqa: F401


@pytest.fixture()
def _file_():
    yield os.path.abspath(__file__)


@pytest.fixture(params=[x for x in all_object_scenarios if (len(x.object_configs) == 1) and (x.motion_model == ConstantVelocityMotionModel)])
def object_motion_fixture(request):
    yield request.param


@pytest.fixture(params=[NearestNeighbourTracker, GaussSumTracker, ProbabilisticDataAssociationTracker])
def tracker(request):
    tracker_hyperparams = {
        "initial_state": Gaussian(x=np.array([-40, -40, 15.0, 5.0]), P=100.0 * np.eye(4)),
    }
    yield (request.param, tracker_hyperparams)
