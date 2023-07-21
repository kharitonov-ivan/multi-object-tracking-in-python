import os

import numpy as np
import pytest

from src.common import Gaussian, GaussianMixture, WeightedGaussian
from src.motion_models import ConstantVelocityMotionModel
from src.scenarios.initial_conditions import all_object_scenarios
from src.trackers.multiple_object_trackers.PHD import GMPHD

from ..testing_trackers import test_synthetic_scenario  # noqa: F401


@pytest.fixture()
def _file_():
    yield os.path.abspath(__file__)


@pytest.fixture(params=[x for x in all_object_scenarios if x.motion_model == ConstantVelocityMotionModel])
def object_motion_fixture(request):
    yield request.param


@pytest.fixture()
def tracker(
    request,
    object_motion_fixture,
):
    tracker_hyperparams = {
        "P_G": 0.99,  # gating size in percentage
        "w_min": 1e-4,  # hypothesis pruning threshold
        "merging_threshold": 4,  # hypothesis merging threshold
        "M": 100,  # maximum number of hypotheses kept in MHT
        "initial_state": [x.initial_state for x in object_motion_fixture.object_configs],
        "n": len(object_motion_fixture.object_configs),
        "P_S": 0.99,
        "birth_model": GaussianMixture(
            [
                WeightedGaussian(log_weight=np.log(0.03), gaussian=Gaussian(x=pos, P=400 * np.eye(4)))
                for pos in [
                    np.array([0, 0, 0, 0]),
                    np.array([400, -600, 0, 0]),
                    np.array([-800, -200, 0, 0]),
                    np.array([-200, 800, 0, 0]),
                ]
            ]
        ),
    }
    yield (GMPHD, tracker_hyperparams)
