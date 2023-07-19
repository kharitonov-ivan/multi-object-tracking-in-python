from dataclasses import asdict

import numpy as np
import pytest
from src.common.gaussian_density import GaussianDensity as GaussianDensity
from src.configs import GroundTruthConfig, SensorModelConfig
from src.measurement_models import (
    ConstantVelocityMeasurementModel,
)
from src.motion_models import ConstantVelocityMotionModel
from src.run import run
from src.scenarios.scenario_configs import linear_sot
from src.simulator import MeasurementsGenerator, ObjectData

from src.trackers.single_object_trackers.nearest_neighbour_tracker import (
    NearestNeighbourTracker,
)


@pytest.mark.parametrize(
    "config, motion_model, meas_model, name, tracker_initial_state",
    [
        (
            linear_sot,
            ConstantVelocityMotionModel,
            ConstantVelocityMeasurementModel,
            "SOT linear case (CV)",
            GaussianDensity(
                means=np.array([-750, -750, 50.0, 50.0]), covs=200.0 * np.eye(4)
            ),
        ),
        # (
        #     nonlinear_sot,
        #     CoordinateTurnMotionModel,
        #     RangeBearingMeasurementModel,
        #     "SOT non linear case (CT)",
        #     Gaussian(
        #         means=np.array([0, 0, 10, 0, np.pi / 180]),
        #         covs=np.power(np.diag([1, 1, 1, 1 * np.pi / 180, 1 * np.pi / 180]), 2),
        #     ),
        # ),
    ],
)
@pytest.mark.parametrize("tracker", [(NearestNeighbourTracker)])  # noqa
def test_tracker(
    config, motion_model, meas_model, name, tracker, tracker_initial_state
):
    config = asdict(config)
    ground_truth = GroundTruthConfig(**config)
    motion_model = motion_model(**config)
    sensor_model = SensorModelConfig(**config)
    meas_model = meas_model(**config)

    object_data = ObjectData(
        ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False
    )
    meas_gen = MeasurementsGenerator(
        object_data=object_data, sensor_model=sensor_model, meas_model=meas_model
    )
    meas_data = [next(meas_gen) for _ in range(ground_truth.total_time)]
    P_G = 0.999  # gating size in percentage

    tracker = tracker(
        meas_model=meas_model,
        sensor_model=sensor_model,
        motion_model=motion_model,
        gating_size=P_G,
        initial_state=tracker_initial_state,
    )
    run(object_data, meas_data, tracker)
