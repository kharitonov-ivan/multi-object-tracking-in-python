import operator
from dataclasses import asdict
from functools import reduce

import pytest
from src.configs.ground_truth_config import GroundTruthConfig
from src.configs.sensor_model_config import SensorModelConfig
from src.measurement_models import ConstantVelocityMeasurementModel
from src.motion_models import ConstantVelocityMotionModel
from src.run import run
from src.scenarios.initial_conditions import linear_n_mot_object_life_params
from src.scenarios.scenario_configs import linear_n_mot
from src.simulator import MeasurementsGenerator
from src.simulator.object_data_generator import ObjectData

from src.trackers.n_object_trackers import GlobalNearestNeighboursTracker


@pytest.mark.parametrize(
    "config, motion_model, meas_model, name, tracker_initial_states",
    [
        # (
        #     linear_sot,
        #     ConstantVelocityMotionModel,
        #     ConstantVelocityMeasurementModel,
        #     "SOT linear case (CV)",
        #     Gaussian(x=np.array([0.0, 0.0, 10.0, 0.0]), P=1.0 * np.eye(4)),
        # ),
        (
            linear_n_mot,
            ConstantVelocityMotionModel,
            ConstantVelocityMeasurementModel,
            "n MOT linear (CV)",
            reduce(
                operator.add, [x.initial_state for x in linear_n_mot_object_life_params]
            ),
        ),
    ],
)
@pytest.mark.parametrize("tracker", [(GlobalNearestNeighboursTracker)])
def test_tracker(
    config, motion_model, meas_model, name, tracker, tracker_initial_states
):
    config = asdict(config)
    ground_truth = GroundTruthConfig(**config)
    motion_model = motion_model(**config)
    sensor_model = SensorModelConfig(**config)
    meas_model = meas_model(**config)

    object_data = ObjectData(
        ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False
    )
    meas_data_gen = MeasurementsGenerator(
        object_data=object_data, sensor_model=sensor_model, meas_model=meas_model
    )

    # Single object tracker parameter setting
    P_G = 0.99  # gating size in percentage
    w_minw = 1e-4  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 100  # maximum number of hypotheses kept in MHT

    tracker = GlobalNearestNeighboursTracker(
        meas_model=meas_model,
        sensor_model=sensor_model,
        motion_model=motion_model,
        P_G=P_G,
        w_min=w_minw,
        merging_threshold=merging_threshold,
        M=M,
        intensity=tracker_initial_states,
    )
    meas_data = [next(meas_data_gen) for _ in range(ground_truth.total_time)]
    run(object_data, meas_data, tracker)

    # tracker_estimations = []
    # meas_data = [next(meas_data_gen) for _ in range(ground_truth.total_time)]
    # for measurement in meas_data:
    #     estimations = tracker.step(measurement[1])
    #     current_estimations = [estimations[idx] for idx in range(len(estimations))]
    #     for idx in range(len(estimations)):
    #         tracker_estimations.append(estimations[idx])

    # # import pdb; pdb.set_trace()
    # Plotter.plot(
    #     [meas_data, object_data, tracker_estimations],
    #     out_path=get_images_dir(__file__) + "/" + name + ".png",
    # )

    # Animator.animate(
    #     [meas_data, object_data, tracker_estimations],
    #     title=name,
    #     filename=get_images_dir(__file__) + "/" + name + ".gif",
    # )
