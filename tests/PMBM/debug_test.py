from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from mot import (
    GaussianMixture,
    Object,
    Gaussian,
    SensorModelConfig,
    ConstantVelocityMotionModel,
    ConstantVelocityMeasurementModel,
    GroundTruthConfig,
    ObjectData,
    MeasurementData,
    WeightedGaussian,
    GaussianDensity,
)
from mot.trackers.multiple_object_trackers.PMBM.common.birth_model import (
    StaticBirthModel,
)
from mot.trackers.multiple_object_trackers.PMBM.pmbm import PMBM
from mot.utils import Plotter, get_images_dir
from pytest import fixture
from tqdm import trange


@fixture(scope="function")
def linear_middle_params():
    return [
        Object(
            initial=Gaussian(x=np.array([0.0, 0.0, 0.0, 50.0]), P=400 * np.eye(4)),
            t_birth=0,
            t_death=69,
        ),
        Object(
            initial=Gaussian(x=np.array([0.0, 0.0, 100.0, 0.0]), P=400 * np.eye(4)),
            t_birth=5,
            t_death=69,
        ),
        # Object(
        #     initial=Gaussian(x=np.array([0.0, -10.0, -3.0, 0.0]), P=10 * np.eye(4)),
        #     t_birth=0,
        #     t_death=100,
        # ),
    ]


# @fixture(scope="function")
# def birth_model():
#     return GaussianMixture(
#         [
#             WeightedGaussian(
#                 np.log(0.1),
#                 Gaussian(x=np.array([0.0, 0.0, 0.0, 0.0]), P=400 * np.eye(4)),
#             ),
#             WeightedGaussian(
#                 np.log(0.1),
#                 Gaussian(x=np.array([0.0, -200.0, 0.0, 0.0]), P=400 * np.eye(4)),
#             ),
#             WeightedGaussian(
#                 np.log(0.1),
#                 Gaussian(x=np.array([0.0, 200.0, 0.0, 0.0]), P=400 * np.eye(4)),
#             ),
#         ]
#     )


def test_pmbm_update_and_predict_linear(linear_middle_params):
    # Choose object detection probability
    detection_probability = 0.9

    # Choose clutter rate (aka lambda_c)
    clutter_rate = 0.1

    # Choose object survival probability
    survival_probability = 0.9

    # Create sensor model - range/bearing measurement
    range_c = np.array([[-1000, 1000], [-1000, 1000]])
    sensor_model = SensorModelConfig(1.0, 0.0, range_c)

    # Create nlinear motion model
    dt = 1.0
    sigma_q = 0.01
    motion_model = ConstantVelocityMotionModel(dt, sigma_q)

    # Create linear measurement model
    sigma_r = 0.01
    meas_model = ConstantVelocityMeasurementModel(sigma_r)

    # Create ground truth model
    n_births = 2
    simulation_steps = 15

    # Generate true object data (noisy or noiseless) and measurement data
    ground_truth = GroundTruthConfig(
        n_births, linear_middle_params, total_time=simulation_steps
    )
    object_data = ObjectData(
        ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False
    )
    meas_data = MeasurementData(
        object_data=object_data, sensor_model=sensor_model, meas_model=meas_model
    )

    # Object tracker parameter setting
    gating_percentage = 1.0  # gating size in percentage
    max_hypothesis_kept = 100  # maximum number of hypotheses kept

    z = np.zeros([simulation_steps, 1, 2])
    # linear motion
    z[:, 0, 1] = 0 + 100 * np.arange(simulation_steps)

    # # outlie
    # z[:, 1, 0] = -100 * np.ones(K)
    # z[:, 1, 1] = -100 * np.ones(K)

    birth_model = GaussianMixture(
        [
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([0.0, 0.0, 0.0, 0.0]), P=10000 * np.eye(4)),
            ),
            # WeightedGaussian(
            #     np.log(0.03),
            #     Gaussian(x=np.array([100.0, 100.0, 0.0, 0.0]), P=100 * np.eye(4)),
            # ),
            # WeightedGaussian(
            #     np.log(0.03),
            #     Gaussian(x=np.array([100.0, -100.0, 0.0, 0.0]), P=100 * np.eye(4)),
            # ),
            # WeightedGaussian(
            #     np.log(0.03),
            #     Gaussian(x=np.array([-100.0, -100.0, 0.0, 0.0]), P=100 * np.eye(4)),
            # ),
        ]
    )

    pmbm = PMBM(
        meas_model=meas_model,
        sensor_model=sensor_model,
        motion_model=motion_model,
        birth_model=StaticBirthModel(birth_model),
        max_number_of_hypotheses=max_hypothesis_kept,
        gating_percentage=gating_percentage,
        detection_probability=detection_probability,
        survival_probability=survival_probability,
        density=GaussianDensity,
    )
    estimates = []
    simulation_time = simulation_steps

    for timestep in trange(simulation_time):
        current_step_estimates = pmbm.step(meas_data[timestep], dt=1.0)
        estimates.append(current_step_estimates)

    Plotter.plot_several(
        [meas_data],
        out_path=get_images_dir(__file__) + "/" + "object_and_meas_data" + ".png",
    )

    fig = plt.figure()
    ax = plt.subplot(111, aspect="equal")
    ax.grid(which="both", linestyle="-", alpha=0.5)
    ax.set_title(label="estimations")
    ax.set_xlabel("x position")
    ax.set_ylabel("y p{osition")
    ax.set_xlim((-1000, 1000))
    ax.set_ylim((-1000, 1000))

    lines = defaultdict(lambda: [])  # target_id: line
    for current_timestep_estimations in estimates:
        if current_timestep_estimations:
            for estimation in current_timestep_estimations:
                for target_id, state_vector in estimation.items():
                    pos_x, pos_y = state_vector[:2]
                    lines[target_id].append((pos_x, pos_y))

    from mot.utils.visualizer.common.plot_series import OBJECT_COLORS as object_colors

    for target_id, estimation_list in lines.items():
        for (pos_x, pos_y) in estimation_list:
            plt.scatter(pos_x, pos_y, color=object_colors[target_id])

    plt.savefig(get_images_dir(__file__) + "/" + "estimation" + ".png")
    from pprint import pprint

    pprint(estimates)
