from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytest
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
    StaticBirthModel, )
from mot.trackers.multiple_object_trackers.PMBM.pmbm import PMBM
from mot.utils import Plotter, get_images_dir
from pytest import fixture
from tqdm import trange

from .params import object_motion_scenarious
from mot.metrics import GOSPA


@pytest.fixture
def object_motion_fixture():
    return object_motion_scenarious.two_objects_linear_motion


def test_pmbm_update_and_predict_linear(object_motion_fixture):
    # Choose object detection probability
    detection_probability = 0.999

    # Choose clutter rate (aka lambda_c)
    clutter_rate = 0.01

    # Choose object survival probability
    survival_probability = 0.9

    # Create sensor model - range/bearing measurement
    range_c = np.array([[-1000, 1000], [-1000, 1000]])
    sensor_model = SensorModelConfig(P_D=0.5,
                                     lambda_c=clutter_rate,
                                     range_c=range_c)

    # Create nlinear motion model
    dt = 1.0
    sigma_q = 10.0
    motion_model = ConstantVelocityMotionModel(dt, sigma_q)

    # Create linear measurement model
    sigma_r = 10.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r)

    # Create ground truth model
    n_births = 1
    simulation_steps = 100

    # Generate true object data (noisy or noiseless) and measurement data
    ground_truth = GroundTruthConfig(n_births,
                                     object_motion_fixture,
                                     total_time=simulation_steps)
    object_data = ObjectData(ground_truth_config=ground_truth,
                             motion_model=motion_model,
                             if_noisy=False)
    meas_data = MeasurementData(object_data=object_data,
                                sensor_model=sensor_model,
                                meas_model=meas_model)

    # Object tracker parameter setting
    gating_percentage = 0.5  # gating size in percentage
    max_hypothesis_kept = 30  # maximum number of hypotheses kept

    z = np.zeros([simulation_steps, 1, 2])
    # linear motion
    z[:, 0, 1] = 0 + 100 * np.arange(simulation_steps)

    # # outlie
    # z[:, 1, 0] = -100 * np.ones(K)
    # z[:, 1, 1] = -100 * np.ones(K)

    birth_model = GaussianMixture([
        WeightedGaussian(
            np.log(0.03),
            Gaussian(x=np.array([0.0, 0.0, 0.0, 0.0]), P=400 * np.eye(4)),
        ),
        WeightedGaussian(
            np.log(0.03),
            Gaussian(x=np.array([300.0, 300.0, 0.0, 0.0]), P=400 * np.eye(4)),
        ),
        WeightedGaussian(
            np.log(0.03),
            Gaussian(x=np.array([300.0, -300.0, 0.0, 0.0]), P=400 * np.eye(4)),
        ),
        WeightedGaussian(
            np.log(0.03),
            Gaussian(x=np.array([-300.0, -300.0, 0.0, 0.0]),
                     P=400 * np.eye(4)),
        ),
        WeightedGaussian(
            np.log(0.03),
            Gaussian(x=np.array([-300.0, 300.0, 0.0, 0.0]), P=400 * np.eye(4)),
        ),
    ])

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
    gospas = []
    simulation_time = simulation_steps

    for timestep in trange(simulation_time):
        current_step_estimates = pmbm.step(meas_data[timestep], dt=1.0)
        targets_vector = np.array(
            [target.x[:2] for target in object_data[timestep].values()])

        estimates_vector = np.array([
            list(estimation.values())[0][:2]
            for estimation in current_step_estimates
        ])
        gospa = GOSPA(targets_vector, estimates_vector)
        gospas.append(gospa)
        estimates.append(current_step_estimates)

    fig, (ax1, ax2, ax0) = plt.subplots(1,
                                        3,
                                        figsize=(6 * 3, 6),
                                        sharey=True,
                                        sharex=True)

    ax0.grid(which="both", linestyle="-", alpha=0.5)
    ax0.set_title(label="ground truth")
    ax0.set_xlabel("x position")
    ax0.set_ylabel("y position")
    ax0 = Plotter.plot_several(
        [object_data],
        ax=ax0,
        out_path=get_images_dir(__file__) + "/" + "object_and_meas_data" +
        ".png",
        is_autoscale=False,
    )

    ax1.grid(which="both", linestyle="-", alpha=0.5)
    ax1.set_title(label="measurements")
    ax1.set_xlabel("x position")
    ax1.set_ylabel("y position")
    ax1 = Plotter.plot_several(
        [meas_data],
        ax=ax1,
        out_path=get_images_dir(__file__) + "/" + "object_and_meas_data" +
        ".png",
        is_autoscale=False,
    )

    ax2.grid(which="both", linestyle="-", alpha=0.5)
    ax2.set_title(label="estimations")
    ax2.set_xlabel("x position")
    ax2.set_ylabel("y position")

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
            ax2.scatter(pos_x, pos_y, color=object_colors[target_id])

    plt.savefig(get_images_dir(__file__) + "/" + "estimation" + ".png")
    from pprint import pprint

    pprint(estimates)
    #TODO check ot
    rms_gospa_scene = np.sqrt(np.mean(np.power(np.array(gospas), 2)))
    assert rms_gospa_scene > 20
