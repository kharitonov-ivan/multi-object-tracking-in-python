import logging
import pprint
from collections import defaultdict

import matplotlib.pyplot as plt
import mot
import mot.scenarios.object_motion_scenarious as object_motion_scenarious
import motmetrics as mm
import numpy as np
import pytest
from mot.common import Gaussian, GaussianDensity, GaussianMixture, WeightedGaussian
from mot.measurement_models import ConstantVelocityMeasurementModel
from mot.metrics import GOSPA
from mot.motion_models import ConstantVelocityMotionModel
from mot.trackers.multiple_object_trackers.PMBM.common.birth_model import (
    StaticBirthModel,
)
from mot.trackers.multiple_object_trackers.PMBM.pmbm import PMBM
from mot.utils import Plotter, get_images_dir
from mot.utils.timer import Timer
from mot.utils.visualizer.common.plot_series import OBJECT_COLORS as object_colors
from tqdm import trange


@pytest.fixture
def object_motion_fixture():
    return object_motion_scenarious.many_objects_linear_motion_delayed


def test_pmbm_update_and_predict_linear(object_motion_fixture):
    # Choose object detection probability
    detection_probability = 0.99

    # Choose clutter rate (aka lambda_c)
    clutter_rate = 10.0

    # Choose object survival probability
    survival_probability = 0.9

    # Create sensor model - range/bearing measurement
    range_c = np.array([[-1000, 1000], [-1000, 1000]])
    sensor_model = mot.configs.SensorModelConfig(
        P_D=detection_probability, lambda_c=clutter_rate, range_c=range_c
    )

    # Create nlinear motion model
    dt = 1.0
    sigma_q = 10.0
    motion_model = ConstantVelocityMotionModel(dt, sigma_q)

    # Create linear measurement model
    sigma_r = 10.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r)

    # Create ground truth model
    n_births = 10
    simulation_steps = 100

    # Generate true object data (noisy or noiseless) and measurement data
    ground_truth = mot.configs.GroundTruthConfig(
        n_births, object_motion_fixture, total_time=simulation_steps
    )
    object_data = mot.simulator.ObjectData(
        ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False
    )
    meas_data = mot.simulator.MeasurementData(
        object_data=object_data, sensor_model=sensor_model, meas_model=meas_model
    )

    logging.debug(f"object motion config {pprint.pformat(object_motion_fixture)}")
    # Object tracker parameter setting
    gating_percentage = 1.0  # gating size in percentage
    max_hypothesis_kept = 100  # maximum number of hypotheses kept
    existense_probability_threshold = 0.8

    # z = np.zeros([simulation_steps, 1, 2])
    # # linear motion
    # z[:, 0, 1] = 0 + 100 * np.arange(simulation_steps)

    # # outlie
    # z[:, 1, 0] = -100 * np.ones(K)
    # z[:, 1, 1] = -100 * np.ones(K)

    birth_model = GaussianMixture(
        [
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([0.0, 0.0, 0.0, 0.0]), P=100 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([400.0, -600.0, 0.0, 0.0]), P=100 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([-800.0, 200.0, 0.0, 0.0]), P=100 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([-200.0, 800.0, 0.0, 0.0]), P=100 * np.eye(4)),
            ),
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
        existense_probability_threshold=existense_probability_threshold,
        density=GaussianDensity,
    )
    estimates = []
    gospas = []
    simulation_time = simulation_steps
    acc = mm.MOTAccumulator()

    for timestep in range(simulation_time):
        logging.info(f"==============current timestep{timestep}===============")
        with Timer(name="Full cycle of step"):
            current_step_estimates = pmbm.step(meas_data[timestep], dt=1.0)
        targets_vector = np.array(
            [target.x[:2] for target in object_data[timestep].values()]
        )
        if current_step_estimates:
            estimates_vector = np.array(
                [
                    list(estimation.values())[0][:2]
                    for estimation in current_step_estimates
                ]
            )
        else:
            estimates_vector = np.array([])
        gospa = GOSPA(targets_vector, estimates_vector)
        gospas.append(gospa)
        estimates.append(current_step_estimates)

        target_ids = []
        target_points = []
        for target_id, target_state in object_data[timestep].items():
            target_ids.append(target_id)
            target_points.append(target_state.x[:2])

        estimation_ids = []
        estimation_points = []
        if current_step_estimates:
            for estimation in current_step_estimates:
                for estimation_id, estimation_state in estimation.items():
                    estimation_ids.append(estimation_id)
                    estimation_points.append(estimation_state[:2])

        target_points = np.array(target_points)
        estimation_points = np.array(estimation_points)
        distance_matrix = mm.distances.norm2squared_matrix(
            target_points, estimation_points
        )

        acc.update(target_ids, estimation_ids, dists=distance_matrix, frameid=timestep)

    fig, (ax1, ax2, ax0, ax3, ax4) = plt.subplots(
        5, 1, figsize=(6, 6 * 5), sharey=False, sharex=False
    )

    ax0.grid(which="both", linestyle="-", alpha=0.5)
    ax0.set_title(label="ground truth")
    ax0.set_xlabel("x position")
    ax0.set_ylabel("y position")
    ax0 = Plotter.plot_several(
        [object_data],
        ax=ax0,
        out_path=None,
        is_autoscale=False,
    )

    ax1.grid(which="both", linestyle="-", alpha=0.5)
    ax1.set_title(label="measurements")
    ax1.set_xlabel("x position")
    ax1.set_ylabel("y position")
    ax1 = Plotter.plot_several(
        [meas_data],
        ax=ax1,
        out_path=None,
        is_autoscale=False,
    )

    ax2.grid(which="both", linestyle="-", alpha=0.5)
    ax2.set_title(label="estimations")
    ax2.set_xlim([-1100, 1100])
    ax2.set_ylim([-1100, 1100])
    ax2.set_xlabel("x position")
    ax2.set_ylabel("y position")

    ax3.grid(which="both", linestyle="-", alpha=0.5)
    ax3.set_title(label="x position over time")
    ax3.set_xlabel("time")
    ax3.set_ylabel("x position")
    ax3.set_xlim([0, simulation_steps])
    ax3.set_xticks(np.arange(0, simulation_time, step=int(simulation_time / 10)))

    ax4.grid(which="both", linestyle="-", alpha=0.5)
    ax4.set_title(label="y position over time")
    ax4.set_xlabel("time")
    ax4.set_ylabel("y position")
    ax4.set_xlim([0, simulation_steps])
    ax4.set_xticks(np.arange(0, simulation_time, step=int(simulation_time / 10)))

    lines = defaultdict(lambda: [])  # target_id: line
    timelines = defaultdict(lambda: [])
    for timestep, current_timestep_estimations in enumerate(estimates):
        if current_timestep_estimations:
            for estimation in current_timestep_estimations:
                for target_id, state_vector in estimation.items():
                    pos_x, pos_y = state_vector[:2]
                    lines[target_id].append((pos_x, pos_y))
                    ax3.scatter(timestep, pos_x, color=object_colors[target_id % 252])
                    ax4.scatter(timestep, pos_y, color=object_colors[target_id % 252])
                    timelines[target_id].append((timestep, pos_x, pos_y))

    for target_id, estimation_list in timelines.items():
        timesteps = [time for (time, _, _) in estimation_list]
        poses_x = [pos_x for (_, pos_x, _) in estimation_list]
        poses_y = [pos_y for (_, _, pos_y) in estimation_list]
        ax3.plot(timesteps, poses_x, color=object_colors[target_id % 252])
        ax4.plot(timesteps, poses_y, color=object_colors[target_id % 252])

    for target_id, estimation_list in lines.items():
        for (pos_x, pos_y) in estimation_list:
            ax2.scatter(pos_x, pos_y, color=object_colors[target_id % 252])

    # TODO check ot
    rms_gospa_scene = np.sqrt(np.mean(np.power(np.array(gospas), 2)))

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["num_frames", "mota", "motp", "idp"], name="acc")
    fig.suptitle(
        f"RMS GOSPA ={rms_gospa_scene:.1f} "
        f"MOTA ={summary['mota'].item():.1f} "
        f"MOTP ={summary['motp'].item():.1f} "
        f"IDP ={summary['idp'].item():.1f} ",
        fontweight="bold",
    )
    plt.savefig(get_images_dir(__file__) + "/" + "estimation" + ".png")
