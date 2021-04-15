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
    StaticBirthModel, )
from mot.trackers.multiple_object_trackers.PMBM.pmbm import PMBM
from mot.utils import Plotter, get_images_dir
from mot.utils.timer import Timer
from mot.utils.visualizer.common.plot_series import OBJECT_COLORS as object_colors
from tqdm import trange
from .params.birth_model import birth_model_params


@pytest.fixture(params=[
    pytest.param(0.8, marks=pytest.mark.skip), 0.99,
    pytest.param(0.5, marks=pytest.mark.skip)
])
def scenario_detection_probability(request):
    yield request.param


@pytest.fixture(params=[
    pytest.param(0.5, marks=pytest.mark.skip), 10.0,
    pytest.param(5.0, marks=pytest.mark.skip)
])
def scenario_clutter_rate(request):
    yield request.param


@pytest.fixture(params=[
    pytest.param(object_motion_scenarious.single_static_object,
                 marks=pytest.mark.skip),
    pytest.param(object_motion_scenarious.two_static_objects,
                 marks=pytest.mark.skip),
    pytest.param(object_motion_scenarious.tree_static_objects,
                 marks=pytest.mark.skip),
    pytest.param(object_motion_scenarious.single_object_linear_motion,
                 marks=pytest.mark.skip),
    pytest.param(object_motion_scenarious.two_objects_linear_motion,
                 marks=pytest.mark.skip),
    pytest.param(object_motion_scenarious.two_objects_linear_motion_delayed,
                 marks=pytest.mark.skip),
    object_motion_scenarious.many_objects_linear_motion_delayed,
])
def object_motion_fixture(request):
    yield request.param


@pytest.fixture(params=[birth_model_params])
def birth_model(request):
    yield request.param


def test_synthetic_scenario(
    object_motion_fixture,
    scenario_detection_probability,
    scenario_clutter_rate,
    birth_model,
):
    # Choose object detection probability
    detection_probability = scenario_detection_probability

    # Choose clutter rate (aka lambda_c)
    clutter_rate = scenario_clutter_rate

    # Choose object survival probability
    survival_probability = 0.9
    # Create nlinear motion model
    dt = 1.0
    sigma_q = 10.0
    motion_model = ConstantVelocityMotionModel(dt, sigma_q)

    # Create linear measurement model
    sigma_r = 10.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r)

    # Create ground truth model
    simulation_steps = 100

    # Create sensor model - range/bearing measurement
    range_c = np.array([[-1000, 1000], [-1000, 1000]])
    sensor_model = mot.configs.SensorModelConfig(P_D=detection_probability,
                                                 lambda_c=clutter_rate,
                                                 range_c=range_c)

    # Generate true object data (noisy or noiseless) and measurement data
    ground_truth = mot.configs.GroundTruthConfig(object_motion_fixture,
                                                 total_time=simulation_steps)
    object_data = mot.simulator.ObjectData(ground_truth_config=ground_truth,
                                           motion_model=motion_model,
                                           if_noisy=False)
    meas_data = mot.simulator.MeasurementData(object_data=object_data,
                                              sensor_model=sensor_model,
                                              meas_model=meas_model)

    logging.debug(
        f"object motion config {pprint.pformat(object_motion_fixture)}")

    # Object tracker parameter setting
    gating_percentage = 1.0  # gating size in percentage
    max_hypothesis_kept = 100  # maximum number of hypotheses kept
    existense_probability_threshold = 0.8

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

    simulation_time = simulation_steps

    estimates = []
    gospas = []
    motmetrics_accumulator = mm.MOTAccumulator()

    for timestep in trange(simulation_time):
        logging.debug(f"===========current timestep{timestep}============")
        with Timer(name="One iterations takes time:"):
            current_step_estimates = pmbm.step(meas_data[timestep], dt=1.0)

        estimates.append(current_step_estimates)

        target_points = np.array(
            [target.x[:2] for target in object_data[timestep].values()])
        target_ids = [target_id for target_id in object_data[timestep].keys()]

        if current_step_estimates:
            estimation_points = np.array([
                list(estimation.values())[0][:2]
                for estimation in current_step_estimates
            ])

            estimation_ids = [
                estimation_ids for estimation in current_step_estimates
                for estimation_ids in estimation.keys()
            ]

        else:
            estimation_points = np.array([])
            estimation_ids = []

        gospas.append(GOSPA(target_points, estimation_points))

        distance_matrix = mm.distances.norm2squared_matrix(
            target_points, estimation_points)

        motmetrics_accumulator.update(target_ids,
                                      estimation_ids,
                                      dists=distance_matrix,
                                      frameid=timestep)

    fig, (ax1, ax2, ax0, ax3, ax4) = plt.subplots(5,
                                                  1,
                                                  figsize=(8, 8 * 5),
                                                  sharey=False,
                                                  sharex=False)

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
    ax3.set_xticks(
        np.arange(0, simulation_time, step=int(simulation_time / 10)))

    ax4.grid(which="both", linestyle="-", alpha=0.5)
    ax4.set_title(label="y position over time")
    ax4.set_xlabel("time")
    ax4.set_ylabel("y position")
    ax4.set_xlim([0, simulation_steps])
    ax4.set_xticks(
        np.arange(0, simulation_time, step=int(simulation_time / 10)))

    lines = defaultdict(lambda: [])  # target_id: line
    timelines = defaultdict(lambda: [])
    for timestep, current_timestep_estimations in enumerate(estimates):
        if current_timestep_estimations:
            for estimation in current_timestep_estimations:
                for target_id, state_vector in estimation.items():
                    pos_x, pos_y = state_vector[:2]
                    lines[target_id].append((pos_x, pos_y))
                    ax3.scatter(timestep,
                                pos_x,
                                color=object_colors[target_id % 252])
                    ax4.scatter(timestep,
                                pos_y,
                                color=object_colors[target_id % 252])
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
    summary = mh.compute(motmetrics_accumulator,
                         metrics=["num_frames", "mota", "motp", "idp"],
                         name="acc")
    fig.suptitle(
        f"RMS GOSPA ={rms_gospa_scene:.1f} "
        f"MOTA ={summary['mota'].item():.1f} "
        f"MOTP ={summary['motp'].item():.1f} "
        f"IDP ={summary['idp'].item():.1f} ",
        fontweight="bold",
    )
    meta = f"P_D={scenario_detection_probability},_lambda_c={scenario_clutter_rate}"
    plt.savefig(get_images_dir(__file__) + "/" + "results_" + meta + ".png")
