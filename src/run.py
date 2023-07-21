import os
import typing as tp
from multiprocessing import Pool

import colorcet
import imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import motmetrics
import numpy as np

from src.configs import GroundTruthConfig, SensorModelConfig
from src.measurement_models import MeasurementModel
from src.metrics import GOSPA
from src.motion_models import MotionModel
from src.simulator import MeasurementData, ObjectData
from src.utils.plotting import setup_ax
from src.utils.visualizer.common.plot_series import (
    OBJECT_COLORS,
    OBJECT_MARKER,
    plot_estimations,
    plot_measurement_scene,
    plot_object_data,
)


def track(object_data, meas_data, tracker):
    tracker_estimations = []
    for timestep, measurements, sources in meas_data:
        estimations = tracker.step(measurements, 1.0)
        tracker_estimations.append(estimations)
    return tracker_estimations


def prepare_scenario(object_configs, total_time, env_motion_model, env_sensor_model, env_meas_model):
    env_gt = GroundTruthConfig(object_configs, total_time)
    env_object_data = ObjectData(ground_truth_config=env_gt, motion_model=env_motion_model, if_noisy=False)
    meas_data_gen = MeasurementData(object_data=env_object_data, sensor_model=env_sensor_model, meas_model=env_meas_model)
    meas_data = [next(meas_data_gen) for _ in range(total_time)]
    return env_gt, env_object_data, meas_data


def visulaize(object_data, meas_data, tracker_estimations, filepath):
    simulation_steps = len(object_data) if object_data is not None else len(meas_data) if meas_data is not None else len(tracker_estimations)

    fig_side = 5
    fig = plt.figure(figsize=(fig_side * 3, fig_side * 4))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.6, wspace=0.6, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1, 1])
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(4)])

    setup_ax(axs[0, 0], "ground truth")
    plot_object_data(object_data, axs[0, 0])

    setup_ax(axs[0, 1], "observation")

    setup_ax(axs[0, 2], "estimations")
    plot_estimations(axs[0, 2], tracker_estimations)

    setup_ax(axs[1, 0], "gt x pos over time", xlim=(0, simulation_steps), aspect="auto", xlabel="time", ylabel="x position")
    axs[1, 0].set_xticks(np.arange(0, simulation_steps, step=10))

    setup_ax(axs[2, 0], "gt y pos over time", ylim=(0, simulation_steps), aspect="auto", xlabel="y position", ylabel="time")
    axs[2, 0].set_yticks(np.arange(0, simulation_steps, step=10))

    if object_data is not None:
        object_colors = colorcet.glasbey_category10[4:]
        for timestep in range(simulation_steps):
            objects_in_scene = object_data[timestep]
            for object_id in objects_in_scene.keys():
                state = objects_in_scene[object_id]
                gt_pos_x, gt_pos_y = state.x[:2]
                axs[1, 0].scatter(timestep, gt_pos_x, color=object_colors[object_id % 252])
                axs[2, 0].scatter(gt_pos_y, timestep, color=object_colors[object_id % 252])

    setup_ax(axs[1, 1], "observation x pos over time", xlim=(0, simulation_steps), aspect="auto", xlabel="time", ylabel="x position")
    axs[1, 1].set_xticks(np.arange(0, simulation_steps, step=10))

    setup_ax(axs[2, 1], "observation y pos over time", ylim=(0, simulation_steps), aspect="auto", xlabel="y position", ylabel="time")
    axs[2, 1].set_yticks(np.arange(0, simulation_steps, step=10))
    plot_measurement_scene(meas_data, axs[0, 1], axs[1, 1], axs[2, 1])

    setup_ax(axs[1, 2], "estimation x pos over time", xlim=(0, simulation_steps), aspect="auto", xlabel="time", ylabel="x position")
    axs[1, 2].set_xticks(np.arange(0, simulation_steps, step=10))

    setup_ax(axs[2, 2], "estimation y pos over time", ylim=(0, simulation_steps), aspect="auto", xlabel="y position", ylabel="time")
    axs[2, 2].set_yticks(np.arange(0, simulation_steps, step=10))

    if tracker_estimations is not None:
        for timestep in range(len(tracker_estimations)):
            objects_in_scene = tracker_estimations[timestep]
            if objects_in_scene is None:
                continue
            for object_id, state in objects_in_scene.items():
                curr_color, curr_marker = (
                    OBJECT_COLORS[object_id],
                    OBJECT_MARKER,
                )
                est_pos_x, est_pos_y = state.x[:2]
                axs[1, 2].scatter(timestep, est_pos_x, color=curr_color, marker=curr_marker)
                axs[2, 2].scatter(est_pos_y, timestep, color=curr_color, marker=curr_marker)

    params = {}
    if object_data is not None and tracker_estimations is not None:
        gospa = get_gospa(object_data, tracker_estimations)

        setup_ax(axs[3, 0], "GOSPA over time", xlim=(0, simulation_steps), ylim=(0, 200), aspect="auto", xlabel="time", ylabel="GOSPA")
        axs[3, 0].set_xticks(np.arange(0, simulation_steps, step=int(simulation_steps / 10)))
        axs[3, 0].plot(gospa)

        summary = get_motmetrics(object_data, tracker_estimations)

        setup_ax(axs[3, 1], "cardianlity", xlim=(0, simulation_steps), ylim=None, aspect="auto", xlabel="time", ylabel="GOSPA")
        axs[3, 1].set_xticks(np.arange(0, simulation_steps, step=int(simulation_steps / 10)))
        cardianlity_gt = [len(obj) for obj in object_data]
        cardianlity_tracker = [len(track) for track in tracker_estimations]
        max_card = max(max(cardianlity_gt), max(cardianlity_tracker))
        axs[3, 1].set_ylim(0, max_card + 2)
        axs[3, 1].plot(cardianlity_gt, label="gt", marker="o")
        axs[3, 1].plot(cardianlity_tracker, label="tracker", marker="o")
        axs[3, 1].legend()

        params.update(
            {
                "RMS GOSPA": np.sqrt(np.mean(np.power(np.array(gospa), 2))),
                "MOTA": summary["mota"].item(),
                "MOTP": summary["motp"].item(),
                "IDP": summary["idp"].item(),
                "Timestamps": summary["num_frames"].item(),
            }
        )

        axs[3, 2].axis("off")
        for i, (key, value) in enumerate(params.items()):
            axs[3, 2].text(0, 0.5 - i * 0.1, f"{key}: {value:.2f}", fontsize=20)

    fig.suptitle(", ".join(f"{key}: {value:.2f}" for key, value in params.items()), fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(filepath if ".png" in filepath else filepath + ".png")
    plt.close()


def get_motmetrics(object_data, tracker_estimations):
    mot_metric_accumulator = motmetrics.MOTAccumulator()
    for timestep in range(len(tracker_estimations)):
        target_points = np.array([target.x[:2] for target in object_data[timestep].values()])
        target_ids = [target_id for target_id in object_data[timestep].keys()]
        if tracker_estimations[timestep] is not None:
            estimation_ids = [est_id for est_id in tracker_estimations[timestep].keys()]
            estimation_points = np.array([gaussian.x[:2] for gaussian in tracker_estimations[timestep].values()])
        else:
            estimation_points = np.array([])
            estimation_ids = np.array([])
        distance_matrix = motmetrics.distances.norm2squared_matrix(target_points, estimation_points)
        mot_metric_accumulator.update(target_ids, estimation_ids, dists=distance_matrix, frameid=timestep)

    mh = motmetrics.metrics.create()
    summary = mh.compute(mot_metric_accumulator, metrics=["num_frames", "mota", "motp", "idp"], name="acc")
    return summary


def get_gospa(object_data, tracker_estimations):
    gospa = []
    for timestep in range(len(tracker_estimations)):
        target_points = np.array([target.x[:2] for target in object_data[timestep].values()])
        if tracker_estimations[timestep] is not None:
            estimation_points = np.array([gaussian.x[:2] for gaussian in tracker_estimations[timestep].values()])
        else:
            estimation_points = np.array([])
        gospa.append(GOSPA(target_points, estimation_points))
    return gospa


def plot(object_data, meas_data, tracker_estimations, meas_model=None):
    fig_side = 2
    # Создаем новый объект figure
    fig = plt.figure(figsize=(3 * fig_side, fig_side))
    # Создаем первый subplot в figure (1 строка, 2 столбца, первый график)
    ax1 = fig.add_subplot(1, 3, 1)  # 1-я строка, 2 столбца, 1-й график
    setup_ax(ax1, "ground truth")
    plot_object_data(object_data, ax1)

    # Создаем второй subplot в figure (1 строка, 2 столбца, второй график)
    ax2 = fig.add_subplot(1, 3, 2)  # 1-я строка, 2 столбца, 2-й график
    setup_ax(ax2, "observation")
    plot_measurement_scene(meas_data, ax2, None, None)

    ax3 = fig.add_subplot(1, 3, 3)
    setup_ax(ax3, "estimations")
    plot_estimations(ax3, tracker_estimations)
    return fig


def save_figures_to_gif(images, filename):
    if ".gif" not in filename:
        filename += ".gif"
    imageio.mimsave(filename, images)


def render(fig):
    dpi = 60  # новое значение DPI
    width, height = fig.get_size_inches() * fig.dpi  # текущий размер в пикселях
    fig.set_dpi(dpi)  # установить новое значение DPI
    fig.set_size_inches(width / dpi, height / dpi)  # установить новый размер в дюймах

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    dpi = fig.dpi
    width, height = fig.get_size_inches() * dpi

    # Make sure they are integer values
    width = int(width)
    height = int(height)

    image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    plt.close(fig)
    return image


def process_timestep(args):
    object_data, meas_data, tracker_estimations, timestep_idx = args
    fig = plot(
        object_data[0:timestep_idx],
        meas_data[timestep_idx : timestep_idx + 1],
        tracker_estimations[0:timestep_idx] if tracker_estimations is not None else None,
    )

    image = render(fig)
    return image


def animate(object_data, meas_data, tracker_estimations, output_filename):
    with Pool() as pool:
        args = [(object_data, meas_data, tracker_estimations, idx) for idx in range(len(meas_data))]
        images = pool.map(process_timestep, args)
    images = [image for image in images if image is not None]
    save_figures_to_gif(images, output_filename)


def generate_environment(object_configs, total_time, env_motion_model, env_meas_model, env_P_D, env_lambda_c, env_range_c):
    ground_truth = GroundTruthConfig(object_configs, total_time)
    env_sensor_model = SensorModelConfig(env_P_D, env_lambda_c, env_range_c)
    object_data = ObjectData(ground_truth, env_motion_model, if_noisy=False)
    meas_data_gen = MeasurementData(object_data, env_sensor_model, env_meas_model)
    meas_data = [next(meas_data_gen) for _ in range(total_time)]
    return ground_truth, env_sensor_model, object_data, meas_data


def run_tracker(
    object_configs: tp.List,
    total_time: int,
    env_P_D: float,
    env_lambda_c: float,
    env_range_c,
    env_motion_model: MotionModel,
    env_meas_model: MeasurementModel,
    tracker,
    tracker_params: {},
    filepath: str,
    tracker_motion_model: MotionModel = None,
    tracker_meas_model: MeasurementModel = None,
    tracker_sensor_model: SensorModelConfig = None,
    tracker_P_G: float = 0.999,
):
    ground_truth, env_sensor_model, object_data, meas_data = generate_environment(object_configs, total_time, env_motion_model, env_meas_model, env_P_D, env_lambda_c, env_range_c)

    tracker_meas_model = env_meas_model if tracker_meas_model is None else tracker_meas_model
    tracker_motion_model = env_motion_model if tracker_motion_model is None else tracker_motion_model
    tracker_sensor_model = env_sensor_model if tracker_sensor_model is None else tracker_sensor_model

    tracker_estimations = None
    if tracker is not None:
        tracker_model = tracker(meas_model=tracker_meas_model, sensor_model=tracker_sensor_model, motion_model=tracker_motion_model, gating_size=tracker_P_G, **tracker_params)
        tracker_estimations = track(object_data, meas_data, tracker_model)
    visulaize(object_data, meas_data, tracker_estimations, filepath)
    if tracker_estimations is not None:
        gospa = get_gospa(object_data, tracker_estimations)  # noqa
        motmetrics = get_motmetrics(object_data, tracker_estimations)  # noqa F841
    if os.getenv("ANIMATE", "True") == "True":
        animate(object_data, meas_data, tracker_estimations, filepath)
