import logging
from functools import singledispatch

import colorcet
import numpy as np
from matplotlib.lines import Line2D

from src.common.state import Gaussian
from src.simulator import MeasurementData, ObjectData
from src.utils.visualizer.common.plot_primitives import BasicPlotter


CLUTTER_COLOR = colorcet.glasbey_category10[3]  # red
CLUTTER_MARKER = "+"
OBJECT_COLORS = colorcet.glasbey_category10[4:] * 10
OBJECT_MARKER = "s"
OBJECT_MEASUREMENT_MARKER = "o"
OBJECT_MEASUREMENT_COLOR = "b"

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def plot_object_data(series: ObjectData, ax):
    if series is None:
        return
    for timestep in range(len(series)):
        objects_in_scene = series[timestep]
        for object_id in objects_in_scene.keys():
            state = objects_in_scene[object_id]
            curr_color, curr_marker = (
                OBJECT_COLORS[object_id],
                OBJECT_MARKER,
            )
            BasicPlotter.plot_state(state, ax=ax, color=curr_color, center_marker=curr_marker)

    object_ids = []
    for objects in series:
        for object_id in objects.keys():
            object_ids.append(object_id)

    legend_elements, labels = ax.get_legend_handles_labels()
    for object_id in list(set(object_ids)):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=OBJECT_MARKER,
                color=OBJECT_COLORS[object_id],
                label=f"id {object_id}",
            )
        )

    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)  # noqa


def plot_estimations(ax, estimations):
    if estimations is None:
        return
    object_ids = []
    for scene in estimations:
        if not scene:
            continue
        for estimated_object_id, gaussian in scene.items():
            gaussian.plot(ax=ax, color=OBJECT_COLORS[estimated_object_id])
            object_ids.append(estimated_object_id)

    legend_elements, labels = ax.get_legend_handles_labels()
    for object_id in list(set(object_ids)):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=OBJECT_MARKER,
                color=OBJECT_COLORS[object_id],
                label=f"id {object_id}",
            )
        )

    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)  # noqa


def plot_measurement_scene(
    meas_data,
    ax_2d,
    ax_xt=None,
    ax_yt=None,
    clutter_color=CLUTTER_COLOR,
    clutter_marker=CLUTTER_MARKER,
):
    if meas_data is None:
        return
    for timestep, measurements, sources in meas_data:
        object_observations, clutter_observation = (
            measurements[sources != -1],
            measurements[sources == -1],
        )
        colors = [OBJECT_COLORS[int(object_id)] for object_id in sources[sources != -1]]

        ax_2d.scatter(object_observations[..., 0], object_observations[..., 1], color=colors, marker=OBJECT_MEASUREMENT_MARKER)
        ax_2d.scatter(clutter_observation[..., 0], clutter_observation[..., 1], color=clutter_color, marker=clutter_marker)
        if ax_xt is not None:
            if object_observations[..., 0].size > 0:
                ax_xt.scatter([timestep] * len(object_observations[..., 0]), object_observations[..., 0], color=colors, marker=OBJECT_MEASUREMENT_MARKER)
            if clutter_observation[..., 0].size > 0:
                ax_xt.scatter([timestep] * len(clutter_observation[..., 0]), clutter_observation[..., 0], color=clutter_color, marker=clutter_marker)
        if ax_yt is not None:
            if object_observations[..., 1].size > 0:
                ax_yt.scatter(object_observations[..., 1], [timestep] * len(object_observations[..., 1]), color=colors, marker=OBJECT_MEASUREMENT_MARKER)
            if clutter_observation[..., 1].size > 0:
                ax_yt.scatter(clutter_observation[..., 1], [timestep] * len(clutter_observation[..., 1]), color=clutter_color, marker=clutter_marker)

    handles = [
        Line2D([0], [0], marker=clutter_marker, color=clutter_color, label="clutter"),
        Line2D(
            [0],
            [0],
            marker=OBJECT_MEASUREMENT_MARKER,
            color=OBJECT_COLORS[0],
            label="object",
        ),
    ]
    ax_2d.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    if ax_xt is not None:
        ax_xt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    if ax_yt is not None:
        ax_yt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)


@singledispatch
def plot_series(series, ax, *args, **kwargs):
    raise NotImplementedError


@plot_series.register(ObjectData)
def _plot_series(series: ObjectData, ax, *args, **kwargs):
    for timestep in range(len(series)):
        objects_in_scene = series[timestep]
        for object_id in objects_in_scene.keys():
            state = objects_in_scene[object_id]
            curr_color, curr_marker = (
                OBJECT_COLORS[object_id],
                OBJECT_MARKER,
            )
            point, arrow, ellipse = BasicPlotter.plot_state(
                state,
                ax=ax,
                color=curr_color,
                center_marker=curr_marker,
            )
            ax.add_artist(point)

    legend_elements, labels = ax.get_legend_handles_labels()
    legend_elements.extend(
        [
            Line2D(
                [0],
                [0],
                marker=OBJECT_MARKER,
                color=OBJECT_COLORS[object_id],
                label=f"object {object_id}",
            )
            for object_id in range(series._ground_truth_config.n_births)
        ]
    )
    lgd = ax.legend(handles=legend_elements, loc="best", bbox_to_anchor=(1, 0.815))  # noqa
    return ax


@plot_series.register(MeasurementData)
def __plot_series(series: MeasurementData, ax, *args, **kwargs):
    for timestep in range(len(series)):
        plot_measurement_scene(ax, series, timestep)

    legend_elements, labels = ax.get_legend_handles_labels()
    legend_elements.append(Line2D([0], [0], marker=CLUTTER_MARKER, color=CLUTTER_COLOR, label="clutter"))
    lgd = ax.legend(handles=legend_elements, loc="best", bbox_to_anchor=(1, 0.815))  # noqa
    return ax


@plot_series.register(list)
def ___plot_series(series: list, ax, *args, **kwargs):
    for timestep in range(len(series)):
        scene = series[timestep]
        if isinstance(scene, Gaussian):
            for patch in BasicPlotter.plot_state(scene, ax=ax, color="m"):
                ax.add_artist(patch)

        elif isinstance(scene, np.ndarray):
            if len(scene) > 0:
                ax.add_artist(
                    BasicPlotter.plot_point(
                        x=scene.squeeze()[0],
                        y=scene.squeeze()[1],
                        ax=ax,
                        color="r",
                        marker="x",
                    )
                )
        elif isinstance(scene, list):
            if scene:
                for curr_object in scene:
                    if isinstance(curr_object, Gaussian):
                        for patch in BasicPlotter.plot_state(curr_object, ax=ax, color="m"):
                            ax.add_artist(patch)
    return ax


@plot_series.register(np.ndarray)
def ____plot_series(series: np.ndarray, ax, *args, **kwargs):
    for timestep in range(len(series)):
        states = series[timestep]
        for state in states:
            ax.add_artist(BasicPlotter.plot_point(x=state[0], y=state[1], ax=ax, color="m"))
    return ax
