from cProfile import label
import logging
from functools import singledispatch

import colorcet
from matplotlib import legend
import numpy as np
from matplotlib.lines import Line2D
from src.common.gaussian_density import GaussianDensity as GaussianDensity
from src.simulator import MeasurementsGenerator, ObjectData

from src.utils.visualizer.common.plot_primitives import BasicPlotter


CLUTTER_COLOR = colorcet.glasbey_category10[3]  # red
CLUTTER_MARKER = "+"
OBJECT_COLORS = colorcet.glasbey_category10[4:]
OBJECT_MARKER = "s"
OBJECT_MEASUREMENT_MARKER = "o"
OBJECT_MEASUREMENT_COLOR = "b"

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt


def clean_legend_labels(ax: plt.Axes):
    """
    This function takes a matplotlib axes object and cleans up the legend labels by removing duplicates.

    Parameters:
    ax (plt.Axes): The axes object containing the plot and legend.

    Returns:
    None
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())


def plot_measurement_scene(
    ax,
    meas_data,
    clutter_color=CLUTTER_COLOR,
    clutter_marker=CLUTTER_MARKER,
):
    for timestep, measurements, sources in meas_data:
        object_observations, clutter_observation = (
            measurements[sources != -1],
            measurements[sources == -1],
        )
        colors = [OBJECT_COLORS[int(object_id)] for object_id in sources[sources != -1]]

        ax.scatter(
            object_observations[..., 0],
            object_observations[..., 1],
            color=colors,
            marker=OBJECT_MEASUREMENT_MARKER,
        )
        ax.scatter(
            clutter_observation[..., 0],
            clutter_observation[..., 1],
            color=clutter_color,
            marker=clutter_marker,
        )

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
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)


def plot_estimations(ax, estimations):
    for scene in estimations:
        for estimated_object_id, gaussian in scene.items():
            gaussian.plot(ax=ax, color=OBJECT_COLORS[estimated_object_id])


@singledispatch
def plot_series(series, ax, *args, **kwargs):
    raise NotImplementedError


def plot_object_data(series: ObjectData, ax):
    for timestep in range(len(series)):
        objects_in_scene = series[timestep]
        for object_id in objects_in_scene.keys():
            state = objects_in_scene[object_id]
            curr_color, curr_marker = (
                OBJECT_COLORS[object_id],
                OBJECT_MARKER,
            )
            BasicPlotter.plot_state(
                state, ax=ax, color=curr_color, center_marker=curr_marker
            )

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

    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3
    )  # noqa


@plot_series.register(MeasurementsGenerator)
def __plot_series(series: MeasurementsGenerator, ax, *args, **kwargs):
    for timestep in range(len(series)):
        plot_measurement_scene(ax, series, timestep)

    legend_elements, labels = ax.get_legend_handles_labels()
    legend_elements.append(
        Line2D([0], [0], marker=CLUTTER_MARKER, color=CLUTTER_COLOR, label="clutter")
    )
    lgd = ax.legend(
        handles=legend_elements, loc="best", bbox_to_anchor=(1, 0.815)
    )  # noqa
    return ax


@plot_series.register(list)
def ___plot_series(series: list, ax, *args, **kwargs):
    for timestep in range(len(series)):
        scene = series[timestep]
        if isinstance(scene, GaussianDensity):
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
                    if isinstance(curr_object, GaussianDensity):
                        for patch in BasicPlotter.plot_state(
                            curr_object, ax=ax, color="m"
                        ):
                            ax.add_artist(patch)
    return ax


@plot_series.register(np.ndarray)
def ____plot_series(series: np.ndarray, ax, *args, **kwargs):
    for timestep in range(len(series)):
        states = series[timestep]
        for state in states:
            ax.add_artist(
                BasicPlotter.plot_point(x=state[0], y=state[1], ax=ax, color="m")
            )
    return ax
