import matplotlib.pyplot as plt
from tqdm import tqdm

from src.common.gaussian_density import GaussianDensity as GaussianDensity
from src.utils.plotting import save_figures_to_gif, setup_ax
from src.utils.visualizer.common.plot_series import (
    plot_estimations,
    plot_measurement_scene,
    plot_object_data,
)


def run(object_data, meas_data, tracker):
    tracker_estimations = []
    for timestep, measurements, sources in meas_data:
        estimations = tracker.step(measurements)
        tracker_estimations.append(estimations)
    return tracker_estimations


def plot(object_data, meas_data, tracker_estimations, meas_model=None):
    # Создаем новый объект figure
    fig = plt.figure(figsize=(18, 6))
    # Создаем первый subplot в figure (1 строка, 2 столбца, первый график)
    ax1 = fig.add_subplot(1, 3, 1)  # 1-я строка, 2 столбца, 1-й график
    setup_ax(ax1, "ground truth")
    plot_object_data(object_data, ax1)

    # Создаем второй subplot в figure (1 строка, 2 столбца, второй график)
    ax2 = fig.add_subplot(1, 3, 2)  # 1-я строка, 2 столбца, 2-й график
    setup_ax(ax2, "observation")
    plot_measurement_scene(ax2, meas_data, meas_model)

    ax3 = fig.add_subplot(1, 3, 3)
    setup_ax(ax3, "estimations")
    plot_estimations(ax3, tracker_estimations)
    plt.tight_layout()  # это помогает предотвратить перекрытие подписей
    return fig


def animate(object_data, meas_data, tracker_estimations, output_filename):
    figs = []
    timestep_start, timestap_end = 0, len(meas_data)
    for timestep_idx in tqdm(range(timestep_start, timestap_end)):
        fig = plot(
            object_data[0:timestep_idx],
            meas_data[timestep_idx : timestep_idx + 1],
            tracker_estimations[0:timestep_idx],
        )
        figs.append(fig)
    save_figures_to_gif(figs, output_filename)
