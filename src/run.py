import matplotlib.pyplot as plt
from tqdm import tqdm

from src.common.gaussian_density import GaussianDensity as GaussianDensity
from src.utils.plotting import save_figures_to_gif, setup_ax
from src.utils.visualizer.common.plot_series import (
    plot_estimations,
    plot_measurement_scene,
    plot_object_data,
)


def plot_sample(object_data, meas_data, tracker_estimations):
    # Создаем новый объект figure
    fig = plt.figure(figsize=(12, 6))
    # Создаем первый subplot в figure (1 строка, 2 столбца, первый график)
    ax1 = fig.add_subplot(1, 2, 1)  # 1-я строка, 2 столбца, 1-й график
    setup_ax(ax1, "ground truth")
    plot_object_data(object_data, ax1)

    # Создаем второй subplot в figure (1 строка, 2 столбца, второй график)
    ax2 = fig.add_subplot(1, 2, 2)  # 1-я строка, 2 столбца, 2-й график
    setup_ax(ax2, "observation")

    plot_measurement_scene(ax2, meas_data)
    plot_estimations(ax2, tracker_estimations)
    plt.tight_layout()  # это помогает предотвратить перекрытие подписей
    return fig


def run(object_data, meas_data, tracker):
    tracker_estimations = []
    for timestep, measurements, sources in meas_data:
        estimations = tracker.step(measurements)
        tracker_estimations.append(estimations)
    timestep = -1
    plot_sample(
        object_data[:timestep], meas_data[:timestep], tracker_estimations[:timestep]
    )

    plt.savefig("test.png")

    figs = []
    timestep_start, timestap_end = 0, len(meas_data)
    for timestep_idx in tqdm(range(timestep_start, timestap_end)):
        import pdb

        pdb.set_trace()
        fig = plot_sample(
            object_data[0:timestep_idx],
            meas_data[timestep_idx : timestep_idx + 1],
            tracker_estimations[0:timestep_idx],
        )

        figs.append(fig)

    save_figures_to_gif(figs, "test.gif")
