import numpy as np
from mot.common.state import Gaussian, WeightedGaussian, GaussianMixture
from mot.motion_models import CoordinateTurnMotionModel, ConstantVelocityMotionModel
from mot.trackers.multiple_object_trackers.PMBM.common import (
    Bernoulli,
    MultiBernouilliMixture,
    Track,
    SingleTargetHypothesis,
)
from mot.trackers.multiple_object_trackers.PMBM.pmbm import PMBM
from collections import defaultdict
from mot.measurement_models import (
    ConstantVelocityMeasurementModel,
)
from mot.trackers.multiple_object_trackers.PMBM.pmbm import PoissonRFS
from mot.configs import GroundTruthConfig, Object, SensorModelConfig
from mot.simulator.object_data_generator import ObjectData
from mot.simulator.measurement_data_generator import MeasurementData
from pytest import fixture
from scipy.stats import chi2
from tqdm import trange
from mot.utils import Plotter, get_images_dir
import matplotlib.pyplot as plt
from collections import defaultdict
import logging


@fixture(scope="function")
def linear_middle_params():
    return [
        Object(
            initial=Gaussian(x=np.array([0.0, 0.0, 0.0, 5.0]), P=np.eye(4)),
            t_birth=0,
            t_death=69,
        ),
        # Object(
        #     initial=Gaussian(x=np.array([-800.0, -200.0, 20.0, -5.0]), P=np.eye(4)),
        #     t_birth=5,
        #     t_death=99,
        # )
    ]


@fixture(scope="function")
def linear_big_params():
    return [
        Object(
            initial=Gaussian(x=np.array([0.0, 0.0, 0.0, -10.0]), P=np.eye(4)),
            t_birth=0,
            t_death=69,
        ),
        Object(
            initial=Gaussian(x=np.array([400.0, -600.0, -10.0, 5.0]), P=np.eye(4)),
            t_birth=0,
            t_death=99,
        ),
        Object(
            initial=Gaussian(x=np.array([-800.0, -200.0, 20.0, -5.0]), P=np.eye(4)),
            t_birth=0,
            t_death=69,
        ),
        Object(
            initial=Gaussian(x=np.array([400.0, -600.0, -7.0, -4.0]), P=np.eye(4)),
            t_birth=19,
            t_death=99,
        ),
        Object(
            initial=Gaussian(x=np.array([400.0, -600.0, -2.5, 10.0]), P=np.eye(4)),
            t_birth=19,
            t_death=99,
        ),
        Object(
            initial=Gaussian(x=np.array([0.0, 0.0, 7.5, -5.0]), P=np.eye(4)),
            t_birth=19,
            t_death=99,
        ),
        Object(
            initial=Gaussian(x=np.array([-800.0, -200.0, 12.0, 7.0]), P=np.eye(4)),
            t_birth=39,
            t_death=99,
        ),
        Object(
            initial=Gaussian(x=np.array([-200.0, 800.0, 15.0, -10.0]), P=np.eye(4)),
            t_birth=39,
            t_death=99,
        ),
        Object(
            initial=Gaussian(x=np.array([-800.0, -200.0, 3.0, 15.0]), P=np.eye(4)),
            t_birth=59,
            t_death=99,
        ),
        Object(
            initial=Gaussian(x=np.array([-200.0, 800.0, -3.0, -15.0]), P=np.eye(4)),
            t_birth=59,
            t_death=99,
        ),
        Object(
            initial=Gaussian(x=np.array([0.0, 0.0, -20.0, -15.0]), P=np.eye(4)),
            t_birth=79,
            t_death=99,
        ),
        Object(
            initial=Gaussian(x=np.array([-200.0, 800.0, 15.0, -5.0]), P=np.eye(4)),
            t_birth=79,
            t_death=99,
        ),
    ]


@fixture(scope="function")
def linear_small_params():
    return [
        Object(
            initial=Gaussian(x=np.array([0.0, 0.0, 0.0, 20.0]), P=np.eye(4)),
            t_birth=0,
            t_death=69,
        ),
        Object(
            initial=Gaussian(x=np.array([400.0, -600.0, -40.0, 45.0]), P=np.eye(4)),
            t_birth=0,
            t_death=99,
        ),
    ]


@fixture(scope="function")
def birth_model():
    return GaussianMixture(
        [
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([0.0, 0.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([400.0, -600.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([-800.0, -200.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([-200.0, 800.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
        ]
    )


def test_PMBM_predict():
    # Create nonlinear motion model (coordinate turn)
    dt = 1.0
    sigma_V = 1.0
    sigma_Omega = np.pi / 180
    motion_model = CoordinateTurnMotionModel(dt, sigma_V, sigma_Omega)

    # Set birth model
    birth_model = GaussianMixture(
        [
            WeightedGaussian(
                log_weight=np.log(0.03),
                gaussian=Gaussian(
                    x=np.array([0.4456, 0.6463, 0.7094, 0.7547, 0.2760]),
                    P=np.diag([1.0, 1.0, 1.0, (np.pi / 90) ** 2, (np.pi / 90) ** 2]),
                ),
            )
        ]
        * 4
    )

    # Set probability of existence
    survival_probability = 0.7

    # Set Poisson RFS
    PPP = PoissonRFS(
        GaussianMixture(
            weighted_components=[
                WeightedGaussian(
                    -0.3861,
                    Gaussian(
                        x=np.array([0.0, 0.0, 5.0, 0.0, np.pi / 180]), P=np.eye(5)
                    ),
                ),
                WeightedGaussian(
                    -0.423,
                    Gaussian(
                        x=np.array([20.0, 20.0, -20.0, 0.0, np.pi / 90]), P=np.eye(5)
                    ),
                ),
                WeightedGaussian(
                    -1.8164,
                    Gaussian(
                        x=np.array([-20.0, 10.0, -10.0, 0.0, np.pi / 360]), P=np.eye(5)
                    ),
                ),
            ]
        )
    )

    # Set Bernoulli RFS
    bern_first = Bernoulli(
        r=0.1112,
        initial_state=Gaussian(
            x=np.array(
                [
                    0.78,
                    0.39,
                    0.24,
                    0.40,
                    0.09,
                ]
            ),
            P=np.eye(5),
        ),
    )
    bern_second = Bernoulli(
        r=0.1319,
        initial_state=Gaussian(
            x=np.array(
                [
                    0.94,
                    0.95,
                    0.57,
                    0.06,
                    0.23,
                ]
            ),
            P=np.eye(5),
        ),
    )

    track_first = Track(SingleTargetHypothesis(bern_first, 0, 0, 0))
    track_second = Track(SingleTargetHypothesis(bern_second, 0, 0, 0))

    MBM = MultiBernouilliMixture()
    MBM.tracks.update({0: track_first, 1: track_second})

    pmbm_tracker = PMBM(
        meas_model=None,
        sensor_model=None,
        motion_model=motion_model,
        birth_model=birth_model,
        max_number_of_hypotheses=None,
        P_D=0.9,
        P_S=0.9,
        gating_size=None,
    )
    pmbm_tracker.PPP = PPP
    pmbm_tracker.MBM = MBM

    pmbm_tracker.predict(birth_model, motion_model, survival_probability, dt)

    bern_first_ref = Bernoulli(
        r=0.0778,
        initial_state=Gaussian(
            x=np.array([1.002, 0.485, 0.242, 0.500, 0.096]),
            P=np.array(
                [
                    [1.85, 0.34, 0.92, -0.095, 0.0],
                    [0.34, 1.20, 0.39, 0.22, 0.0],
                    [0.92, 0.39, 2.0, 0.0, 0.0],
                    [-0.095, 0.22, 0.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                ]
            ),
        ),
    )

    bern_second_ref = Bernoulli(
        r=0.092,
        initial_state=Gaussian(
            x=np.array([1.51, 0.99, 0.57, 0.29, 0.23]),
            P=np.array(
                [
                    [1.997, 0.039, 0.998, -0.034, 0.0],
                    [0.039, 1.33, 0.059, 0.574, 0.0],
                    [0.998, 0.059, 2.0, 0.0, 0.0],
                    [-0.034, 0.574, 0.0, 2.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.000],
                ]
            ),
        ),
    )

    track_first_ref = Track(SingleTargetHypothesis(bern_first_ref, 0, 0, 0))
    track_first_ref.track_id = 0
    track_second_ref = Track(SingleTargetHypothesis(bern_second_ref, 0, 0, 0))
    track_second_ref.track_id = 1

    MBM_ref = MultiBernouilliMixture()
    MBM_ref.tracks.append(track_first_ref)
    MBM_ref.tracks.append(track_second_ref)
    for i in range(1):
        np.testing.assert_allclose(
            pmbm_tracker.MBM.tracks[i].hypotheses[0].bernoulli.state.x,
            MBM_ref.tracks[i].hypotheses[0].bernoulli.state.x,
            rtol=0.1,
        )
        np.testing.assert_allclose(
            pmbm_tracker.MBM.tracks[i].hypotheses[0].bernoulli.state.P,
            MBM_ref.tracks[i].hypotheses[0].bernoulli.state.P,
            rtol=0.1,
        )

    ppp_w_ref = np.array(
        [-3.5066, -3.5066, -3.5066, -3.5066, -2.1731, -0.7797, -0.7428]
    )
    ppp_w_got = np.array(
        sorted([component.w for component in pmbm_tracker.PPP.intensity])
    )
    np.testing.assert_allclose(ppp_w_ref, ppp_w_got, rtol=0.01)

    PPP_ref = PoissonRFS(
        initial_intensity=GaussianMixture(
            weighted_components=[
                WeightedGaussian(
                    w=-0.7428,
                    gm=Gaussian(
                        x=np.array([5.0, 0.0, 5.0, 0.0175, 0.0175]),
                        P=np.array(
                            [
                                [2.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 26.0, 0.0, 5.0, 0.0],
                                [1.0, 0.0, 2.0, 0.0, 0.0],
                                [0.0, 5.0, 0.0, 2.0, 1.0],
                                [0.0, 0.0, 0.0, 1.0, 1.00030462],
                            ]
                        ),
                    ),
                ),
                WeightedGaussian(
                    w=-0.7797,
                    gm=Gaussian(
                        x=np.array([0.0, 20.0, -20.0, 0.0349, 0.0349]),
                        P=np.array(
                            [
                                [2.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 401.0, 0.0, -20.0, 0.0],
                                [1.0, 0.0, 2.0, 0.0, 0.0],
                                [0.0, -20.0, 0.0, 2.0, 1.0],
                                [0.0, 0.0, 0.0, 1.0, 1.0003],
                            ]
                        ),
                    ),
                ),
                WeightedGaussian(
                    w=-2.1731,
                    gm=Gaussian(
                        x=np.array([-30.0, 10.0, -10.0, 0.0087, 0.0087]),
                        P=np.array(
                            [
                                [2.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 101.0, 0.0, -10.0, 0.0],
                                [1.0, 0.0, 2.0, 0.0, 0.0],
                                [0.0, -10.0, 0.0, 2.0, 1.0],
                                [0.0, 0.0, 0.0, 1.0, 1.0003],
                            ]
                        ),
                    ),
                ),
                WeightedGaussian(
                    w=-3.5066,
                    gm=Gaussian(
                        x=np.array([0.4456, 0.6463, 0.7094, 0.7547, 0.2760]),
                        P=np.diag([1.0, 1.0, 1.0, 0.0012, 0.0012]),
                    ),
                ),
                WeightedGaussian(
                    w=-3.5066,
                    gm=Gaussian(
                        x=np.array([0.4456, 0.6463, 0.7094, 0.7547, 0.2760]),
                        P=np.diag([1.0, 1.0, 1.0, 0.0012, 0.0012]),
                    ),
                ),
                WeightedGaussian(
                    w=-3.5066,
                    gm=Gaussian(
                        x=np.array([0.4456, 0.6463, 0.7094, 0.7547, 0.2760]),
                        P=np.diag([1.0, 1.0, 1.0, 0.0012, 0.0012]),
                    ),
                ),
                WeightedGaussian(
                    w=-3.5066,
                    gm=Gaussian(
                        x=np.array([0.4456, 0.6463, 0.7094, 0.7547, 0.2760]),
                        P=np.diag([1.0, 1.0, 1.0, 0.0012, 0.0012]),
                    ),
                ),
            ]
        )
    )

    for first_state, second_state in zip(pmbm_tracker.PPP.intensity, PPP_ref.intensity):
        np.testing.assert_allclose(first_state.gm.x, second_state.gm.x, rtol=0.05)
        np.testing.assert_allclose(first_state.gm.P, second_state.gm.P, rtol=0.05)
        np.testing.assert_allclose(first_state.w, second_state.w, rtol=0.05)


def test_pmbm_update_and_predict_linear(linear_big_params, birth_model):
    # Choose object detection probability
    P_D = 0.98

    # Choose clutter rate
    lambda_c = 0.01

    # Choose object survival probability
    survival_probability = 0.9999

    # Create sensor model - range/bearing measurement
    range_c = np.array([[-1000, 1000], [-1000, 1000]])
    sensor_model = SensorModelConfig(P_D, lambda_c, range_c)

    # Create nlinear motion model
    dt = 1.0
    sigma_q = 5.0
    motion_model = ConstantVelocityMotionModel(dt, sigma_q)

    # Create linear measurement model
    sigma_r = 10.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r)

    # Create ground truth model
    n_births = 12
    K = 20

    # Generate true object data (noisy or noiseless) and measurement data
    ground_truth = GroundTruthConfig(n_births, linear_big_params, total_time=K)
    object_data = ObjectData(
        ground_truth_config=ground_truth, motion_model=motion_model, if_noisy=False
    )
    meas_data = MeasurementData(
        object_data=object_data, sensor_model=sensor_model, meas_model=meas_model
    )

    # Object tracker parameter setting
    P_G = 0.999  # gating size in percentage
    w_min = 1e-3  # hypothesis pruning threshold
    merging_threshold = 2  # hypothesis merging threshold
    M = 20  # maximum number of hypotheses kept
    r_min = 1e-3  # Bernoulli component pruning threshold
    r_recycle = 0.1  # Bernoulli component recycling threshold
    r_estimate = 0.4  # Threshold used to extract estimates from Bernoullis

    pmbm = PMBM(
        meas_model=meas_model,
        sensor_model=sensor_model,
        motion_model=motion_model,
        birth_model=birth_model,
        merging_threshold=merging_threshold,
        max_number_of_hypotheses=M,
        gating_percentage=P_G
        w_min=w_min,
        P_D=P_D,
        survival_probability=survival_probability,
    )
    estimates = []
    simulation_time = K
    from pprint import pprint

    for timestep in trange(simulation_time):
        import logging

        logging.debug(pmbm.__repr__())
        pmbm.increment_timestep()
        if len(meas_data[timestep]) > 0:
            pmbm.update(z=meas_data[timestep])

        current_step_estimates = pmbm.estimator()
        pmbm.reduction()
        estimates.append(current_step_estimates)
        pmbm.predict(survival_probability=survival_probability)

    Plotter.plot_several(
        [object_data, meas_data],
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
    pprint(estimates)

    # plt.show()
