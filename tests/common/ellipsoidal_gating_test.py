import numpy as np
from scipy.stats import chi2

from src.common.gaussian_density import GaussianDensity
from src.common.state import Gaussian
from src.configs import GroundTruthConfig, Object, SensorModelConfig
from src.measurement_models import ConstantVelocityMeasurementModel
from src.motion_models import ConstantVelocityMotionModel
from src.simulator.measurement_data_generator import MeasurementData
from src.simulator.object_data_generator import ObjectData


TOL = 1e-4


def test_ellipsoidal_gating():
    total_time = 10
    objects = [
        Object(
            initial=Gaussian(x=np.array([0.0, 0.0, 5.0, 5.0]), P=np.eye(4)),
            t_birth=0,
            t_death=10,
        )
    ]

    grount_truth_config = GroundTruthConfig(object_configs=objects, total_time=total_time)
    test_dt = 1.0
    test_sigma_q = 2.0
    motion_model = ConstantVelocityMotionModel(dt=test_dt, sigma_q=test_sigma_q)

    test_P_D = 1.0
    test_lambda_c = 60.0
    test_range_c = np.array([[-200.0, 200.0], [-200, 200]])
    sensor_model = SensorModelConfig(P_D=test_P_D, lambda_c=test_lambda_c, range_c=test_range_c)

    test_sigma_r = 10.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r=test_sigma_r)

    object_data = ObjectData(grount_truth_config, motion_model, if_noisy=False)

    meas_data_gen = MeasurementData(object_data=object_data, sensor_model=sensor_model, meas_model=meas_model)
    meas_data = [next(meas_data_gen) for _ in range(len(object_data))]
    gating_size = chi2.ppf(0.99, df=meas_model.d)
    id = grount_truth_config.object_configs[0].id

    states = [None for i in range(grount_truth_config.total_time)]
    for timestep in range(0, grount_truth_config.total_time):
        states[timestep] = Gaussian(x=np.array(object_data[timestep][id].x), P=np.eye(motion_model.d))

        if timestep == 0:
            continue
        [z_ingate, meas_in_gate] = GaussianDensity.ellipsoidal_gating(
            state_prev=states[timestep - 1],
            z=meas_data[timestep][1],
            measurement_model=meas_model,
            gating_size=gating_size,
        )
