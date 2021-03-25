import numpy as np
from mot.common.gaussian_density import GaussianDensity
from mot.simulator.measurement_data_generator import MeasurementData
from mot.measurement_models import ConstantVelocityMeasurementModel
from mot.configs import GroundTruthConfig, Object, SensorModelConfig
from mot.motion_models import ConstantVelocityMotionModel
from mot.simulator.object_data_generator import ObjectData
from mot.common.state import State

TOL = 1e-4


def test_predict_likelihood():
    n_births = 1
    total_time = 10
    objects = [
        Object(
            initial=State(x=np.array([0.0, 0.0, 5.0, 5.0]), P=np.eye(4)),
            t_birth=0,
            t_death=10,
        )
    ]
    grount_truth_config = GroundTruthConfig(
        n_births=n_births, object_configs=objects, total_time=total_time
    )

    test_dt = 1.0
    test_sigma_q = 2.0
    motion_model = ConstantVelocityMotionModel(dt=test_dt, sigma_q=test_sigma_q)

    test_P_D = 1.0
    test_lambda_c = 10.0
    test_range_c = np.array([[-59.0, 50.0], [-50, 50]])
    sensor_model = SensorModelConfig(
        P_D=test_P_D, lambda_c=test_lambda_c, range_c=test_range_c
    )

    test_sigma_r = 5.0
    meas_model = ConstantVelocityMeasurementModel(sigma_r=test_sigma_r)

    object_data = ObjectData(grount_truth_config, motion_model, if_noisy=False)

    meas_data = MeasurementData(
        object_data=object_data, sensor_model=sensor_model, meas_model=meas_model
    )

    expected_predicted_likelihood = np.array(
        [
            -5.90643946792181,
            -21.0573161628118,
            -106.879840108121,
            -194.607232058810,
            -347.240521996516,
            -206.002804285254,
            -17.8573220798290,
            -158.897617023405,
        ]
    )
    for timestep in range(grount_truth_config.total_time):

        state = State(
            x=np.array(object_data.X[timestep][0].x), P=np.eye(motion_model.d)
        )

        # Run learner solution
        predicted_likelihood = GaussianDensity.predicted_likelihood(
            state_pred=state,
            z=np.array(meas_data.measurements[timestep]),
            measurement_model=meas_model,
        )
