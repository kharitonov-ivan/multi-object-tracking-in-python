import numpy as np
import pytest

from src.run import run_tracker


@pytest.mark.usefixtures("env_measurement_model", "env_clutter_rate", "env_detection_probability", "filepath_fixture")
def test_synthetic_scenario(object_motion_fixture, env_detection_probability, env_clutter_rate, env_measurement_model, tracker, filepath_fixture):
    env_motion_model, object_motion_cfg = object_motion_fixture.motion_model(sigma_q=10.0, random_state=42), object_motion_fixture.object_configs
    tracker_constructor, tracker_params = tracker
    run_tracker(
        object_configs=object_motion_cfg,
        total_time=100,
        env_P_D=env_detection_probability,
        env_lambda_c=env_clutter_rate,
        env_range_c=np.array([[-1000, 1000], [-1000, 1000]]),
        env_motion_model=env_motion_model,
        env_meas_model=env_measurement_model,
        tracker=tracker_constructor,
        tracker_params=tracker_params,
        filepath=filepath_fixture,
    )
