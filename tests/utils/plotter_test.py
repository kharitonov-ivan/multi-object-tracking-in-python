import numpy as np
import pytest

from src.common.state import Gaussian
from src.configs import GroundTruthConfig
from src.motion_models import ConstantVelocityMotionModel
from src.run import animate, generate_environment, visulaize
from src.scenarios.initial_conditions import all_object_scenarios
from src.simulator import ObjectData
from src.utils.get_path import delete_images_dir, get_images_dir


@pytest.fixture(params=[x for x in all_object_scenarios if (len(x.object_configs) == 1) and (x.motion_model == ConstantVelocityMotionModel)])
def object_motion_fixture(request):
    yield request.param


@pytest.fixture(scope="session", autouse=True)
def do_something_before_all_tests():
    delete_images_dir(__file__)


@pytest.fixture()
def env_motion_model(object_motion_fixture):
    yield ConstantVelocityMotionModel(random_state=42, sigma_q=1.0)


def test_plot_object_data(object_motion_fixture, env_motion_model):
    total_time = 100
    ground_truth = GroundTruthConfig(object_motion_fixture.object_configs, total_time)
    object_data = ObjectData(ground_truth, env_motion_model, if_noisy=False)
    visulaize(object_data, None, None, get_images_dir(__file__) + "/" + "obj_data" + ".png")


def test_plot_meas_data(object_motion_fixture, env_motion_model, env_measurement_model, env_detection_probability, env_clutter_rate):
    total_time = 100
    env_range_c = np.array([[-1000, 1000], [-1000, 1000]])
    ground_truth, env_sensor_model, object_data, meas_data = generate_environment(
        object_motion_fixture.object_configs, total_time, env_motion_model, env_measurement_model, env_detection_probability, env_clutter_rate, env_range_c
    )
    visulaize(object_data, meas_data, None, get_images_dir(__file__) + "/" + "meas_data" + ".png")


def test_plot_one_gaussian():
    gaussian = Gaussian(x=np.array([0, 0, 10, 10]), P=np.diag([400, 200, 0, 0]))
    visulaize(None, None, [{0: gaussian}], get_images_dir(__file__) + "/" + "one_gaussian" + ".png")


def test_plot_gaussians(object_motion_fixture, env_motion_model, env_measurement_model, env_detection_probability, env_clutter_rate):
    total_time = 100
    env_range_c = np.array([[-1000, 1000], [-1000, 1000]])
    ground_truth, env_sensor_model, object_data, meas_data = generate_environment(
        object_motion_fixture.object_configs, total_time, env_motion_model, env_measurement_model, env_detection_probability, env_clutter_rate, env_range_c
    )
    estimations = [
        {
            idx: Gaussian(x=pos, P=400 * np.eye(4))
            for idx, pos in enumerate(
                [
                    np.array([0, 0, 0, 0]),
                    np.array([400, -600, 0, 0]),
                    np.array([-800, -200, 0, 0]),
                    np.array([-200, 800, 0, 0]),
                ]
            )
        }
    ] * 10
    visulaize(object_data, meas_data, estimations, get_images_dir(__file__) + "/" + "gaussians" + ".png")


def test_animate_gaussians(object_motion_fixture, env_motion_model, env_measurement_model, env_detection_probability, env_clutter_rate):
    total_time = 100
    env_range_c = np.array([[-1000, 1000], [-1000, 1000]])
    ground_truth, env_sensor_model, object_data, meas_data = generate_environment(
        object_motion_fixture.object_configs, total_time, env_motion_model, env_measurement_model, env_detection_probability, env_clutter_rate, env_range_c
    )
    estimations = [
        {
            idx: Gaussian(x=pos, P=400 * np.eye(4))
            for idx, pos in enumerate(
                [
                    np.array([0, 0, 0, 0]),
                    np.array([400, -600, 0, 0]),
                    np.array([-800, -200, 0, 0]),
                    np.array([-200, 800, 0, 0]),
                ]
            )
        }
    ] * 10
    animate(object_data, meas_data, estimations, get_images_dir(__file__) + "/" + "animate" + ".gif")
