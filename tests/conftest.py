import numpy as np
import pytest

from src.common.state import Gaussian, GaussianMixture, WeightedGaussian
from src.measurement_models import ConstantVelocityMeasurementModel
from src.utils.get_path import get_images_dir


@pytest.fixture()
def artefacts_folder(_file_):
    yield get_images_dir(_file_)


@pytest.fixture()
def filepath_fixture(artefacts_folder, object_motion_fixture, env_clutter_rate, env_detection_probability, env_measurement_model, tracker):
    components = [
        tracker[0].__name__,
        object_motion_fixture.motion_model.__name__,
        object_motion_fixture.movement_type,
        f"n_objects={len(object_motion_fixture.object_configs)}",
        f"clutter_rate={env_clutter_rate}",
        f"detection_probability={env_detection_probability}",
        f"measurement_model={env_measurement_model.__class__.__name__}",
    ]
    yield artefacts_folder + "/" + "-".join(components)


@pytest.fixture(params=[0.9])
def env_detection_probability(request):
    yield request.param


@pytest.fixture(params=[0.1, 10.0])
def env_clutter_rate(request):
    yield request.param


@pytest.fixture
def env_measurement_model(request):
    yield ConstantVelocityMeasurementModel(sigma_r=10.0)


@pytest.fixture(scope="function")
def initial_PPP_intensity_linear():
    return GaussianMixture(
        [
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([-400.0, 200.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
            WeightedGaussian(
                np.log(0.03),
                Gaussian(x=np.array([-400.0, -200.0, 0.0, 0.0]), P=400 * np.eye(4)),
            ),
        ]
    )


@pytest.fixture(scope="function")
def initial_PPP_intensity_nonlinear():
    return GaussianMixture(
        [
            WeightedGaussian(
                -0.3861,
                Gaussian(x=np.array([0.0, 0.0, 5.0, 0.0, np.pi / 180]), P=np.eye(5)),
            ),
            WeightedGaussian(
                -0.423,
                Gaussian(x=np.array([20.0, 20.0, -20.0, 0.0, np.pi / 90]), P=np.eye(5)),
            ),
            WeightedGaussian(
                -1.8164,
                Gaussian(x=np.array([-20.0, 10.0, -10.0, 0.0, np.pi / 360]), P=np.eye(5)),
            ),
        ]
    )
