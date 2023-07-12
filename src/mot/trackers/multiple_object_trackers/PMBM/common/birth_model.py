from copy import deepcopy

import numpy as np

from mot.common import Gaussian, GaussianMixture, WeightedGaussian


class BirthModel:
    def get_born_objects_intensity(self, params) -> GaussianMixture:
        raise NotImplementedError


class StaticBirthModel(BirthModel):
    def __init__(self, birth_model_density: GaussianMixture):
        super(StaticBirthModel, self).__init__()
        self.birth_model_density = birth_model_density

    def get_born_objects_intensity(self, measurements= None, ego_pose=None) -> GaussianMixture:
        return self.birth_model_density


class RandomSampledBirthModel(BirthModel):
    def __init__(self):
        super(RandomSampledBirthModel, self).__init__()
        raise NotImplementedError


class MeasurementDrivenBirthModel(BirthModel):
    def __init__(self):
        super(MeasurementDrivenBirthModel, self).__init__()

    def get_born_objects_intensity(self, params) -> GaussianMixture:
        measurements = params["measurements"]
        num_of_born_components_per_measurement = 10

        generated_intensity = []

        random_width = 10
        for measurement in measurements:
            delta_x = np.random.uniform(-random_width, random_width, num_of_born_components_per_measurement)
            delta_y = np.random.uniform(-random_width, random_width, num_of_born_components_per_measurement)

            for d_x, d_y in zip(delta_x, delta_y):
                state = np.array([measurement.measurement[0] + d_x, measurement.measurement[1] + d_y, 0.0, 0.0])
                sample_gaussian = Gaussian(x=state, P=100 * np.eye(4))
                generated_intensity.append(WeightedGaussian(log_weight=np.log(0.03), gaussian=sample_gaussian))
        return GaussianMixture(generated_intensity)
