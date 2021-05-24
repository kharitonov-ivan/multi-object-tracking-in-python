from copy import deepcopy
from typing import Dict

from .....common import GaussianMixture


class BirthModel:
    def get_born_objects_intensity(self, params) -> GaussianMixture:
        raise NotImplementedError


class StaticBirthModel(BirthModel):
    def __init__(self, birth_model_config: GaussianMixture):
        self.birth_model_config = deepcopy(birth_model_config)
        super(StaticBirthModel, self).__init__()

    def get_born_objects_intensity(self, params=None) -> GaussianMixture:
        birth_model = deepcopy(self.birth_model_config)

        if params is not None:
            ego_pose = params["ego_pose"]
            translation, rotation = ego_pose["translation"], ego_pose["rotation"]

            for i, birth_component in enumerate(birth_model):
                birth_model[i].gaussian.x[:2] += translation[:2]

        return birth_model


class RandomSampledBirthModel(BirthModel):
    def __init__(self):
        super(RandomSampledBirthModel, self).__init__()
        raise NotImplementedError


class MeasurementDrivenBirthModel(BirthModel):
    def __init__(self):
        super(MeasurementDrivenBirthModel, self).__init__()

    def get_born_objects_intensity(self, measurements) -> GaussianMixture:
        raise NotImplementedError
