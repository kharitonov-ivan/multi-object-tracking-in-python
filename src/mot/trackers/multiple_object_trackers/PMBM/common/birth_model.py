from copy import deepcopy
from mot.common import GaussianMixture


class BirthModel:
    def get_born_objects_intensity(self, params) -> GaussianMixture:
        raise NotImplementedError


class StaticBirthModel(BirthModel):
    def __init__(self, birth_model_config: GaussianMixture):
        self.birth_model_config = deepcopy(birth_model_config)
        super(StaticBirthModel, self).__init__()

    def get_born_objects_intensity(self, params=None) -> GaussianMixture:
        return self.birth_model_config


class RandomSampledBirthModel(BirthModel):
    def __init__(self):
        super(RandomSampledBirthModel, self).__init__()
        raise NotImplementedError


class MeasurementDrivenBirthModel(BirthModel):
    def __init__(self):
        super(MeasurementDrivenBirthModel, self).__init__()

    def get_born_objects_intensity(self, measurements) -> GaussianMixture:
        raise NotImplementedError
