import numpy as np

from mot.common import Gaussian, GaussianMixture, WeightedGaussian, GaussianMixtureNumpy


birth_model_params = GaussianMixtureNumpy(np.array([0.0, 0.0, 0.0, 0.0]), 100 * np.eye(4), np.log(0.03)) + \
                     GaussianMixtureNumpy(np.array([400.0, -600.0, 0.0, 0.0]), 100 * np.eye(4), np.log(0.03)) + \
                     GaussianMixtureNumpy(np.array([-800.0, 200.0, 0.0, 0.0]), 100 * np.eye(4), np.log(0.03)) + \
                     GaussianMixtureNumpy(np.array([-200.0, 800.0, 0.0, 0.0]), 100 * np.eye(4), np.log(0.03))