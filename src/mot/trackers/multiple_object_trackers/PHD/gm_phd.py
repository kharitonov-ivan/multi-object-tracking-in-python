import logging
from chardet import detect

import numpy as np
from scipy.stats import chi2
from tqdm import tqdm as tqdm

from mot.common.gaussian_density import GaussianDensity
from mot.common.hypothesis_reduction import HypothesisReduction
from mot.configs import SensorModelConfig
from mot.measurement_models import MeasurementModel
from mot.motion_models import BaseMotionModel


class GMPHD:
    """Tracks multiple objects using Gaussian Mixture Probanility Hypothesis Density filter

    Vo, B.-N., & Ma, W.-K. (2006).
    The Gaussian Mixture Probability Hypothesis Density Filter.
    IEEE Transactions on Signal Processing, 54(11),
    4091–4104. doi:10.1109/TSP.2006.881190"""

    def __init__(
        self,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: BaseMotionModel,
        birth_model: GaussianDensity,
        M,
        merging_threshold,
        P_G,
        w_min,
        P_D,
        P_S: float,
        *args,
        **kwargs,
    ):
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.birth_model = birth_model
        self.P_S = P_S
        self.gating_size = P_G
        self.w_min = np.log(w_min)
        self.M = M
        self.P_D = P_D
        self.merging_threshold = merging_threshold
        self.gmm_components = GaussianDensity()

    @property
    def method(self):
        return "GM-PHD-tracker"

    def estimate(self):
        """
        Performs obhect state estimation in the GM PHD filetes
        Output: estimates object states in matrix from of size
        (object state dimention) x (number of objects)
        Geat a mean estimate (expected number of objects)
        if the cardinality of object sby takings the summation of
        the weights of the Gaussian components
        (rounded to the nearest integes), denotes as n
        """
        if self.gmm_components:
            E = np.sum(np.exp(self.gmm_components.weights))  # number exptected value
            n = np.min([len(self.gmm_components), int(np.round(E))])
            top_n_hypotheses = sorted(self.gmm_components, key=lambda x: x.weights)[:n]
            estimates = [comp for comp in top_n_hypotheses]
            return estimates

    def step(self, current_measurements: np.ndarray):
        """
        Tracks multilple
        For each filter recursion iteration implemented next steps:
        0) preform prefict for each local hypotheses
        1) for each prior perform prediction
        2) for each hypothesis, perform ellipsoidal gating
           and only create object detection hypotheses for detections
        3) construct 2D cost matrix of size
           (number of objects x number of z_ingate + number of objects)
        4) find best assignment using a 2D assignment solver
        5) create new local hypotheses accotding to the best assgnment matrix obtained
        6) get obhect state estimates
        """

        # # PPP prediction
        self.predict_step(dt=1.0)

        # PPP update
        self.update(current_measurements)

        # PPP approximation
        self.components_reduction()

        # # Extract state estimates from the PPP
        estimates = self.estimate()

        return estimates

    def predict_step(self, dt):
        # surviving process - predict and store the components that survived
        if len(self.gmm_components) > 0:
            self.gmm_components = GaussianDensity.predict(self.gmm_components, self.motion_model, dt)
            self.gmm_components.weights += np.log(self.P_S)

        # birth process - copy birth weight and birth states
        self.gmm_components = self.gmm_components + self.birth_model if self.gmm_components else self.birth_model

    def update(self, measurements):
        """Performs PPP update step and PPP approximation"""
        # TODO check shape of measurements

        # Construct update components resulted from missed detection.
        # It represents objects that are undetected at time k.

        missdetection_hypotheses = self.gmm_components
        if missdetection_hypotheses:
            missdetection_hypotheses.weights += np.log(1 - self.P_D)

        detection_hypotheses = GaussianDensity()
        mask, dists = GaussianDensity.ellipsoidal_gating(
            self.gmm_components,
            measurements,
            self.meas_model,
            self.gating_size,
        )

        new_means, new_covs, _ = GaussianDensity.update(
            self.gmm_components,
            measurements,
            self.meas_model,
        )

        predicted_likelihood = GaussianDensity.predict_loglikelihood(self.gmm_components, measurements, self.meas_model)
        new_weights = np.log(self.P_D) + predicted_likelihood + self.gmm_components.weights[..., None]
        detection_hypotheses += GaussianDensity(new_means[mask], new_covs[mask], new_weights[mask])

        W_sum = np.sum(detection_hypotheses.weights)
        detection_hypotheses.weights = np.log(detection_hypotheses.weights / (self.sensor_model.intensity_c + W_sum))
        self.gmm_components = detection_hypotheses + missdetection_hypotheses

    def components_reduction(self):
        if len(self.gmm_components) == 0:
            return  # nothing to reduce

        # Delete components with weight less than w_min
        self.gmm_components = HypothesisReduction.prune(self.gmm_components, threshold=self.w_min)

        # Hypotheses merging
        # if len(self.gmm_components) > 1:
        #     self.gmm = HypothesisReduction.merge(
        #         self.gmm
        #         threshold=self.merging_threshold,
        #     )

        # Cap the number of the hypotheses and then re-normalise the weights
        self.gmm_components = HypothesisReduction.cap(self.gmm_components, top_k=self.M)
