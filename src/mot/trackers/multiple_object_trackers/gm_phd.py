from mot.common.hypothesis_reduction import Hypothesisreduction
import numpy as np
from mot.common.state import Gaussian
from mot.common.state import Gaussian
from mot.configs import SensorModelConfig
from mot.measurement_models import (
    MeasurementModel,
)
from mot.motion_models import (
    MotionModel,
)
from tqdm import tqdm as tqdm
from typing import List

from mot.common.state import Gaussian, WeightedGaussian, GaussianMixture
from mot.common.gaussian_density import GaussianDensity as GD
from scipy.stats import chi2

import logging

logging.basicConfig(level=logging.DEBUG)


class GMPHD:
    """Tracks multiple objects using Gaussian Mixture Probanility Hypothesis Density filter"""

    """Vo, B.-N., & Ma, W.-K. (2006). The Gaussian Mixture Probability Hypothesis Density Filter. IEEE Transactions on Signal Processing, 54(11), 4091–4104. doi:10.1109/TSP.2006.881190"""

    def __init__(
        self,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: MotionModel,
        birth_model: GaussianMixture,
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
        self.gating_size = chi2.ppf(P_G, df=self.meas_model.d)
        self.w_min = np.log(w_min)
        self.M = M
        self.P_D = P_D
        self.merging_threshold = merging_threshold
        self.gmm_components = GaussianMixture([])

    @property
    def method(self):
        return "GM-PHD-tracker"

    def PHD_estimator(self):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        # Performs obhect state estimation in the GM PHD filetes
        # Output: estimates object states in matrix from of size (object state dimention) x (number of objects)
        # Geat a mean estimate (expected number of objects) if the cardinality of object sby takings the summation of the weights of the Gaussian components
        # (rounded to the nearest integes), denotes as n

        E = np.sum(np.exp(self.gmm_components.weights))  # number exptected value
        n = np.min([len(self.gmm_components), int(np.round(E))])
        top_n_hypotheses = sorted(self.gmm_components, key=lambda x: x.w)[:n]
        estimates = [comp.gm for comp in top_n_hypotheses]
        return estimates

    def estimate(self, measurements, verbose=False):
        """Tracks multilple

        For each filter recursion iteration implemented next steps:
        1) for each prior perform prediction
        2) for each hypothesis, perform ellipsoidal gating
           and only create object detection hypotheses for detections
        3) construct 2D cost matrix of size
           (number of objects x number of z_ingate + number of objects)
        4) find best assignment using a 2D assignment solver
        5) create new local hypotheses accotding to the best assgnment matrix obtained
        6) get obhect state estimates
        7) preform prefict for each local hypotheses
        """

        estimations = [[] for x in range(len(measurements))]
        for timestep, measurements_in_scene in tqdm(
            enumerate(measurements), total=len(measurements)
        ):
            estimations[timestep].extend(self.estimation_step(measurements_in_scene))
        return tuple(estimations)

    def estimation_step(self, current_measurements: np.ndarray):
        # PPP update
        self.update(current_measurements)

        # PPP approximation
        self.components_reduction()

        # # Extract state estimates from the PPP
        estimates = self.PHD_estimator()

        # # PPP prediction
        self.predict_step()
        logging.debug(f"number of components: {len(self.gmm_components)}")
        return estimates

    def predict_step(self):
        # surviving process - predict and store the components that survived
        for idx in range(len(self.gmm_components)):
            self.gmm_components[idx].gm = GD.predict(
                self.gmm_components[idx].gm, self.motion_model
            )
            self.gmm_components[idx].w += np.log(self.P_S)

        # birth process - copy birth weight and birth states
        self.gmm_components.extend(self.birth_model)

    def update(self, z):
        """Performs PPP update step and PPP approximation"""
        # TODO check shape of measurements

        new_gmm_components = GaussianMixture()
        # Construct update components resulted from missed detection.
        # It represents objects that are undetected at time k.
        missdetection_hypotheses = GaussianMixture(
            [
                WeightedGaussian(weight=comp.w + np.log(1 - self.P_D), gm=comp.gm)
                for comp in self.gmm_components
            ]
        )
        new_gmm_components.extend(missdetection_hypotheses)

        for meas_idx, measurement in enumerate(z):
            new_gm_list = GaussianMixture([])
            for hyp_idx in range(len(self.gmm_components)):

                component = self.gmm_components[hyp_idx]
                z_in_gate, is_meas_in_gate = GD.ellipsoidal_gating(
                    component.gm,
                    np.array(measurement, ndmin=2),
                    gating_size=self.gating_size,
                    measurement_model=self.meas_model,
                )
                if is_meas_in_gate[0]:
                    new_gm = GD.update(
                        component.gm, np.array(measurement, ndmin=2), self.meas_model
                    )
                    predicted_likelihood = GD.predicted_likelihood(
                        component.gm, np.array(measurement, ndmin=2), self.meas_model
                    ).squeeze()
                    w = np.log(self.P_D) + predicted_likelihood + component.w
                    new_gm_list.append(WeightedGaussian(weight=w, gm=new_gm))

            W_sum = np.sum(new_gm_list.weights)
            for idx, _ in enumerate(new_gm_list):
                new_gm_list[idx].w = np.log(
                    new_gm_list[idx].w / (self.sensor_model.intensity_c + W_sum)
                )
            if len(new_gm_list) > 0:
                new_gmm_components.extend(new_gm_list)

        self.gmm_components = new_gmm_components

    def components_reduction(self):
        # apptoximates the PPP by represingti its intensity with fewer parameters
        try:
            (pruned_hypotheses_weight, pruned_hypotheses,) = Hypothesisreduction.prune(
                self.gmm_components.weights,
                self.gmm_components.states,
                threshold=self.w_min,
            )

            self.gmm_components = GaussianMixture(
                [
                    WeightedGaussian(w, gm)
                    for (w, gm) in zip(pruned_hypotheses_weight, pruned_hypotheses)
                ]
            )
        except:
            print("Empty hupotheses")

        # Hypotheses merging
        if len(self.gmm_components.weights) > 1:
            (merged_hypotheses_weights, merged_hypotheses,) = Hypothesisreduction.merge(
                self.gmm_components.weights,
                self.gmm_components.states,
                threshold=self.merging_threshold,
            )

            self.gmm_components = GaussianMixture(
                [
                    WeightedGaussian(w, gm)
                    for (w, gm) in zip(merged_hypotheses_weights, merged_hypotheses)
                ]
            )

        # Cap the number of the hypotheses and then re-normalise the weights
        (capped_hypotheses_weights, capped_hypotheses,) = Hypothesisreduction.cap(
            self.gmm_components.weights, self.gmm_components.states, top_k=self.M
        )

        self.gmm_components = GaussianMixture(
            [
                WeightedGaussian(w, gm)
                for (w, gm) in zip(capped_hypotheses_weights, capped_hypotheses)
            ]
        )