from mot.common.state import Gaussian
from dataclasses import dataclass
from mot.motion_models import MotionModel
from mot.common.gaussian_density import GaussianDensity
from mot.trackers.multiple_object_trackers.PMBM.common.track import (
    SingleTargetHypothesis,
    Track,
)
from typing import List, Tuple
import numpy as np


class Bernoulli:
    """Bernoulli component

    Parameters
    ----------
    state : Gaussian
        a struct contains parameters describing the object pdf
    r : scalar
        probability of existence
    """

    def __init__(self, initial_state: Gaussian, r: float):
        self.state: Gaussian = initial_state
        self.r: float = r

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f"(r={self.r:.2f}, " f"state={self.state}, ")

    @staticmethod
    def predict(
        bern,
        motion_model: MotionModel,
        survival_probability: float,
        density=GaussianDensity,
        dt: float = 1.0,
    ):
        """Performs prediciton step for a Bernoulli component"""

        # Probability of survival * Probability of existence
        next_r = survival_probability * bern.r

        # Kalman prediction of the new state
        next_state = density.predict(bern.state, motion_model, dt)

        predicted_bern = Bernoulli(r=next_r, initial_state=next_state)
        return predicted_bern

    @staticmethod
    def undetected_update(bern, detection_probability: float):
        """Calculates the likelihood of missed detection,
        and creates new local hypotheses due to missed detection.
        NOTE: from page 88 lecture 04

        Parameters
        ----------
        detection_probability : scalar
            object detection probability

        Returns
        -------
        Bern
            [description]

        likelihood_undetectd : scalar

        """

        # missdetection likelihood l_0 = 1 - P_D
        # update probability of existence
        posterior_r = (bern.r * (1 - detection_probability)) / (1 - bern.r + bern.r * (1 - detection_probability))

        posterior_bern = Bernoulli(
            initial_state=bern.state, r=posterior_r
        )  # state remains the same

        # predicted likelihoood
        likelihood_predicted = 1 - bern.r + bern.r * (1 - detection_probability)
        log_likelihood_predicted = np.log(likelihood_predicted)

        return posterior_bern, log_likelihood_predicted

    @staticmethod
    def detected_update_likelihood(
        bern, z, meas_model, detection_probability: float, density=GaussianDensity
    ) -> np.ndarray:
        """Calculates the predicted likelihood for a given local hypothesis.
        NOTE page 86 lecture 04

        Parameters
        ----------
        tt_entry : [type]
            [description]
        z : np.array (measurement dimension x num of measurements)
            [description]
        meas_model : [type]
            [description]
        detection_probability : scalar
            object detection probability

        Returns
        -------
        np.ndarray (number of measurements)
            predicted likelihood on logarithmix scale
        """
        likelihood_detected = (
            density.predicted_likelihood(bern.state, z, meas_model)
            + np.log(detection_probability)
            + bern.r
        )
        return likelihood_detected

    @staticmethod
    def detected_update_state(bern, z, meas_model, density=GaussianDensity):
        """Creates the new local hypothesis due to measurement update.
        NOTE: page 85 lecture 04

        """
        updated_density = density.update(bern.state, z, meas_model)
        update_bern = Bernoulli(initial_state=updated_density, r=1.0)
        return update_bern
