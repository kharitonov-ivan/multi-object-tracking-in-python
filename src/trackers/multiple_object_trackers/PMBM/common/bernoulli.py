import typing as tp

import numpy as np
from copy import copy, deepcopy
from src.common.gaussian_density import GaussianDensity
from src.common.state import ObjectMetadata
from src.measurement_models import MeasurementModel
from src.motion_models import BaseMotionModel


class Bernoulli:
    """Bernoulli component

    Parameters
    ----------
    state : GaussianDensity
        a struct contains parameters describing the object pdf
    r : scalar
        probability of existence
    """

    def __init__(
        self,
        state: GaussianDensity,
        existence_probability: float,
        metadata: ObjectMetadata = None,
    ):
        self.state: GaussianDensity = deepcopy(state)
        self.existence_probability: float = existence_probability
        self.metadata = metadata

    def __repr__(self) -> str:
        return self.__class__.__name__ + (f"(r={self.existence_probability:.4f}, " f"state={self.state}")

    def predict(self, motion_model: BaseMotionModel, survival_probability: float, density: GaussianDensity, dt: float) -> None:
        self.existence_probability *= survival_probability
        self.state = density.predict(self.state, motion_model, dt)

    def undetected_update_state(self, detection_probability: float):
        """Calculates the likelihood of missed detection,
        and creates new local hypotheses due to missed detection.
        NOTE: from page 88 lecture 04
        """

        posterior_existence_probability = (self.existence_probability * (1 - detection_probability)) / (
            1 - self.existence_probability + self.existence_probability * (1 - detection_probability)
        )
        return Bernoulli(self.state, posterior_existence_probability)

    def undetected_update_loglikelihood(self, detection_probability: float):
        missdetecion_probability = 1 - detection_probability
        likelihood_predicted = 1 - self.existence_probability + self.existence_probability * missdetecion_probability
        log_likelihood_predicted = np.log(likelihood_predicted)
        return log_likelihood_predicted

    def detected_update_loglikelihood(
        self,
        measurement: np.ndarray,
        meas_model: MeasurementModel,
        detection_probability: float,
        density=GaussianDensity,
    ) -> float:
        """Calculates the predicted likelihood for a given local hypothesis.
        NOTE page 86 lecture 04
        """
        log_likelihood_detected = density.predict_loglikelihood(self.state, measurement, meas_model).item() + np.log(detection_probability * self.existence_probability)
        return log_likelihood_detected

    def detected_update_state(
        self,
        measurement: np.ndarray,
        meas_model: MeasurementModel,
        density=GaussianDensity,
    ):
        """Creates the new local hypothesis due to measurement update.
        NOTE: page 85 lecture 04
        """

        new_means, new_covs, _ = density.update(self.state, measurement, meas_model)
        return Bernoulli(state=GaussianDensity(new_means[0], new_covs[0]), existence_probability=1.0)


class SingleTargetHypothesis:
    def __init__(
        self,
        bernoulli: Bernoulli,
        log_likelihood: float,
        cost: float = None,
        meas_idx: int = None,
        sth_id: int = None,
    ):
        assert isinstance(bernoulli, Bernoulli)
        self.bernoulli = bernoulli
        self.log_likelihood = log_likelihood
        self.cost = cost
        self.meas_idx = meas_idx  # associated measurements
        self.sth_id = sth_id
        self.missdetection_hypothesis = None
        self.detection_hypotheses = {}

    def create_missdetection_hypothesis(self, detection_probability: float, sth_id):
        missdetection_bernoulli = self.bernoulli.undetected_update_state(detection_probability)
        missdetection_loglikelihood = self.bernoulli.undetected_update_loglikelihood(detection_probability)
        missdetection_hypothesis = SingleTargetHypothesis(
            bernoulli=missdetection_bernoulli,
            log_likelihood=missdetection_loglikelihood.item(),
            cost=0,
            sth_id=sth_id,
        )
        return missdetection_hypothesis

    def create_detection_hypothesis(
        self,
        measurement: np.ndarray,
        detection_probability: float,
        meas_model: MeasurementModel,
        density: GaussianDensity,
        sth_id: int,
    ):
        detection_bernoulli = self.bernoulli.detected_update_state(measurement, meas_model, density)
        detection_log_likelihood = detection_bernoulli.detected_update_loglikelihood(measurement, meas_model, detection_probability, density)

        missdetection_log_likelihood = (
            self.missdetection_hypothesis.log_likelihood
            # or self.bernoulli.undetected_update_loglikelihood(detection_probability)
        )

        detection_hypothesis = SingleTargetHypothesis(
            bernoulli=detection_bernoulli,
            log_likelihood=detection_log_likelihood.item(),
            cost=(-(detection_log_likelihood - missdetection_log_likelihood)).item(),
            sth_id=sth_id,
        )
        return detection_hypothesis

    def create_detection_hypotheses(
        self,
        measurements: tp.Annotated[np.ndarray, "(n_measurements, dim_measurement)"],
        detection_probability: float,
        model_measurement: MeasurementModel,
        density: GaussianDensity,
        sth_ids: tuple[int],
    ):
        # next_states, next_covariances, _ = density.update(
        #     initial_state=self.bernoulli.state,
        #     measurements=measurements,
        #     model_measurement=model_measurement,
        # ) # (1, n_meas, n_dim), (1, n_meas, n_dim, n_dim)

        # new_states = GaussianDensity(
        #     next_states[0], next_covariances[0]
        # )  # from one state to many measurements
        # import pdb; pdb.set_trace()

        # missdetection_log_likelihood = (
        #     self.missdetection_hypothesis.log_likelihood
        #     or self.bernoulli.undetected_update_loglikelihood(detection_probability)
        # )

        # loglikelihoods = (
        #     density.predict_loglikelihood(new_states, measurements, model_measurement)
        #     + np.log(detection_probability)
        #     + np.log(1.0)
        # )

        # return {
        #     idx: SingleTargetHypothesis(
        #         bernoulli=Bernoulli(
        #             state=GaussianDensity(
        #                 next_states[0], next_covariances[0]
        #             ),
        #             existence_probability=1.0,
        #         ),
        #         log_likelihood=loglikelihoods[idx],
        #         cost=-(loglikelihoods[idx] - missdetection_log_likelihood),
        #         sth_id=sth_ids[idx],
        #     )
        #     for idx in range(len(measurements))
        # }
        answer = {}
        for meas_idx in range(len(measurements)):
            answer[meas_idx] = self.create_detection_hypothesis(
                measurements[meas_idx][None, ...],
                detection_probability,
                model_measurement,
                density,
                sth_ids[meas_idx],
            )

        return answer
