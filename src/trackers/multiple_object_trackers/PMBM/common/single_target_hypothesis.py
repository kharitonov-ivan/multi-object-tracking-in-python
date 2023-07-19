import typing as tp

import numpy as np
from src.common.gaussian_density import GaussianDensity
from src.measurement_models import MeasurementModel

from .bernoulli import Bernoulli


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
        missdetection_bernoulli = self.bernoulli.undetected_update_state(
            detection_probability
        )
        missdetection_loglikelihood = self.bernoulli.undetected_update_loglikelihood(
            detection_probability
        )
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
        assert measurement.ndim == 1
        detection_bernoulli = self.bernoulli.detected_update_state(
            measurement, meas_model, density
        )
        detection_log_likelihood = detection_bernoulli.detected_update_loglikelihood(
            measurement, meas_model, detection_probability, density
        )

        missdetection_log_likelihood = (
            self.missdetection_hypothesis.log_likelihood
            or self.bernoulli.undetected_update_loglikelihood(detection_probability)
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
        next_states, next_covariances, _ = density.update(
            initial_state=self.bernoulli.state,
            measurements=measurements,
            model_measurement=model_measurement,
        )

        missdetection_log_likelihood = (
            self.missdetection_hypothesis.log_likelihood
            or self.bernoulli.undetected_update_loglikelihood(detection_probability)
        )
        state = GaussianDensity(
            next_states[0], next_covariances[0]
        )  # from one state to many measurements
        loglikelihoods = (
            density.predict_loglikelihood(state, measurements, model_measurement)
            + np.log(detection_probability)
            + np.log(1.0)
        )

        return {
            idx: SingleTargetHypothesis(
                bernoulli=Bernoulli(
                    state=GaussianDensity(
                        next_states[0], next_covariances[0]
                    ),  # TODO проверить
                    existence_probability=1.0,
                ),
                log_likelihood=loglikelihoods[idx],
                cost=-(loglikelihoods[idx] - missdetection_log_likelihood),
                sth_id=sth_ids[idx],
            )
            for idx in range(len(measurements))
        }
