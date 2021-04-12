import numpy as np

from .....common import GaussianDensity
from .....measurement_models import MeasurementModel
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

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(log_likelihood={self.log_likelihood:.2f}, "
            f"bernoulli={self.bernoulli}, "
            f"cost={self.cost:.2f}, "
            f"sth_id={self.sth_id}")

    def create_missdetection_hypothesis(self, detection_probability: float,
                                        sth_id):
        missdetection_bernoulli = self.bernoulli.undetected_update_state(
            detection_probability)
        missdetection_loglikelihood = self.bernoulli.undetected_update_loglikelihood(
            detection_probability)
        missdetection_hypothesis = SingleTargetHypothesis(
            bernoulli=missdetection_bernoulli,
            log_likelihood=np.asscalar(missdetection_loglikelihood),
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
            measurement, meas_model, density)
        detection_log_likelihood = detection_bernoulli.detected_update_loglikelihood(
            measurement, meas_model, detection_probability, density)
        missdetection_log_likelihood = (
            self.missdetection_hypothesis.log_likelihood
            or self.bernoulli.undetected_update_loglikelihood(
                detection_probability))

        detection_hypothesis = SingleTargetHypothesis(
            bernoulli=detection_bernoulli,
            log_likelihood=np.asscalar(detection_log_likelihood),
            cost=np.asscalar(-(detection_log_likelihood -
                               missdetection_log_likelihood)),
            sth_id=sth_id,
        )
        return detection_hypothesis
