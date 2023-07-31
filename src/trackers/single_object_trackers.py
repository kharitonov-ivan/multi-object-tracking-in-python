import numpy as np
from scipy.stats import chi2
from tqdm import tqdm as tqdm

from src.common import Gaussian, HypothesisReduction, normalize_log_weights
from src.common.gaussian_density import GaussianDensity
from src.configs import SensorModelConfig
from src.measurement_models import MeasurementModel
from src.motion_models import MotionModel


class SingleObjectTracker:
    def step(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class NearestNeighbourTracker(SingleObjectTracker):
    def __init__(
        self,
        gating_size: float,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: MotionModel,
        initial_state: Gaussian,
        *args,
        **kwargs,
    ) -> None:
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.gating_size = gating_size
        self.state = initial_state
        self.time_step = 0
        super().__init__()

    @property
    def name(self):
        return "Nearest Neighbout SOT"

    def step(self, measurements, dt):
        self.time_step += dt
        self.predict(dt)
        self.update(measurements, dt)
        return self.estimate()

    def predict(self, dt):
        self.state = GaussianDensity.predict(self.state, self.motion_model, dt)

    def update(self, current_measurements: np.ndarray, dt: float):
        """Tracks a single object using nearest neighbour association

        For each filter recursion iteration implemented next steps:
        1) gating
        2) calculates the predicted likelihood for each measurement in the gate
        3) find the nearest neighbour measurement
        4) compares the weight of the missed detection hypotheses and
           the weight of the object detection hypothesis created using
           the nearest neigbour measurement
        5) if the object detection hypothesis using the nearest neighbour
           measurement has the hightes weight, perform Kalman update
        6) extract object state estimate
        7) prediction
        """
        # 1. Gating
        (meas_in_gate, _) = GaussianDensity.ellipsoidal_gating(
            state_prev=self.state,
            z=current_measurements,
            measurement_model=self.meas_model,
            gating_size=self.gating_size,
        )

        if meas_in_gate.size == 0:  # number of hypothesis
            return  # no hypothesis

        # 2. Calculate the predicted likelihood for each measurement in the gate

        predicted_likelihood = GaussianDensity.predict_loglikelihood(
            state_pred=self.state,
            z=meas_in_gate,
            measurement_model=self.meas_model,
        )

        # Hypothesis evaluation
        w_detection_k = predicted_likelihood + np.log(self.sensor_model.P_D / self.sensor_model.intensity_c)
        w_missdetection = 1 - self.sensor_model.P_D

        # 3. Compare the weight of the missed detection
        # hypothesis and the weight of the object detection hypothesis
        # using the nearest neighbour measurement
        max_k = np.argmax(w_detection_k)
        max_w_theta_k = w_detection_k[max_k]

        if w_missdetection < max_w_theta_k:
            # nearest neighbour measurement
            z_NN = meas_in_gate[max_k]
            z_NN = np.atleast_2d(z_NN)
            self.state = GaussianDensity.update(
                state_pred=self.state,
                z=z_NN,
                measurement_model=self.meas_model,
            )

    def estimate(self):
        return {0: self.state}


class ProbabilisticDataAssociationTracker(SingleObjectTracker):
    """Tracks a single object using probabilistic data association

    For each filter recursion iteration implemented next steps:
    1) gating
    2) create missed detection hypothesis
    3) create object detection hypothesis for each detection inside the gate
    4) normalise hypothesis weights
    5) prune hypothesis with small weights, and then re-normalise the weights
    6) merge different hypotheses using Gaussian moment matching
    7) extract object state estimate
    8) prediction
    """

    def __init__(
        self,
        gating_size: float,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: MotionModel,
        initial_state: Gaussian,
        w_min: float = 1e-3,
        *args,
        **kwargs,
    ):
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.w_min = w_min
        self.gating_size = gating_size
        self.state = initial_state
        self.time_step = 0
        super(ProbabilisticDataAssociationTracker).__init__()

    @property
    def name(self):
        return "PDA SOT"

    def step(self, measurements, dt):
        self.time_step += dt
        self.predict(dt)
        self.update(measurements)
        return self.estimate()

    def predict(self, dt):
        self.state = GaussianDensity.predict(self.state, self.motion_model, dt)

    def estimate(self):
        if self.state is not None:
            return {0: self.state}

    def update(self, current_measurements: np.ndarray):
        # 1. Gating
        (meas_in_gate, _) = GaussianDensity.ellipsoidal_gating(
            state_prev=self.state,
            z=current_measurements,
            measurement_model=self.meas_model,
            gating_size=self.gating_size,
        )

        if meas_in_gate.size == 0:  # number of hypothesis
            return

        # 2. Create miss detection hypothesis
        # misdetection
        w_theta_0 = 1 - self.sensor_model.P_D
        multi_hypotheses = [self.state]  # no detection hypothesis

        # 3. Create object detection hypothesis for each detection inside the gate
        predicted_likelihood = []
        for z_ingate in meas_in_gate:
            multi_hypotheses.append(
                GaussianDensity.update(
                    state_pred=self.state,
                    z=z_ingate,
                    measurement_model=self.meas_model,
                )
            )

            ll = GaussianDensity.predict_loglikelihood(state_pred=self.state, z=z_ingate[None, :], measurement_model=self.meas_model)[0]
            predicted_likelihood.append(ll)

        # Hypothesis evaluation
        # detection
        w_theta_factor = np.log(self.sensor_model.P_D / self.sensor_model.intensity_c)
        w_theta_k = np.array(predicted_likelihood) + w_theta_factor

        hypotheses_weights_log = [np.log(w_theta_0)] + w_theta_k.tolist()
        log_w, log_sum_ = normalize_log_weights(hypotheses_weights_log)
        hypotheses_weight_log, multi_hypotheses = HypothesisReduction.prune(
            hypotheses_weights=log_w,
            multi_hypotheses=multi_hypotheses,
            threshold=np.log(self.w_min),
        )

        log_w, log_sum_ = normalize_log_weights(hypotheses_weight_log)
        self.state = GaussianDensity.moment_matching(weights=log_w, states=multi_hypotheses)


class GaussSumTracker(SingleObjectTracker):
    def __init__(
        self,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: MotionModel,
        initial_state: Gaussian,
        M=100,
        merging_threshold=2,
        gating_size=0.99,
        w_min=1e-3,
        *args,
        **kwargs,
    ) -> None:
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.w_min = w_min
        self.P_G = gating_size
        self.gating_size = chi2.ppf(gating_size, df=self.meas_model.d)
        self.M = M
        self.merging_threshold = merging_threshold
        self.hypotheses_weight = [np.log(1.0)]
        self.multi_hypotheses_bank = [initial_state]

        super(GaussSumTracker).__init__()

    def step(self, measurements, dt):
        self.predict(dt)
        self.update(measurements)
        return self.estimate()

    def predict(self, dt):
        # For each hypotheses do prediction
        self.multi_hypotheses_bank = [GaussianDensity.predict(hypothesis, self.motion_model, dt) for hypothesis in self.multi_hypotheses_bank]

    def update(self, current_measurements, verbose=False):
        """Tracks a single object using Gauss sum filtering

        For each filter recursion iteration implemented next steps:
        1) for each hypothesis, create missed detection hypothesis
        2) for each hypothesis, perform ellipsoidal gating
           and only create object detection hypotheses for detections
           inside the gate
        3) normalise hypotheses weights
        4) prune hypotheses with small weights and then re-normalise the weights
        5) hypothese merging
        6) cap the number of the hypotheses and then re-normalise the weights
        7) extract object state estimate using the most probably
           hypothesis estimation
        8) for each hypothesis, perform prediction
        """

        new_hypotheses, new_weights = [], []
        w_theta_factor = np.log(self.sensor_model.P_D / self.sensor_model.intensity_c)
        w_theta_0 = np.log(1 - self.sensor_model.P_D)  # misdetection

        for _old_idx, (curr_weight, curr_hypothesis) in enumerate(zip(self.hypotheses_weight, self.multi_hypotheses_bank)):
            # 1) for each hypothesis, create missed detection hypothesis
            new_hypotheses.append(curr_hypothesis)
            new_weights.append(w_theta_0 + curr_weight)

            # 2) for each hypothesis, perform ellipsoidal gating
            # and only create object detection hypotheses for detection
            # inside the gate
            z_ingate, _ = GaussianDensity.ellipsoidal_gating(
                curr_hypothesis,
                current_measurements,
                self.meas_model,
                self.gating_size,
            )

            if z_ingate.size == 0:
                return

            predicted_likelihood = GaussianDensity.predict_loglikelihood(curr_hypothesis, z_ingate, self.meas_model)

            # for each measurement create detection hypotheses
            for idx, meausurement in enumerate(z_ingate):
                new_hypotheses.append(GaussianDensity.update(curr_hypothesis, meausurement, self.meas_model))
                new_weights.append(predicted_likelihood[idx] + w_theta_factor)

        self.hypotheses_weight = new_weights
        self.multi_hypotheses_bank = new_hypotheses
        assert len(self.hypotheses_weight) == len(self.multi_hypotheses_bank)

        # 3.normalise hypotheses weights
        self.hypotheses_weight, _ = normalize_log_weights(self.hypotheses_weight)

        # 4. Prune hypotheses with small weights and then re-normalise the weights
        self.hypotheses_weight, self.multi_hypotheses_bank = HypothesisReduction.prune(self.hypotheses_weight, self.multi_hypotheses_bank, threshold=np.log(self.w_min))
        self.hypotheses_weight, _ = normalize_log_weights(self.hypotheses_weight)

        # 5. Hypotheses merging and normalize
        self.hypotheses_weight, self.multi_hypotheses_bank = HypothesisReduction.merge(
            self.hypotheses_weight,
            self.multi_hypotheses_bank,
            threshold=self.merging_threshold,
        )
        self.hypotheses_weight, _ = normalize_log_weights(self.hypotheses_weight)

        # 6. Cap the number of the hypotheses and then re-normalise the weights
        self.hypotheses_weight, self.multi_hypotheses_bank = HypothesisReduction.cap(self.hypotheses_weight, self.multi_hypotheses_bank, top_k=self.M)
        self.hypotheses_weight, _ = normalize_log_weights(self.hypotheses_weight)

    def estimate(self):
        if self.multi_hypotheses_bank:
            return {0: self.multi_hypotheses_bank[np.argmax(self.hypotheses_weight)]}

    @property
    def method(self):
        return "gauss sum filter"
