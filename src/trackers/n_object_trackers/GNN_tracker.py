import numpy as np
from scipy.optimize import linear_sum_assignment
from src.common.gaussian_density import GaussianDensity
from src.configs import SensorModelConfig
from src.measurement_models import MeasurementModel
from src.motion_models import BaseMotionModel
from tqdm import tqdm as tqdm

from .base_n_object_tracker import KnownObjectTracker


class GlobalNearestNeighboursTracker(KnownObjectTracker):
    def __init__(
        self,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: BaseMotionModel,
        M,
        merging_threshold,
        P_G,
        w_min,
        intensity: GaussianDensity,
    ) -> None:
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.w_min = w_min
        self.P_G = P_G
        self.gating_size = P_G
        self.M = M
        self.n = 5
        self.merging_threshold = merging_threshold
        self.hypotheses_weight = None
        self.multi_hypotheses_bank = None
        self.intensity = intensity
        super(GlobalNearestNeighboursTracker).__init__()

    def step(self, measurements: np.ndarray):
        """Tracks a single object using Gauss sum filtering

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
        self.intensity = GaussianDensity.predict(
            self.intensity, self.motion_model, dt=1.0
        )

        # 1) elipsoidal gating separately for each object
        mask, dists = GaussianDensity.ellipsoidal_gating(
            self.intensity, measurements, self.meas_model, self.gating_size
        )
        mask = np.ones_like(mask, dtype=bool)
        # 2) Disconsider measurements which do not fall inside any object gate
        mask_to_keep = np.sum(mask, axis=0, dtype=bool)
        source_to_considered_ids = np.where(mask_to_keep)[0]
        considered_to_source_ids = np.sort(source_to_considered_ids)
        considered_measurements = measurements[source_to_considered_ids]
        n_filtered_measurements = considered_measurements.shape[0]

        # 3) construct 2D cost matrix of size
        #    (number of objects x number of z_ingate + number of objects)
        cost_matrix = np.full((self.n, n_filtered_measurements + self.n), np.inf)

        # Misdetection cost
        cost_matrix[:, n_filtered_measurements:] = -np.log(
            1 - self.sensor_model.P_D
        )  # misdetection cost

        # Detection cost -log(sensormodel.P_D/sensormodel.intensity_c) - predicted_likelihood_log;    % detection weights
        ll = GaussianDensity.predict_loglikelihood(
            self.intensity, considered_measurements, self.meas_model
        )
        cost_matrix[:, :n_filtered_measurements] = -ll - np.log(
            self.sensor_model.P_D / self.sensor_model.intensity_c
        )

        # 4) find best assignment using a 2D assignment solver
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 5) create new local hypotheses accotding to the best assgnment matrix obtained
        for object_idx in range(self.n):
            if col_ind[object_idx] >= n_filtered_measurements:
                # Misdetection
                pass
            else:
                # Detection
                measurement_idx = considered_to_source_ids[col_ind[object_idx]]
                updated_means, updated_covs, _ = GaussianDensity.update(
                    self.intensity[object_idx],
                    measurements[measurement_idx, None],
                    self.meas_model,
                )
                self.intensity[object_idx] = GaussianDensity(
                    updated_means[0], updated_covs[0]
                )

        # 6) get object state estimates
        return {idx: self.intensity[idx] for idx in range(len(self.intensity))}
