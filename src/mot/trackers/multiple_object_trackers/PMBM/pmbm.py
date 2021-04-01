import copy
import logging
from typing import List, Tuple

import numpy as np
import scipy
from mot.common.gaussian_density import GaussianDensity
from mot.common.state import GaussianMixture
from mot.configs import SensorModelConfig
from mot.measurement_models import MeasurementModel
from mot.motion_models import MotionModel
from murty import Murty

logging.basicConfig(level=logging.DEBUG)
from mot.common.estimation import Estimation
from mot.common.normalize_log_weights import normalize_log_weights

from .common import (GlobalHypothesis, MultiBernouilliMixture, PoissonRFS,
                     SingleTargetHypothesis, Track)


class PMBM:
    """
    1. Prediction of Bernoulli component.
    2. Misdetection update of Bernoulli component.
    3. Object detection update of Bernoulli component.
    4. Prediction of Poisson Point Process (PPP).
    5. Misdetection update of PPP.
    6. Object detection update of PPP.
    7. PMBM prediction.
    8. PMBM update.
    9. Object states extraction.
    """
    def __init__(
        self,
        meas_model: MeasurementModel,
        sensor_model: SensorModelConfig,
        motion_model: MotionModel,
        birth_model: GaussianMixture,
        merging_threshold,
        max_number_of_hypotheses: int,
        gating_percentage : float,
        w_min,
        detection_probability,
        survival_probability: float,
        density: GaussianDensity = GaussianDensity,
        *args,
        **kwargs,
    ):
        self.timestep = 0
        self.density = density
        self.meas_model = meas_model
        self.sensor_model = sensor_model
        self.motion_model = motion_model
        self.birth_model = birth_model
        self.survival_probability = survival_probability  # models death of an object (aka P_S)

        # Interval for ellipsoidal gating (aka P_G)
        self.gating_percentage = gating_percentage
        self.gating_size = scipy.stats.chi2.ppf(self.gating_percentage,
                                                df=self.meas_model.d)
        self.w_min = w_min  # in log domain
        self.detection_probability = detection_probability  # models detection (and missdetection) of an object (aka P_)
        self.merging_threshold = merging_threshold
        self.desired_num_global_hypotheses = 10
        self.max_number_of_hypotheses = max_number_of_hypotheses

        self.PPP = PoissonRFS(initial_intensity=copy.deepcopy(birth_model))
        self.MBM = MultiBernouilliMixture()

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(current_timestep={self.timestep}, "
            f"MBM components ={len(self.MBM.tracks)}, "
            f"Global hypotheses={len(self.MBM.global_hypotheses)}, "
            f"PPP components={len(self.PPP.intensity)}, ")

    def increment_timestep(self):
        self.timestep += 1

    def predict(
        self,
        birth_model: GaussianMixture,
        motion_model: MotionModel,
        survival_probability: float,
        dt: float,
        density: GaussianDensity,
    ) -> None:
        """Performs PMBM preidction step"""

        # MBM predict
        self.MBM.predict(motion_model, survival_probability, density, dt)
        # PPP predict
        self.PPP.predict(motion_model, copy.deepcopy(birth_model), survival_probability, dt)

    def create_cost_for_associated_targets(self,
                                           global_hypothesis: GlobalHypothesis,
                                           z: np.ndarray) -> np.ndarray:
        """Creates cost matrix for associated measurements for one global"""
        # get data associated weigths
        # cost matrix with shape - number of cosidering measurements (gated) x number of hypothesis trees
        L_d = np.full((len(z), len(global_hypothesis.hypothesis)), np.inf)

        for column_idx, (track_idx, parent_sth_idx) in enumerate(
                global_hypothesis.hypothesis):
            parent_sth = self.MBM.tracks[track_idx].single_target_hypotheses[
                parent_sth_idx]

            children_meas_vector = np.full((len(z)), np.inf)
            for meas_idx, sth in parent_sth.children.items():
                if meas_idx == -1:
                    continue
                children_meas_vector[meas_idx] = sth.cost
            L_d[:, column_idx] = children_meas_vector
        return L_d

    def create_cost_for_undetected(self, z: np.ndarray,
                                   new_tracks: List[Track]) -> np.ndarray:
        # Using association between measurements and previously undetected objects

        L_u = np.full((len(z), len(z)), np.inf)
        assert len(z) == len(new_tracks)

        sth_idx = 0  # we have olny one sth for new targets
        for meas_idx in range(len(z)):
            L_u[meas_idx, meas_idx] = (
                new_tracks[meas_idx].single_target_hypotheses[sth_idx].cost)
        return L_u

    def costruct_new_global_hypothesis(self,
                                       global_hypothesis: GlobalHypothesis,
                                       z: np.ndarray,
                                       new_tracks) -> List[GlobalHypothesis]:
        new_global_hypotheses = []
        L_d = self.create_cost_for_associated_targets(global_hypothesis, z)
        L_u = self.create_cost_for_undetected(z, new_tracks)
        L = np.hstack([L_d, L_u])  # cost matrix

        for each_column_idx in range(L_d.shape[0]):
            if (L_d[each_column_idx] == np.inf).all():
                logging.debug(
                    'it seems we have new measurements which do not stack with our MBM'
                )
        logging.debug(f'\n Considered global hypotheses {global_hypothesis}')
        logging.debug(f'\n cost matrix = \n{L}')
        logging.debug(f'\n detected part = \n{L_d}')
        logging.debug(f'\n undetected part = \n{L_u}')
        logging.debug(f'\n measurements =\n {z}')
        murty_solver = Murty(copy.deepcopy(L))
        max_k = 5
        # max_k = int(
        #     np.ceil(
        #         self.desired_num_global_hypotheses * np.exp(global_hypothesis.weight)
        #     )
        # )

        for k in range(max_k):
            try:
                ok, cost, column_for_meas = murty_solver.draw()
                if not column_for_meas.tolist():
                    break
                logging.debug(
                    f'\n Assignment solution  = {column_for_meas} Cost = {cost}'
                )
            except:
                import pdb
                pdb.set_trace()
            if not ok:
                # logging.info("Break! Murty solver is over!")
                break
            if cost > 10000:
                break

            num_of_current_targets = len(global_hypothesis.hypothesis)
            hypotheses = []
            for measurement_row, target_column in enumerate(
                    column_for_meas.tolist()):
                if target_column + 1 > num_of_current_targets:
                    # the target of this measurements is assigned
                    # not in the current global hypothesis but to is a newly created target
                    track_id = new_tracks[
                        target_column -
                        num_of_current_targets].track_id  # index of bernoulli from PPP birhted bernoulli
                    sth_id = 0  # new target - one leaf
                else:
                    # the target of this measurement os assignes to is a target previously detected
                    track_id, parent_sth_id = global_hypothesis.hypothesis[
                        target_column]
                    parent_sth = self.MBM.tracks[
                        track_id].single_target_hypotheses[parent_sth_id]
                    child_sth = parent_sth.children[measurement_row]
                    sth_id = self.MBM.tracks[
                        track_id].sth_id_generator.__next__()
                    self.MBM.tracks[track_id].single_target_hypotheses.update(
                        {sth_id: copy.deepcopy(child_sth)})
                hypotheses.append((track_id, sth_id))

            log_weight = global_hypothesis.weight - cost
            new_global_hypothesis = GlobalHypothesis(
                weight=log_weight, hypothesis=tuple(hypotheses))
            new_global_hypotheses.append(new_global_hypothesis)
            logging.debug(f'new global hypo: {new_global_hypothesis}')
        assert isinstance(new_global_hypotheses, List)
        return new_global_hypotheses

    def update(self, z):
        """PMBM update.
        1. Perform ellipsoidal gating for each Bernoulli state density and
        each mixture component in the PPP intensity.

        2. Bernoulli update. For each Bernoulli state density,
        create a misdetection hypothesis (Bernoulli component), and
        m object detection hypothesis (Bernoulli component),
        where m is the number of detections inside the ellipsoidal gate of the given state density.

        3. Update PPP with detections.
        Note that for detections that are not inside the gate of undetected objects,
        create dummy Bernoulli components with existence probability r = 0;
        in this case, the corresponding likelihood is simply the clutter intensity.

        4. For each global hypothesis,
        construct the corresponding cost matrix and use Murty's algorithm to obtain
        the M best global hypothesis with highest weights.
        Note that for detections that are only inside the gate of undetected objects, they do not need to be taken into account when forming the cost matrix.

        5. Update PPP intensity with misdetection.

        6. Update the global hypothesis look-up table.

        7. Prune global hypotheses with small weights and cap the number. (Reduction step ?)

        8. Prune local hypotheses (or hypothesis trees) that do not appear in the maintained global hypotheses, and re-index the global hypothesis look-up table. (Reduction step ?)

         Parameters
         ----------
         z : [type]
             [description]
        """

        # 1.1 Perform ellipsoidal gating for each  mixture component in the PPP intensity
        gating_matrix_undetected, used_meas_undetected = self.PPP.gating(
            z, self.density, self.meas_model, self.gating_size)

        # 1.2 Perform ellipsoidal gating for each Bernoulli state density
        gating_matrix_detected, used_meas_detected = self.MBM.gating(
            z, self.density, self.meas_model, self.gating_size)

        # 2. Bernoulli update. For each Bernoulli state density,
        # create a misdetection hypothesis (Bernoulli component), and
        # m object detection hypothesis (Bernoulli component),
        # where m is the number of detections inside the ellipsoidal gate of
        # the given state density
        self.MBM.update(self.detection_probability, z, gating_matrix_detected, self.meas_model)

        # 3. Update PPP with detections.
        # Update of potential new object detected for the first time -> new Bern
        # Note that for detections that are not inside the gate of undetected objects,
        # create dummy Bernoulli components with existence probability r = 0;
        # in this case, the corresponding likelihood is simply the clutter intensity.
        # Each measurement creates hypothessis tree (new track).

        new_tracks = self.PPP.get_targets_detected_for_first_time(
            z,
            gating_matrix_undetected,
            self.sensor_model.lambda_c,
            self.meas_model,
            self.detection_probability,
        )
        logging.debug(
            f'\n===============current timestep: {self.timestep}==============='
        )
        logging.debug(f'\n new tracks from PPP {new_tracks}')
        logging.debug(
            f'\n current global hypotheses {self.MBM.global_hypotheses}')

        # 4. Update global hypothesis
        # 4.1 If list of global hypothesis is empty, creates the one.
        if self.timestep == 1:
            for new_track in new_tracks:
                self.MBM.add_track(new_track)

            hypo_list = []
            for track_id, track in self.MBM.tracks.items():
                for sth_id, sth in track.single_target_hypotheses.items():
                    assert sth_id == 0
                    hypo_list.append((track_id, sth_id))
            self.MBM.global_hypotheses.append(
                GlobalHypothesis(weight=0.0, hypothesis=tuple(hypo_list)))

        # 4.2 Otherwise, for each global hypothesis construct cost matrix,
        # solve linear programming problem and construct new k global hypothesis for each.
        else:
            self.MBM.normalize_global_hypotheses_weights()
            new_global_hypotheses = []
            for global_hypothesis in self.MBM.global_hypotheses:
                new_global_hypotheses_step = self.costruct_new_global_hypothesis(
                    global_hypothesis, z, new_tracks)
                for _ in new_global_hypotheses_step:
                    new_global_hypotheses.append(_)
            self.MBM.global_hypotheses.clear()
            self.MBM.global_hypotheses = new_global_hypotheses
            for track_id, track in self.MBM.tracks.items():
                self.MBM.tracks[track_id].cut_tree()

            for new_track in new_tracks:
                self.MBM.add_track(new_track)

        # Update of PPP intensity for undetected objects that remain undetected
        self.PPP.undetected_update(self.detection_probability)

    def estimator(self):
        estimates = self.MBM.estimator()
        logging.debug(f'estimates = {estimates}')
        return estimates

    def reduction(self) -> None:
        self.PPP.prune(threshold=-10)
        # self.MBM.prune_global_hypotheses(threshold=np.log(0.001))
        self.MBM.cap_global_hypothesis(self.max_number_of_hypotheses)
        # self.MBM.remove_unused_bernoullies()

    def estimation_step(self, current_measurements: np.ndarray):
        # PMBM prediction
        self.PMBM_predict()

        # PMBM update
        self.update(current_measurements)

        # Extract state estimates from the PPP
        estimates = self.PMBM_estimator()

        return estimates
