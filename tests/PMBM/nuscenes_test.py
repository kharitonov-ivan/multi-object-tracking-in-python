import json
import logging as log
from dataclasses import dataclass
from os import wait
from typing import Dict, List

import mot
import tqdm
import numpy as np
from mot.common import Gaussian, GaussianMixture, WeightedGaussian
from mot.configs import SensorModelConfig
import mot.motion_models
from mot.trackers.multiple_object_trackers.PMBM.common.birth_model import (
    StaticBirthModel,
)
from mot.trackers.multiple_object_trackers.PMBM.pmbm import PMBM
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.nuscenes import NuScenes
import multiprocessing as mp


@dataclass
class NuscenesDatasetConfig:
    data_path: str  # path to data
    version: str  # datasetVersion


# @dataclass
# class ObjectDetection3D:
#     center_position:
#     velocity:
#     object_class:
#     size:


class NuscenesTrackerEvaluator:
    def __init__(self, detection_filepath: str, nuscens_config: NuscenesDatasetConfig) -> None:
        # with mp.Pool(processes=8) as pool:

        #     # result1 = pool.map_async(self.initialize_nuscenes, [nuscens_config])

        #     # result2 = pool.map_async(self.read_detection_file, [detection_filepath])
        #     # result1 = result1.get()
        #     # result2 = result2.get()
        _, _, self.detection_results = self.read_detection_file(detection_filepath)
        self.nuscenes_helper = self.initialize_nuscenes(nuscens_config)
        self.estimatios = {}

    def evaluate(self):

        scene_tokens = self.get_scenes_from_detections()

        for scene_token in tqdm.tqdm(scene_tokens):
            self.estimatios[scene_token] = self.process_scene(scene_token)

    def process_scene(self, scene_token: str):
        scene_object = self.nuscenes_helper.get(table_name="scene", token=scene_token)
        log.debug(f"Process scene with name {scene_object['name']}")
        first_sample_token = scene_object["first_sample_token"]
        scene_sample_tokens = self.get_scene_token_list(firts_frame_token=first_sample_token)

        detection_probability = 0.9
        dt = 0.5
        birth_model = GaussianMixture(
            [
                WeightedGaussian(
                    np.log(0.03),
                    Gaussian(x=np.array([0.0, 0.0, 0.0, 0.0]), P=100 * np.eye(4)),
                ),
                WeightedGaussian(
                    np.log(0.03),
                    Gaussian(x=np.array([400.0, -600.0, 0.0, 0.0]), P=100 * np.eye(4)),
                ),
                WeightedGaussian(
                    np.log(0.03),
                    Gaussian(x=np.array([-800.0, 200.0, 0.0, 0.0]), P=100 * np.eye(4)),
                ),
                WeightedGaussian(
                    np.log(0.03),
                    Gaussian(x=np.array([-200.0, 800.0, 0.0, 0.0]), P=100 * np.eye(4)),
                ),
            ]
        )
        tracker = PMBM(
            meas_model=mot.measurement_models.ConstantVelocityMeasurementModel(sigma_r=5.0),
            sensor_model=SensorModelConfig(
                P_D=detection_probability,
                lambda_c=10.0,
                range_c=np.array([[-2000, 2000], [-2000, 2000]]),
            ),
            motion_model=mot.motion_models.ConstantVelocityMotionModel(dt, sigma_q=5.0),
            birth_model=StaticBirthModel(birth_model),
            max_number_of_hypotheses=5,
            gating_percentage=1.0,
            detection_probability=detection_probability,
            survival_probability=0.9,
            existense_probability_threshold=0.7,
            density=mot.GaussianDensity,
        )
        estimations = []
        for sample_token in tqdm.tqdm(scene_sample_tokens):
            measurements = self.get_measurements_for_one_sample(token=sample_token)
            estimation = tracker.step(measurements, dt)
            estimations.append(estimation)
        return estimations

    def get_measurements_for_one_sample(self, token):
        detection_results = self.detection_results[token]
        z = []
        for detection in detection_results:
            z.append(detection.translation[:2])  # get only x,y position
        return np.array(z)

    def get_scenes_from_detections(self):
        # get list of scenes from detection file
        scene_tokens = set()

        for sample_token in self.detection_results.sample_tokens:
            scene_token = self.nuscenes_helper.get(table_name="sample", token=sample_token)[
                "scene_token"
            ]
            scene_tokens.add(scene_token)
        return scene_tokens

    def read_detection_file(self, detection_filespath: str):
        print("Reading detection file")
        with open(detection_filespath) as file:
            detection_data = json.load(file)
        metadata = detection_data["meta"]
        results = EvalBoxes.deserialize(detection_data["results"], DetectionBox)
        return detection_data, metadata, results

    def initialize_nuscenes(self, config: NuscenesDatasetConfig):
        nuscenes = NuScenes(version=config.version, dataroot=config.data_path, verbose=True)
        return nuscenes

    def get_scene_token_list(self, firts_frame_token: str):
        samples = self.traverse_linked_list(
            nuscenes=self.nuscenes_helper,
            first_object=self.nuscenes_helper.get("sample", firts_frame_token),
            direction="next",
        )
        frame_tokens = [sample["token"] for sample in samples]
        return tuple(frame_tokens)

    def traverse_linked_list(
        self,
        nuscenes: NuScenes,
        first_object,
        table_hey: str = "sample",
        direction: str = "prev",
        inclusive: bool = False,
    ):
        assert direction in ("prev", "next")

        current_object = first_object

        if not inclusive:
            sequence = []
        else:
            sequence = [current_object]

        while True:
            if len(current_object[direction]):
                current_object = nuscenes.get(table_name=table_hey, token=current_object[direction])
                sequence.append(current_object)
            else:
                break
        if direction == "prev":
            return sequence[::-1]
        else:
            return sequence


def main_test():
    evaluator = NuscenesTrackerEvaluator(
        detection_filepath="/Users/a18677982/repos/Multi-Object-Tracking-for-Automotive-Systems-in-python/data/nuscenes/detection-megvii/megvii_val.json",
        nuscens_config=NuscenesDatasetConfig(
            data_path="/Users/a18677982/repos/Multi-Object-Tracking-for-Automotive-Systems-in-python/data/nuscenes/dataset",
            version="v1.0-trainval",
        ),
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main_test()
