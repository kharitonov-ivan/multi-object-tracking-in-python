import logging
from collections import UserDict
from dataclasses import dataclass

import numpy as np
import pytest
import tqdm
import ujson
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.nuscenes import NuScenes

import mot.motion_models
from mot.common import Gaussian, GaussianMixture, WeightedGaussian
from mot.configs import SensorModelConfig
from mot.trackers.multiple_object_trackers.PMBM.common.birth_model import (
    StaticBirthModel,
)
from mot.trackers.multiple_object_trackers.PMBM.pmbm import PMBM

from .evaluator import OneSceneMOTevaluator


@dataclass
class NuscenesDatasetConfig:
    data_path: str  # path to data
    version: str  # datasetVersion


# @dataclass
# class ObjectGroundTruth:
#     center_translation:
#     center_roatation
#     velocity:
#     object_class:
#     size:


class NuscenceInstanceTokenReidentification(UserDict):
    def __init__(self):
        self.max_id = 0
        self.data = {}

    def add_id(self, instance_token):
        if instance_token not in self.data.keys():
            self.data[instance_token] = self.max_id
            self.max_id += 1


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
        logging.debug(f"Process scene with name {scene_object['name']}")
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
            max_number_of_hypotheses=3,
            gating_percentage=1.0,
            detection_probability=detection_probability,
            survival_probability=0.9,
            existense_probability_threshold=0.7,
            density=mot.common.GaussianDensity,
        )
        scene_estimations = []
        scene_measurements = []
        gt = []
        id_pool = NuscenceInstanceTokenReidentification()

        logging.info(f"Tracking {scene_token} scene")
        evaluator = OneSceneMOTevaluator()

        for timestep, sample_token in tqdm.tqdm(enumerate(scene_sample_tokens[:10])):
            measurements = self.get_measurements_for_one_sample(token=sample_token)
            logging.info(f"tracker condition: {tracker}")
            logging.info(f"got {len(measurements)} measurements")

            current_sample_data = self.nuscenes_helper.get("sample", sample_token)
            lidar_top_data_token = current_sample_data["data"]["LIDAR_TOP"]
            lidar_data = self.nuscenes_helper.get("sample_data", lidar_top_data_token)
            lidar_top_ego_pose = self.nuscenes_helper.get("ego_pose", lidar_data["ego_pose_token"])
            current_ego_pose = lidar_top_ego_pose
            estimation = tracker.step(measurements, dt, current_ego_pose)

            scene_measurements.append(measurements)
            scene_estimations.append(estimation)
            annotations = [
                self.nuscenes_helper.get("sample_annotation", annotation_token)
                for annotation_token in current_sample_data["anns"]
            ]

            scene_data = {}
            for annotation in annotations:
                id_pool.add_id(annotation["instance_token"])
                object_id = id_pool.data[annotation["instance_token"]]
                object_pos_x, object_pos_y = annotation["translation"][:2]
                scene_data[object_id] = Gaussian(x=np.array([object_pos_x, object_pos_y, 0.0, 0.0]), P=np.eye(4))

            evaluator.step(
                sample_measurements=measurements, sample_estimates=estimation, sample_gt=scene_data, timestep=timestep
            )

            gt.append(annotations)
            if not annotations:
                logging.info("annotation empty")

        evaluator.post_processing()
        # meta = f"scene_token={scene_token}"
        # plt.savefig(get_images_dir(__file__) + "/" + "results_" + meta + ".png")
        return scene_estimations

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
            try:
                scene_token = self.nuscenes_helper.get(table_name="sample", token=sample_token)["scene_token"]
                scene_tokens.add(scene_token)
            except KeyError:
                logging.debug("This token is not exist!")
                continue
        logging.info(f"Tokens: {scene_tokens}")
        # TODO убрать костыль
        scene_tokens = set(["fcbccedd61424f1b85dcbf8f897f9754"])
        return scene_tokens

    def read_detection_file(self, detection_filespath: str):
        print("Reading detection file")
        with open(detection_filespath) as file:
            detection_data = ujson.load(file)
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


@pytest.fixture
def nuscenes_config():
    yield NuscenesDatasetConfig(
        data_path="/Users/a18677982/repos/Multi-Object-Tracking-for-Automotive-Systems-in-python/data/nuscenes/dataset/v1.0-mini",  # noqa
        version="v1.0-mini",
    )


def test_nuscenes(nuscenes_config):
    evaluator = NuscenesTrackerEvaluator(
        detection_filepath="/Users/a18677982/repos/Multi-Object-Tracking-for-Automotive-Systems-in-python/data/nuscenes/detection-megvii/megvii_val.json",  # noqa
        nuscens_config=nuscenes_config,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    test_nuscenes()
