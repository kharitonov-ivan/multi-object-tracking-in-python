import unittest
from unittest.main import main

import numpy as np
import pytest
from mot.configs import GroundTruthConfig, Object
from mot.common.state import State

TOL = 1e-4


class Test_GroundTruthConfig(unittest.TestCase):
    def test_ground_truth_config_single_object(self):
        test_n_birth = 1
        test_total_time = 10
        test_objects = [
            Object(
                initial=State(x=np.array([0.0, 1.0]), P=np.eye(2)),
                t_birth=0,
                t_death=10,
            )
        ]

        got_ground_truth_config = GroundTruthConfig(
            n_births=test_n_birth,
            object_configs=test_objects,
            total_time=test_total_time,
        )
        print(got_ground_truth_config)

    def test_ground_truth_config_type(self):
        test_n_birth = "str"
        test_total_time = 10
        test_objects = [
            Object(
                initial=State(x=np.array([0.0, 1.0]), P=np.eye(2)),
                t_birth=0,
                t_death=10,
            )
        ]
        with pytest.raises(Exception):
            got_ground_truth_config = GroundTruthConfig(
                n_births=test_n_birth,
                object_configs=test_objects,
                total_time=test_total_time,
            )
            print(got_ground_truth_config)

