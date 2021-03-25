import numpy as np
import pytest
import unittest
from mot.configs import Object
from mot.common.state import State

TOL = 1e-4


class Test_ObjectConfig(unittest.TestCase):
    def test_create_object_config(self):
        state = State(x=np.array([0, 0]), P=np.eye(2))
        object_config = Object(initial=state, t_birth=0, t_death=10)
        print(object_config)

    def test_birth_after_death(self):
        state = State(x=np.array([0, 0]), P=np.eye(2))
        with pytest.raises(Exception):
            object_config = Object(initial=state, t_birth=10, t_death=0)
            print(object_config)
