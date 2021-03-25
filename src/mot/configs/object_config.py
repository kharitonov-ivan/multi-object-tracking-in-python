import itertools

from mot.common.state import Gaussian


class Object:
    new_id = itertools.count(start=0, step=1)

    @staticmethod
    def restart(cls):
        cls.new_id = itertools.count(start=0, step=1)

    def __init__(self, initial: Gaussian, t_birth: int, t_death: int, *args, **kwargs):
        """Construct object configuration for ground truth config

        Parameters
        ----------
        initial_state : Gaussian
            objects initial state
        t_birth : int
             object birth (appearing time)
        t_death : int
            the last time the object exists

        Attributes
        ----------
        id : int
            unique id of object
        """
        assert isinstance(initial, Gaussian), "Argument of wrong type!"
        assert isinstance(t_birth, int), "Argument of wrong type!"
        assert t_birth >= 0, "object birth should be positive!"
        assert isinstance(t_death, int), "Argument of wrong type!"
        assert t_death >= 0, "object death should be positive!"
        assert t_death >= t_birth, "object death should be greater or equal than birth!"

        self.id = next(Object.new_id)
        self.initial_state = initial
        self.t_birth = t_birth
        self.t_death = t_death

    def __repr__(self) -> str:
        return self.__class__.__name__ + (
            f"(id={self.id}, "
            f"initial_state={self.initial_state}, "
            f"t_birth={self.t_birth}, "
            f"t_death={self.t_death}, "
        )
