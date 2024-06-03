from typing import Callable, Union


class Repeat:
    """
    Class to control the repetition of an operation based on count or condition.

    This class allows repeating an operation a specific number of times or until a condition is met.

    Attributes:
        index (int): The current index of repetition.

    Args:
        repeat (int): Number of times to repeat the operation. Defaults to 0.
        repeat_cond (Callable[[], bool]): A callable condition to determine whether to repeat the operation. Defaults to None.
        init_cond (Union[bool, Callable[[], bool]]): A flag or callable condition to determine the initial repetition check. Defaults to None.

    Examples:
        >>> # Repeat 3 times
        >>> repeat = Repeat(repeat=3)
        >>> [bool(repeat) for _ in range(5)]
        [True, True, True, False, False]

        >>> # Repeat while a condition is True
        >>> condition = lambda: False
        >>> repeat = Repeat(repeat_cond=condition)
        >>> [bool(repeat) for _ in range(3)]
        [False, False, False]

        >>> # Repeat 2 times and then while a condition is True
        >>> condition = lambda: True
        >>> repeat = Repeat(repeat=2, repeat_cond=condition)
        >>> [bool(repeat) for _ in range(5)]
        [True, True, False, False, False]

        >>> # Skip the initial check and repeat while a condition is True
        >>> condition = lambda: False
        >>> repeat = Repeat(repeat_cond=condition, init_cond=True)
        >>> [bool(repeat) for _ in range(3)]
        [True, False, False]

        >>> # Repeat based on a condition that switches from True to False
        >>> condition_list = [True, True, True, False]
        >>> repeat_cond_list = lambda: condition_list.pop(0)
        >>> repeat = Repeat(repeat_cond=repeat_cond_list)
        >>> [bool(repeat) for _ in range(4)]
        [True, True, True, False]

        >>> # Repeat with an initial condition that is True
        >>> init_condition = lambda: True
        >>> repeat = Repeat(init_cond=init_condition)
        >>> [bool(repeat) for _ in range(2)]
        [True, False]

        >>> # Repeat with an initial condition that is False
        >>> init_condition = lambda: False
        >>> repeat = Repeat(repeat=3, init_cond=init_condition)
        >>> [bool(repeat) for _ in range(2)]
        [False, False]
    """

    def __init__(self, repeat: int = 0, repeat_cond: Callable[[], bool] = None, init_cond: Union[bool, Callable[[], bool]] = None):
        """
        Initializes the Repeat class.

        Args:
            repeat (int): Number of times to repeat the operation. Defaults to 0.
            repeat_cond (Callable[[], bool]): A callable condition to determine whether to repeat the operation. Defaults to None.
            init_cond (Union[bool, Callable[[], bool]]): A flag or callable condition to determine the initial repetition check. Defaults to None.
        """
        self._repeat = repeat
        self._repeat_cond = repeat_cond
        self._init_cond = init_cond
        self.index = 0

    def __bool__(self):
        """
        Determines whether to continue repeating based on the index, repeat count, and conditions.

        Returns:
            bool: True if the operation should be repeated, False otherwise.
        """
        if self.index == 0 and self._init_cond is not None:
            if not (self._init_cond is True or self._init_cond()):
                return False
        else:
            if self._repeat_cond is None:
                if not (self.index < self._repeat):
                    return False
            else:
                if self._repeat > 0:
                    if not (
                            (self.index < self._repeat)
                            and self._repeat_cond()
                    ):
                        return False
                else:
                    if not self._repeat_cond():
                        return False

        self.index += 1
        return True
