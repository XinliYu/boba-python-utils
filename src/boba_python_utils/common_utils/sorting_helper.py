from enum import Enum
from functools import partial
from typing import Union, Iterable, Callable, Optional
from boba_python_utils.common_utils.iter_helper import unzip
from boba_python_utils.common_utils.map_helper import get_
from boba_python_utils.common_utils.typing_helper import iterable


class SortOptions(str, Enum):
    NoSorting = 'none',
    Descending = 'desc'
    Ascending = 'asc'


def _get_sort_key(key: Union[str, Iterable[str], Callable]) -> Optional[Callable]:
    """
    Internal function for solving sorting key.

    When the sorting key is string, or a sequence of strings,
    then we assume the objects to sort contain fields/values associated with the specified string
    keys, and we construct a callable to use in the `sorted` function as the sorting key.

    If `key` is None, then None is returned.

    """
    if key is None:
        return None
    elif isinstance(key, str):
        return partial(get_, key1=key)
    elif iterable(key) and isinstance(next(iter(key)), str):
        return lambda x: tuple(get_(x, _key) for _key in key)
    elif callable(key):
        return key
    else:
        raise ValueError("the sorting 'key' must be a callable, "
                         "or a string, "
                         "or a list/tuple of strings; "
                         f"got '{key}'")


def sorted_(
        _iterable,
        key: Union[str, Iterable[str], Callable] = None,
        reverse: bool = False,
        no_sort_if_key_is_none: bool = False,
        sort_option: Union[str, SortOptions] = None
):
    """
    Performs the same sorting operation as the system function `sorted`,
    but provides additional options to tweak some minor behaviors.

    This function also supports using string(s) as strong key(s).

    When `key` is a string, or a list/tuple of strings,
    then we assume the objects to sort contain fields/values associated with the specified string
    keys, and we construct a callable to use in the `sorted` function as the sorting key.

    Args:
        _iterable: the iterable to sort.
        key: the sorting key; can be a string, a sequence of strings, or a callable.
        reverse: True to sort the iterable in the descending order.
        no_sort_if_key_is_none: True to perform no sorting if `key` is set None, instead of
            the default behavior of sorting in the ascending order.
        sort_option: one convenient option to control whether we perform no soring,
            or sort in the ascending order, or sort in the descending order; 
            the purpose is to use one argument to control this frequently used behavior option 
            instead of using two arguments from `reverse` and `no_sort_if_key_is_none`.
            If specified, this parameter has higher priority.

    Returns: a new list containing all items from the iterable in a sorted order.

    Examples:
        >>> sorted_([4, 5, 2, 1, 3])
        [1, 2, 3, 4, 5]
        >>> sorted_([4, 5, 2, 1, 3], reverse=True)
        [5, 4, 3, 2, 1]
        >>> sorted_([4, 5, 2, 1, 3], sort_option=SortOptions.Descending)
        [5, 4, 3, 2, 1]
        >>> sorted_([4, 5, 2, 1, 3], no_sort_if_key_is_none=True)
        [4, 5, 2, 1, 3]
        >>> sorted_([4, 5, 2, 1, 3], sort_option=SortOptions.NoSorting)
        [4, 5, 2, 1, 3]
        >>> sorted_(
        ...     [{'k': 2, 'v': 'b'}, {'k': 1 , 'v': 'c'}, {'k': 3 , 'v': 'a'}],
        ...     key='k'
        ... )
        [{'k': 1, 'v': 'c'}, {'k': 2, 'v': 'b'}, {'k': 3, 'v': 'a'}]
        >>> sorted_(
        ...     [{'k': 2, 'v': 'b'}, {'k': 1 , 'v': 'c'}, {'k': 3 , 'v': 'a'}],
        ...     key='v',
        ...     reverse=True
        ... )
        [{'k': 1, 'v': 'c'}, {'k': 2, 'v': 'b'}, {'k': 3, 'v': 'a'}]
        >>> sorted_(
        ...     [{'k': 2, 'v': 'b'}, {'k': 1 , 'v': 'c'}, {'k': 3 , 'v': 'a'}],
        ...     key=('v', 'a')
        ... )
        [{'k': 3, 'v': 'a'}, {'k': 2, 'v': 'b'}, {'k': 1, 'v': 'c'}]

    """
    if sort_option is not None:
        if sort_option == SortOptions.NoSorting:
            if reverse:
                raise ValueError("'reverse is set True' but 'sort_option' "
                                 "asks for no sorting")
            return _iterable
        elif sort_option == SortOptions.Ascending:
            if reverse:
                raise ValueError("'reverse is set True' but 'sort_option' "
                                 "asks for soring in the ascending order")
            reverse = False
        elif sort_option == SortOptions.Descending:
            reverse = True

    if key is None:
        if no_sort_if_key_is_none:
            return _iterable
        else:
            return sorted(_iterable, reverse=reverse)
    else:
        key = _get_sort_key(key)
        return sorted(_iterable, key=_get_sort_key(key), reverse=reverse)


def iter_sorted_(_iterable, key, reverse=False, element_transform=None, sort_before_transform=True, no_sort_if_key_is_none=False):
    if element_transform is None:
        return sorted_(_iterable, key=key, reverse=reverse, no_sort_if_key_is_none=no_sort_if_key_is_none)
    else:
        if sort_before_transform:
            return map(element_transform, sorted_(_iterable, key=key, reverse=reverse, no_sort_if_key_is_none=no_sort_if_key_is_none))
        else:
            return sorted_(map(element_transform, _iterable), key=key, reverse=reverse, no_sort_if_key_is_none=no_sort_if_key_is_none)


def sorted__(_iterable, key, reverse: bool = False, return_tuple=False, return_indexes=False):
    """
    An enhanced alternative for the build-in `sorted` function.
    1) Allows the `key` be a sequence of values (as the sorting keys) for the `_iterable`.
    2) Allows returning items indexes in the original sequence along with the sorted items
        by setting `return_indexes` as True.
    3) Allows returning a tuple instead of a list for convenience
        by setting `return_tuple` as True.

    >>> class A:
    ...    def __init__(self, x):
    ...        self._x = x
    ...
    ...    def __repr__(self):
    ...        return str(self._x)

    >>> sorted__(list(map(A, range(10))), key=(x % 2 == 0 for x in range(10)))
    [1, 3, 5, 7, 9, 0, 2, 4, 6, 8]

    :param _iterable: a sequence of objects to sort.
    :param key: the sorting key; can be a function such like the `key` parameter for the build-in sorted function, a sequence of values as the sorting keys.
    :param reverse: `True` to sort descendingly; `False` to sort ascendingly.
    :param return_tuple: `True` to return a tuple; `False` to return a list.
    :return: a list or a tuple of sorted values from the `_iterable`.
    """

    if return_indexes is True:
        if callable(key):
            _key = lambda x: key(x[0])
        else:
            _key = key
        return sorted__(
            ((x, i) for i, x in enumerate(_iterable)),
            key=_key,
            reverse=reverse,
            return_tuple=return_tuple,
            return_indexes=False
        )
    elif return_indexes == 'labels':
        sorted_tups = sorted__(
            ((x, i) for i, x in enumerate(_iterable)),
            key=key,
            reverse=reverse,
            return_tuple=True,
            return_indexes=False
        )
        labels = [0] * len(sorted_tups)
        for j, (x, i) in enumerate(sorted_tups):
            labels[i] = j
        out = ((x, l) for (x, i), l in zip(sorted_tups, labels))
        return tuple(out) if return_tuple else list(out)

    if callable(key):
        s = sorted(_iterable, key=key, reverse=reverse)
        return tuple(s) if return_tuple else s
    else:
        s = unzip(
            unzip(
                sorted(zip(key, enumerate(_iterable)), reverse=reverse),
                1
            ), 1
        )  # `enumerate(_iterable)` ensures the original order of the `_iterable` when keys are the same
        return s if return_tuple else list(s)
