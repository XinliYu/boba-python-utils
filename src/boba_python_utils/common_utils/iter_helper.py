import warnings
from itertools import zip_longest, chain, product
from typing import Iterator, Union, Tuple, List, Type, Iterable, Callable, Optional, Mapping, Sequence
import random

from boba_python_utils.common_utils.array_helper import split_list
from boba_python_utils.common_utils.typing_helper import iterable__, sliceable


def tqdm_wrap(
        _it: Union[Iterable, Iterator],
        use_tqdm: bool,
        tqdm_msg: str = None,
        verbose: bool = __debug__
) -> Union[Iterator, Iterable]:
    """
    Wraps an iterator/iterable in a tadm object to display iteration progress.
    Args:
        _it: the iterator/iterable.
        use_tqdm: True to enable the tqdm wrap.
        tqdm_msg: the tqdm description to display along with the progress.
        verbose: if tqdm package somehow fails to load, and this argument is set True,
            then `tqdm_msg` will still be printed.

    Returns: a tqdm wrap of the input iterator/iterable if the tqdm package loads successfully;
        otherwise the input iterator/iterable.

    """
    if isinstance(use_tqdm, str):
        tqdm_msg = use_tqdm
        use_tqdm = bool(use_tqdm)

    try:
        from tqdm import tqdm
    except Exception as error:
        warnings.warn(f"unable to load tqdm package; got error '{error}'")
        tqdm = None
        use_tqdm = False

    if use_tqdm:
        _it = tqdm(_it)
        if tqdm_msg:
            _it.set_description(tqdm_msg)
    elif verbose and tqdm_msg is not None:
        print(tqdm_msg)
    return _it


def _get_non_atom_types(
        non_atom_types: Union[Tuple[Type, ...], List[Type], Type],
        always_consider_iterator_as_non_atom: bool = True
) -> Union[Iterable[Type], Type]:
    """
    Internal function. Used by `iter_`.

    Adds `Iterator` type to the type list `non_atom_types`
    when `always_consider_iterator_as_non_atom` is set True.

    If `always_consider_iterator_as_non_atom` is set False,
    this function takes no action, and `non_atom_types` is returned.

    Examples:
        >>> _get_non_atom_types(
        ...     non_atom_types=(list, tuple),
        ...     always_consider_iterator_as_non_atom=True
        ... )
        (typing.Iterator, <class 'list'>, <class 'tuple'>)

        >>> _get_non_atom_types(
        ...     non_atom_types=(list, tuple),
        ...     always_consider_iterator_as_non_atom=False
        ... )
        (<class 'list'>, <class 'tuple'>)

    """
    if always_consider_iterator_as_non_atom:
        if isinstance(non_atom_types, (list, tuple)):
            if Iterator not in non_atom_types:
                return (Iterator, *non_atom_types)
        elif non_atom_types is not Iterator:
            return Iterator, non_atom_types
    return non_atom_types


# region get iterable information
def len_(x, non_atom_types=(list, tuple)):
    return len(x) if isinstance(x, non_atom_types) else 1


def len__(x, atom_types=(str,)):
    return 1 if isinstance(x, atom_types) or (not hasattr(x, '__len__')) else len(x)


def is_homogeneous_iterable(items: Iterable) -> bool:
    """
    Determines if all elements in an iterable are of the same type.

    Args:
        items (Iterable): An iterable (list, tuple, set, generator, etc.) of elements.

    Returns:
        bool: True if all elements are of the same type, or the iterable is empty; False otherwise.

    Examples:
        >>> is_homogeneous_iterable([1, 2, 3])
        True
        >>> is_homogeneous_iterable([1, '2', 3])
        False
        >>> is_homogeneous_iterable(iter([1.0, 2.0, 3.0]))
        True
        >>> is_homogeneous_iterable([])
        True
        >>> is_homogeneous_iterable(['hello', 'world', 'test'])
        True
        >>> is_homogeneous_iterable(iter(['hello', 'world', 3]))
        False
        >>> is_homogeneous_iterable(iter([1, 2, 3, 4.5]))
        False
        >>> is_homogeneous_iterable(iter([{'a': 1}, {'b': 2}]))
        True
        >>> is_homogeneous_iterable(iter([1, 2, [3]]))
        False
    """
    _iter = iter(items)
    try:
        first_type = type(next(_iter))
    except StopIteration:
        return True
    return all(isinstance(item, first_type) for item in _iter)


# endregion

def tuple_(
        x,
        size_or_default_values: Union[int, Sequence] = None,
        non_atom_types=(list, tuple),
        cutoff: bool = False
):
    """
    Convert the input into a tuple with a specified size or default values.

    Args:
        x: The input object to be converted into a tuple.
        size_or_default_values (Union[int, Sequence], optional):
            Either an integer specifying the desired size of the tuple or a sequence of default values
            to fill the tuple if its length is less than the length of the defaults. Defaults to None.
        non_atom_types (tuple, optional):
            Tuple of non-atomic types. Defaults to (list, tuple).
        cutoff (bool, optional):
            If True, truncates the input tuple to match the specified size or default values.
            If False, raises a ValueError if the input tuple is longer than the specified size. Defaults to False.

    Returns:
        tuple: The converted tuple.

    Raises:
        ValueError:
            If the input does not match the expected size or default values,
            or if 'size_or_default_values' is not of type int, Sequence, or None.

    Examples:
        >>> tuple_("hello")
        ('hello',)

        >>> tuple_([1, 2, 3], 5)
        (1, 2, 3, None, None)

        >>> tuple_((1, 2, 3, 4), 2)
        Traceback (most recent call last):
            ...
        ValueError: expected maximum tuple length 2; got 4

        >>> tuple_((1, 2, 3, 4), 2, cutoff=True)
        (1, 2)

        >>> tuple_((1, 2), (3, 4, 5))
        (1, 2, 5)
    """
    if isinstance(size_or_default_values, int):
        if not isinstance(x, non_atom_types):
            return x, *([None] * (size_or_default_values - 1))

        if not isinstance(x, tuple):
            x = tuple(x)
        if len(x) == size_or_default_values:
            return x
        elif len(x) > size_or_default_values:
            if cutoff:
                return x[:size_or_default_values]
            else:
                raise ValueError(f'expected maximum tuple length {size_or_default_values}; got {len(x)}')
        else:
            return x + (None,) * (size_or_default_values - len(x))
    elif size_or_default_values is None:
        if not isinstance(x, non_atom_types):
            return x,
        elif not isinstance(x, tuple):
            return tuple(x)
        else:
            return x
    elif isinstance(size_or_default_values, Sequence):
        len_defaults = len(size_or_default_values)
        if len(x) == len_defaults:
            return x
        elif len(x) > len_defaults:
            if cutoff:
                return x[:size_or_default_values]
            else:
                raise ValueError(f'expected maximum tuple length {size_or_default_values}; got {len(x)}')
        else:
            return x + tuple(size_or_default_values[len(x):])
    else:
        raise ValueError(f"'defaults' can only be an integer, a Sequence, or None, got '{size_or_default_values}'")


def iter_(
        _it,
        non_atom_types=(list, tuple),
        infinitely_yield_atom: bool = False,
        iter_none: bool = False,
        always_consider_iterator_as_non_atom: bool = True
) -> Iterator:
    """
    Get an iterator for an iterable object, whose type must be one of `non_atom_types`;
    otherwise the function yields the object itself.

    If `always_consider_iterator_as_non_atom` is set True,
    then an object of :type:`Iterator` will always be considered as a non-atomic iterable type,
    even it is not specified in `non_atom_types`.

    In many use cases, an python iterable type, such as a string,
    cannot be treated as an iterable.

    Args:
        _it: the object to iterate through; it will only be considered as an iterable
            if its type is one of the types specified by `non_atom_types`,
            or if it is an :type:`Iterator` object and
            `always_consider_iterator_as_non_atom` is set True.
        non_atom_types: constraints the types that are considered as being iterable;
            all other types are not considered as being iterable,
            except for an :type:`Iterator` object
            if `always_consider_iterator_as_non_atom` is set True.
        infinitely_yield_atom: if `_it` is an atom, then yield it infinitely;
            this option is useful when combining `iter_` with `zip`.
        iter_none: True to still yield `_it` if it is None.
        always_consider_iterator_as_non_atom: True to always consider an :type:`Iterator` object
            as an iterable, regardless of `non_atom_types`.

    Returns: an iterator iterating through `_it` if it is iterable and whose type is
        one of `non_atom_types`, or an :type:`Iterator` if `always_consider_iterator_as_non_atom`
        is set True; otherwise an iterator only yielding `_it` itself.

    Examples:
        >>> list(iter_(0))
        [0]
        >>> list(iter_('123'))
        ['123']
        >>> list(iter_('123', non_atom_types=(str, )))
        ['1', '2', '3']
        >>> list(iter_(None))
        []
        >>> list(iter_(None, iter_none=True))
        [None]
        >>> list(iter_((1, 2, None)))
        [1, 2, None]
        >>> list(iter_({1, 2, None}))
        [{1, 2, None}]
        >>> list(zip([1,2,3], iter_(0, infinitely_yield_atom=True)))
        [(1, 0), (2, 0), (3, 0)]
    """
    non_atom_types = _get_non_atom_types(
        non_atom_types,
        always_consider_iterator_as_non_atom
    )
    if _it is not None:
        if isinstance(_it, non_atom_types):
            if infinitely_yield_atom:
                for x in _it:
                    yield x
                while True:
                    yield x
            else:
                yield from _it
        elif infinitely_yield_atom:
            while True:
                yield _it
        else:
            yield _it
    elif iter_none:
        if infinitely_yield_atom:
            while True:
                yield _it
        else:
            yield _it


def iter__(
        _it,
        atom_types=(str,),
        infinitely_yield_atom: bool = False,
        iter_none: bool = False
) -> Iterator:
    """
    Get an iterator for an iterable object which is not of `atom_types`;
    otherwise yield the object itself.

    Args:
        _it: the object to iterate through.
        atom_types: if `_it` is of `atom_types`, then it is treated as a non-iterable;
            by default we treat a string object as non-iterable.
        infinitely_yield_atom: if `_it` is an atom, then yield it infinitely;
            this option is useful when combining `iter_` with `zip_longest`.
        iter_none: True to still yield `_it` if it is None.

    Returns: an iterator iterating through `_it` if it is iterable and not of `atom_types`;
        otherwise an iterator yielding `_it` itself.

    Examples:
        >>> list(iter__(0))
        [0]
        >>> list(iter__('123'))
        ['123']
        >>> list(iter__('123', atom_types=None))
        ['1', '2', '3']
        >>> list(iter__(None))
        []
        >>> list(iter__((1, 2, None)))
        [1, 2, None]
        >>> list(iter__(None, iter_none=True))
        [None]
        >>> from itertools import zip_longest
        >>> list(zip_longest([1,2,3], iter__(0, infinitely_yield_atom=True)))
        [(1,0), (2,0), (3,0)]

    """
    if _it is not None:
        if iterable__(_it, atom_types=atom_types):
            yield from _it
        elif infinitely_yield_atom:
            while True:
                yield _it
        else:
            yield _it
    elif iter_none:
        if infinitely_yield_atom:
            while True:
                yield _it
        else:
            yield _it


def product__(*iterables, atom_types=(str,), ignore_none=False):
    """
    Cartesian product of input iterables like `product`, but any one of `iterables` of `atom_types`
    will be treated as non-iterable, and any None value in `iterables` can be ignored
    if `ignore_none` is set True.

    Examples:
        >>> list(product__([1,2], 3))
        [(1, 3), (2, 3)]
        >>> list(product__([1,2], None, 3))
        [(1, None, 3), (2, None, 3)]
        >>> list(product__([1,2], None, 3, ignore_none=True))
        [(1, 3), (2, 3)]

    """
    if ignore_none:
        yield from product(
            *(iter__(x, atom_types=atom_types) for x in iterables if x is not None)
        )
    else:
        yield from product(
            *(iter__(x, atom_types=atom_types, iter_none=True) for x in iterables)
        )


def product_(*iterables, non_atom_types=(list, tuple, set), ignore_none=False):
    """
    Cartesian product of input iterables like `product`, but any one of `iterables` of `atom_types`
    will be treated as non-iterable, and any None value in `iterables` can be ignored
    if `ignore_none` is set True.

    Examples:
        >>> list(product_([1,2], 3))
        [(1, 3), (2, 3)]
        >>> list(product_([1,2], None, 3))
        [(1, None, 3), (2, None, 3)]
        >>> list(product_([1,2], None, 3, ignore_none=True))
        [(1, 3), (2, 3)]

    """
    if ignore_none:
        yield from product(
            *(iter_(x, non_atom_types=non_atom_types) for x in iterables if x is not None)
        )
    else:
        yield from product(
            *(iter_(x, non_atom_types=non_atom_types, iter_none=True) for x in iterables)
        )


def chain__(*_its, atom_types=(str,), iter_none=False):
    return chain(iter__(_it, atom_types=atom_types, iter_none=iter_none) for _it in _its)


def zip__(*iterables, atom_types=(str,), iter_none: bool = True):
    """
    Allows zipping None or atoms (of `atom_types`) with iterables.

    Examples:
        >>> list(zip__('12', '34'))
        [('12', '34')]
        >>> list(zip__(0, 1, 2))
        [(0, 1, 2)]
        >>> list(zip__(0, [1,2,3], [5,6,7,8]))
        [(0, 1, 5), (0, 2, 6), (0, 3, 7)]
        >>> list(zip__([1,2,3], None))
        [(1, None), (2, None), (3, None)]
        >>> list(zip__([1,2,3], None, [5,6,7,8]))
        [(1, None, 5), (2, None, 6), (3, None, 7)]
        >>> list(zip__(None, None, None))
        [(None, None, None)]
        >>> list(zip__(None, None, None, iter_none=False))
        []
    """
    if any(iterable__(x, atom_types=atom_types) for x in iterables):
        yield from zip(
            *(
                iter__(x, atom_types=atom_types, infinitely_yield_atom=True, iter_none=iter_none)
                for x in iterables
            )
        )
    elif iter_none or any(x is not None for x in iterables):
        yield iterables


def zip_longest__(
        *iterables,
        atom_types=(str,),
        fill_none_by_previous_values: Union[bool, Tuple[bool], List[bool]] = True
):
    """
    Allows zipping atoms with iterables.

    Instead of using None as a placeholder when one interable is shorter,
    this function can use the last available value at the same position;
    set `fill_none_by_previous_values` to True or False
    to enable/disable this behavior for all dimensions,
    or a list of True/False values to control the behavior for each dimension.

    Examples:
        >>> list(zip_longest__(1, None))
        [(1, None)]
        >>> list(zip_longest__([1, 2, 3], None))
        [(1, None), (2, None), (3, None)]
        >>> list(zip_longest__(0, [1, 2, 3], [5, 6, 7, 8]))
        [(0, 1, 5), (0, 2, 6), (0, 3, 7), (0, 3, 8)]
        >>> list(zip_longest__([1, 2, 3], None, [5, 6, 7, 8]))
        [(1, None, 5), (2, None, 6), (3, None, 7), (3, None, 8)]
        >>> list(zip_longest__(0, [1, 2, 3], [5, 6, 7, 8], fill_none_by_previous_values=False))
        [(0, 1, 5), (None, 2, 6), (None, 3, 7), (None, None, 8)]
        >>> list(
        ...   zip_longest__(
        ...      0, [1, 2, 3], [5, 6, 7, 8],
        ...      fill_none_by_previous_values=[True, False, False, False]
        ...   )
        ... )
        [(0, 1, 5), (0, 2, 6), (0, 3, 7), (0, None, 8)]
    """
    zip_obj = zip_longest(*(iter__(x, atom_types=atom_types) for x in iterables))

    if not fill_none_by_previous_values:
        yield from zip_obj
    else:
        _items = next(zip_obj)
        yield _items

        if isinstance(fill_none_by_previous_values, (list, tuple)):
            for items in zip_obj:
                _items = tuple(
                    (x if (x is not None or not _fill_none) else y)
                    for x, y, _fill_none in zip(items, _items, fill_none_by_previous_values)
                )
                yield _items
        elif fill_none_by_previous_values is True:
            for items in zip_obj:
                _items = tuple(
                    (x if x is not None else y)
                    for x, y in zip(items, _items)
                )
                yield _items


def unzip(
        tuples: Iterable[Tuple],
        idx: Optional[Union[int, Iterable[int]]] = None
) -> Union[Tuple, Iterable[Tuple]]:
    """
    Unzips a sequence of tuples to a tuple of sequences.
    Can choose to optionally return the sequence at one or more specified index(es).

    Examples:
        >>> zipped_seq = [(1, -1), (2, -2), (3, -3), (4, -4), (5, -5)]
        >>> list(unzip(zipped_seq))
        [(1, 2, 3, 4, 5), (-1, -2, -3, -4, -5)]
        >>> list(unzip(zipped_seq, idx=1))
        [-1, -2, -3, -4, -5]
    """
    if idx is None:
        return zip(*tuples)
    elif idx == 0:
        return next(zip(*tuples))
    elif type(idx) is int:
        return tuple(zip(*tuples))[idx]
    else:
        unzips = tuple(zip(*tuples))
        return (unzips[_idx] for _idx in idx)


def max_len(x, atom_types=(str,), default=0):
    if x is None:
        return default
    return max(len(_x) for _x in iter__(x, atom_types=atom_types))


def min_len(x, atom_types=(str,), default=0):
    if x is None:
        return default
    return min(len(_x) for _x in iter__(x, atom_types=atom_types))


def flatten_iter(x: Iterable, non_atom_types=(list, tuple), always_consider_iterator_as_non_atom: bool = True) -> Iterator:
    """
    Flattens a nested iterable into a flat generator, yielding elements one by one.
    Handles nesting by recursively yielding from any sub-iterable that is considered
    non-atomic according to the given types.

    Args:
        x: The iterable to flatten. Can contain nested iterables.
        non_atom_types: A tuple of types to consider as non-atomic.
            These are types that should be flattened.
        always_consider_iterator_as_non_atom: If True, any iterator type is considered non-atomic by default.
            If False, only the types specified in non_atom_types are considered.

    Yields:
        Elements of x, flattened.

    Examples:
        >>> list(flatten_iter([1, 2, [3, 4], (5, 6)]))
        [1, 2, 3, 4, 5, 6]
        >>> list(flatten_iter([1, (2, [3, 4], 5), 6], non_atom_types=(tuple,)))
        [1, 2, [3, 4], 5, 6]
        >>> list(flatten_iter([1, [2, [3, 4]], 5]))
        [1, 2, [3, 4], 5]
    """
    non_atom_types = _get_non_atom_types(non_atom_types, always_consider_iterator_as_non_atom)
    for _x in x:
        if isinstance(_x, non_atom_types):
            yield from _x
        else:
            yield _x


def filter_(_filter, _it):
    if callable(_filter):
        return filter(_filter, _it)
    else:
        yield from (x for x in _it if x in _filter)


def filter_by_head_element(_filter, _it):
    if not _filter:
        return _it
    if callable(_filter):
        return filter(lambda x: _filter(next(iter(x))), _it)
    else:
        return (x for x in _it if next(iter(x)) in _filter)


def filter_tuples_by_head_element(_filter, _it):
    """
    Filter an iterable of tuples based on the head element (first element) of each tuple.

    Args:
        _filter: A filter condition. It can be None, a callable, or an iterable.
            - If None, the input iterable (_it) is returned unmodified.
            - If callable, it should accept the head element and return a boolean value.
            - If iterable, it should contain elements to be matched with the head element.
        _it: An iterable of tuples to be filtered.

    Returns:
        An iterable of tuples filtered based on the head element according to the given filter.

    Examples:
        >>> input_tuples = [(1, 'apple'), (2, 'banana'), (3, 'cherry')]
        >>> list(filter_tuples_by_head_element(lambda x: x % 2 == 0, input_tuples))
        [(2, 'banana')]

        >>> list(filter_tuples_by_head_element([2, 3], input_tuples))
        [(2, 'banana'), (3, 'cherry')]

        >>> list(filter_tuples_by_head_element(None, input_tuples))
        [(1, 'apple'), (2, 'banana'), (3, 'cherry')]
    """
    if not _filter:
        return _it
    if callable(_filter):
        return filter(lambda x: _filter(x[0]), _it)
    else:
        return (x for x in _it if x[0] in _filter)


def get_by_indexes(x, *index):
    return tuple(x[i] for i in index)


def shuffle_together(*arrs: Iterable):
    """
    Randomly shuffles multiple lists altogether, so that elements at the same position
    of each list still have the same indices after shuffle.

    Examples:
        # expects to return results like ((2, 1, 3, 4), ('ii', 'i', 'iii', 'iv'))
        >>> shuffle_together([1,2,3,4],['i', 'ii', 'iii', 'iv'])

    Args:
        arrs: the lists to shuffle together.

    Returns: the lists after shuffle.
    """
    tmp = list(zip(*arrs))
    random.shuffle(tmp)
    return tuple(zip(*tmp))


def split_iter(
        it: Union[Iterator, Iterable, List],
        num_splits: int,
        use_tqdm=False,
        tqdm_msg=None
) -> List[List]:
    """
    Splits the items read from an iterator into a list of lists, where each nested list is a split
    of the input iterator.

    See :func:`split_list`.

    """
    return split_list(
        list_to_split=it if sliceable(it) else list(tqdm_wrap(it, use_tqdm, tqdm_msg)),
        num_splits=num_splits
    )


def all__(_it, cond: Callable, atom_types=(str,)):
    if isinstance(_it, atom_types):
        return cond(_it)
    else:
        return all(cond(item) for item in _it)


def get_item_if_singleton(x):
    """
    If the size of the input can be measured by `len`,
            and the input contains a single item, then returns the single item;
        in particular, if the input is a mapping and contains a single key/value pair,
            then the single value will be returned.
    Otherwise, returns the input object itself.
    """
    if isinstance(x, Mapping):
        return next(iter(x.values())) if len(x) == 1 else x
    elif hasattr(x, '__len__'):
        return x[0] if len(x) == 1 else x
    else:
        return x


# region extraction

def first(x, cond: Callable = None):
    """
    Returns the first element of an iterable that meets a condition. If no condition
    is specified, returns the first element of the iterable.

    Args:
        x: An iterable from which the element is returned.
        cond : A callable that takes an element of the iterable and returns
                a boolean. If this callable returns True for an element, that
                element is returned.

    Returns:
        The first element that satisfies the condition or the first element if no condition is provided.

    Examples:
        >>> first([1, 2, 3, 4], lambda x: x > 2)
        3
        >>> first([1, 2, 3, 4])
        1
    """
    if cond is None:
        return x[0]
    else:
        for _x in x:
            if cond(_x):
                return _x


def first_(x, non_atom_types=(list, tuple), cond: Callable = None):
    """
    Returns the first element of a sequence or the sequence itself if it is not a list or tuple.
    If a condition is specified, it returns the first element that meets the condition.

    Args:
        x: A sequence or any object.
        non_atom_types: Types that are considered non-atomic, typically list and tuple.
        cond: A callable to evaluate each element of the sequence.

    Returns:
        The first element that satisfies the condition, the first element if no condition is provided,
        or the object itself if it is not a list or tuple.

    Examples:
        >>> first_([1, 2, 3], cond=lambda x: x > 1)
        2
        >>> first_([1, 2, 3])
        1
        >>> first_('hello')
        'hello'
    """
    if isinstance(x, non_atom_types):
        if cond is None:
            return x[0]
        else:
            for _x in x:
                if cond(_x):
                    return _x
    else:
        return x


def first__(x, atom_types=str, cond: Callable = None):
    """
    Returns the first element of a sequence or the sequence itself if it's of an atomic type
    or does not support item access. If a condition is specified, it returns the first element that meets the condition.

    Args:
        x: A sequence or any object.
        atom_types: Types that are considered atomic, typically string.
        cond: A callable to evaluate each element of the sequence.

    Returns:
        The first element that satisfies the condition, the first element if no condition is provided,
        or the object itself if it is of an atomic type or does not support item access.

    Examples:
        >>> first__([1, 2, 3], cond=lambda x: x > 2)
        3
        >>> first__([1, 2, 3])
        1
        >>> first__('hello')
        'hello'
    """
    if isinstance(x, atom_types) or (not hasattr(x, '__getitem__')):
        return x
    elif cond is None:
        return x[0]
    else:
        for _x in x:
            if cond(_x):
                return _x


def last(x, cond: Callable = None):
    """
    Returns the last element of an iterable that meets a condition. If no condition
    is specified, returns the last element of the iterable.

    Args:
        x: An iterable from which the element is returned.
        cond: A callable that takes an element of the iterable and returns
              a boolean. If this callable returns True for an element, that
              element is considered for being the last one returned.

    Returns:
        The last element that satisfies the condition or the last element if no condition is provided.

    Examples:
        >>> last([1, 2, 3, 4], lambda x: x < 4)
        3
        >>> last([1, 2, 3, 4])
        4
    """
    if isinstance(x, Sequence):
        for i in range(len(x) - 1, -1, -1):
            if cond is None or cond(x[i]):
                return x[i]
    else:
        result = None
        for _x in x:
            if cond is None or cond(_x):
                result = _x
        return result


def last_(x, non_atom_types=(list, tuple), cond: Callable = None):
    """
    Returns the last element of a sequence or the sequence itself if it is not a list or tuple.
    If a condition is specified, it returns the last element that meets the condition.

    Args:
        x: A sequence or any object.
        non_atom_types: Types that are considered non-atomic, typically list and tuple.
        cond: A callable to evaluate each element of the sequence.

    Returns:
        The last element that satisfies the condition, the last element if no condition is provided,
        or the object itself if it is not a list or tuple.

    Examples:
        >>> last_([1, 2, 3, 4], cond=lambda x: x % 2 == 1)
        3
        >>> last_([1, 2, 3, 4])
        4
        >>> last_('hello')
        'hello'
    """
    if isinstance(x, non_atom_types):
        if isinstance(x, Sequence):
            for i in range(len(x) - 1, -1, -1):
                if cond is None or cond(x[i]):
                    return x[i]
        else:
            result = None
            for _x in x:
                if cond is None or cond(_x):
                    result = _x
            return result
    else:
        return x


def last__(x, atom_types=(str,), cond: Callable = None):
    """
    Returns the last element of a sequence or the sequence itself if it's of an atomic type
    or does not support item access. If a condition is specified, it returns the last element that meets the condition.

    Args:
        x: A sequence or any object.
        atom_types: Types that are considered atomic, typically string.
        cond: A callable to evaluate each element of the sequence.

    Returns:
        The last element that satisfies the condition, the last element if no condition is provided,
        or the object itself if it is of an atomic type or does not support item access.

    Examples:
        >>> last__([1, 2, 3, 4], cond=lambda x: x > 1)
        4
        >>> last__([1, 2, 3, 4])
        4
        >>> last__('hello')
        'hello'
    """
    if isinstance(x, atom_types) or (not hasattr(x, '__getitem__')):
        return x
    elif isinstance(x, Sequence):
        for i in range(len(x) - 1, -1, -1):
            if cond is None or cond(x[i]):
                return x[i]
    else:
        result = None
        for _x in x:
            if cond is None or cond(_x):
                result = _x
        return result


# endregion

# region sub iterable

def head(iterable: Iterable, cond: Callable) -> Iterator:
    """
    Yields elements from the beginning of an iterable until a condition is met.

    Args:
        iterable: An iterable from which elements are yielded.
        cond: A callable that takes an element of the iterable and returns
                a boolean. If this callable returns True for an element, the iteration
                will break after yielding that element.

    Yields:
        Elements from the iterable until the condition is satisfied.

    Examples:
        >>> list(head([1, 2, 3, 4, 5], lambda x: x >= 3))
        [1, 2, 3]
    """
    for x in iterable:
        yield x
        if cond(x):
            break


def tail(iterable: Iterable, cond: Callable) -> Iterator:
    """
    Yields elements from an iterable starting from the element just after a condition is first met.

    Args:
        iterable: An iterable from which elements are yielded.
        cond: A callable that takes an element of the iterable and returns
                a boolean. When this callable returns True for the first time,
                the function begins to yield every subsequent element.

    Yields:
        Elements from the iterable, starting from the element after the condition is first met.

    Examples:
        >>> list(tail([1, 2, 3, 4, 5], lambda x: x < 3))
        [3, 4, 5]
    """
    start_yield = False
    for x in iterable:
        if not start_yield:
            if cond(x):
                start_yield = True
        else:
            yield x

# endregion
