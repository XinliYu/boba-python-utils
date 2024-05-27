import copy
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from itertools import islice
from typing import Dict, Union, Iterable, Tuple, Any, Mapping, List, Callable, Iterator, Counter, Set, Type, Sequence

from boba_python_utils.common_utils.iter_helper import zip_longest__, tqdm_wrap, iter_, flatten_iter
from boba_python_utils.common_utils.typing_helper import solve_key_value_pairs, is_str, nonstr_iterable
from boba_python_utils.string_utils.prefix_suffix import remove_prefix_suffix, add_suffix


@contextmanager
def temporary_value(d: Dict, key: Any, value: Any):
    _replace = False
    _old_value = None

    try:
        if key in d:
            _replace = True
            _old_value = d[key]
            d[key] = value
        else:
            d[key] = value
        yield
    finally:
        if _replace:
            d[key] = _old_value
        else:
            del d[key]


@contextmanager
def use_namespace(d: Dict, namespace, sep='_', namespace_essential_keys: Iterable[str] = None):
    """

    Args:
        d:
        namespace:
        sep:

    Returns:
    Examples:
        >>> d = {'key1': 1, 'key2': 2, 'key3': 3, 'x_key1': -1, 'x_key3': -3}
        >>> with use_namespace(d, 'x'):
        ...    print(d)
        {'key1': -1, 'key2': 2, 'key3': -3}
        >>> with use_namespace(d, 'x', namespace_essential_keys=(['key2', 'key4'])):
        ...    print(d)
        {'key1': -1, 'key3': -3, 'key2': None, 'key4': None}

        >>> print(d)
        {'x_key1': -1, 'key1': 1, 'x_key3': -3, 'key3': 3, 'key2': 2}

    """
    if sep and namespace[-1] != sep:
        namespace += sep
    _changed_key_values = {}
    _delete = set()
    _update = {}
    try:
        for k, v in d.items():
            if k.startswith(namespace):
                _changed_key_values[k] = v
                _k = k[len(namespace):]
                if _k in d:
                    _changed_key_values[_k] = d[_k]
                _update[_k] = v
                _delete.add(k)
        for k in iter_(namespace_essential_keys):
            if k in d:
                _changed_key_values[k] = d[k]
                _delete.add(k)
            _update[k] = None

        for k in _delete:
            del d[k]
        d.update(_update)
        _update = tuple(_update.keys())
        yield
    finally:
        for k in _update:
            del d[k]
        d.update(_changed_key_values)


def _exchange_values(d: Dict, key_map: Tuple, tmp_keys: set):
    for k1, k2 in key_map:
        if k1 not in d:
            d[k1] = None
            tmp_keys.add(k1)
        if k2 not in d:
            d[k2] = None
            tmp_keys.add(k2)
        d[k1], d[k2] = d[k2], d[k1]


@contextmanager
def key_value_exchanged(d: Dict, *key_map):
    """

    Args:
        d:
        *key_map:

    Returns:

    Examples:
        >>> d = {'key1': 1, 'key2': 2, 'key3': 3}
        >>> with key_value_exchanged(d, {'key1': 'key3'}):
        ...    print(d)
        {'key1': 3, 'key2': 2, 'key3': 1}
        >>> print(d)
        {'key1': 1, 'key2': 2, 'key3': 3}
        >>> with key_value_exchanged(d, {'key1': 'key4'}):
        ...    print(d)
        {'key1': None, 'key2': 2, 'key3': 3, 'key4': 1}
        >>> print(d)
        {'key1': 1, 'key2': 2, 'key3': 3}
        >>> with key_value_exchanged(d, {'key1': 'key4'}):
        ...    d['key4'] = 4
        ...    print(d)
        {'key1': None, 'key2': 2, 'key3': 3, 'key4': 4}
        >>> print(d)
        {'key1': 4, 'key2': 2, 'key3': 3}
        >>> d = {'key1': 1, 'key2': 2, 'key3': 3}
        >>> with key_value_exchanged(d, {'key1': 'key4'}):
        ...    d['key1'] = 4
        ...    print(d)
        {'key1': 4, 'key2': 2, 'key3': 3, 'key4': 1}
        >>> print(d)
        {'key1': 1, 'key2': 2, 'key3': 3, 'key4': 4}
    """
    key_map = tuple(solve_key_value_pairs(*key_map))
    tmp_keys = set()
    try:
        _exchange_values(d=d, key_map=key_map, tmp_keys=tmp_keys)
        yield
    finally:
        for k1, k2 in key_map:
            d[k1], d[k2] = d[k2], d[k1]
        for k in tmp_keys:
            if d[k] is None:
                del d[k]


def key_prefix_removed(d: Dict, prefix: str, prefix_sep: str = '_'):
    """
    A context manager that temporarily removes the specified prefixes from keys  of the dictionary.
    Args:
        d:
        prefix:
        prefix_sep:
    Returns:

    Examples:
        >>> d = {'key1': 1, 'new_key1': 2, 'new_key2': 3}
        >>> with key_prefix_removed(d, 'new'):
        ...    print(d)
        {'key1': 2, 'new_key1': 1, 'new_key2': None, 'key2': 3}
        >>> print(d)
        {'key1': 1, 'new_key1': 2, 'new_key2': 3}
        >>> with key_prefix_removed(d, 'new'):
        ...    d['key1'] = 4
        ...    print(d)
        {'key1': 4, 'new_key1': 1, 'new_key2': None, 'key2': 3}
        >>> print(d)
        {'key1': 1, 'new_key1': 4, 'new_key2': 3}
        >>> d = {'key1': 1, 'new_key1': 2, 'new_key2': 3}
        >>> with key_prefix_removed(d, 'new'):
        ...    d['new_key2'] = 4
        ...    print(d)
        {'key1': 2, 'new_key1': 1, 'new_key2': 4, 'key2': 3}
        >>> print(d)
        {'key1': 1, 'new_key1': 2, 'new_key2': 3, 'key2': 4}
    """
    key_map = {
        remove_prefix_suffix(k, prefixes=prefix, sep=prefix_sep): k
        for k in d if k.startswith(prefix)
    }
    return key_value_exchanged(d, key_map)


def get_category_dict(arr: Iterable, categorization: Union[Callable, str] = len) -> Dict[Any, List]:
    """
    Categorizes an iterable into a category dictionary.
    Args:
        arr: the iterable.
        categorization: an attribute of each element of `arr` whose value can be used as the
            category label, or a function to extract a category label from each element of `arr`.

    Returns: a dictionary mapping category labels to elements under the category.

    Examples:
        >>> get_category_dict([(1, 2), (3, 4), (5, 6, 7)])
        defaultdict(<class 'list'>, {2: [(1, 2), (3, 4)], 3: [(5, 6, 7)]})

    """
    categories = defaultdict(list)
    for x in arr:
        categories[
            getattr(x, categorization) if is_str(categorization) else categorization(x)
        ].append(x)
    return categories


def get_keys(d: Union[Mapping, Any]) -> Iterable:
    """
    Gets keys from an object. If the object is a Mapping, we return its keys,
    otherwise we return the keys of `__dict__` of that object.

    Examples:
        >>> args = { 'arg1': '1', 'arg2': '2' }
        >>> tuple(get_keys(args))
        ('arg1', 'arg2')
        >>> from argparse import Namespace
        >>> args = Namespace(**args)
        >>> tuple(get_keys(args))
        ('arg1', 'arg2')
    """

    if isinstance(d, Mapping):
        return d.keys()
    else:
        return d.__dict__.keys()


# region value fetching
def get_(
        d: Union[Mapping, Any],
        key1: Union[Callable, Any],
        key2: Union[Callable, Any] = None,
        default: Any = None,
        raise_key_error: bool = False
) -> Any:
    """
    Fetches a value from a mapping or an attribute from an object, with two possible keys `key1`
    or `key2`. This function first tries `key1`, and if unsuccessful, tries `key2`. The keys can be
    either direct references in the mapping/object or callable functions that process the mapping/object.

    Args:
        d: The mapping to retrieve a value from or an object to retrieve an attribute from.
        key1: The first key or a callable to try.
        key2: The alternative key or a callable to try.
        default: Returns this default value if both `key1` and `key2` do not exist, and
            `raise_key_error` is set to False.
        raise_key_error: True to raise a KeyError if both `key1` or `key2` do not exist,
            and in this case the `default` will be ignored.

    Returns:
        A value retrieved from the mapping by `key1` or `key2`, or the default if both fail.

    Examples:
        >>> d = {'a': 1, 'b': 2, 'c': 3}
        >>> get_(d, 'a', 'b')
        1
        >>> get_(d, 'x', 'y', default=0)
        0
        >>> get_(d, lambda x: x.get('a') * 2, 'b')
        2
        >>> get_(d, lambda x: x['z'], lambda x: x.get('b') * 2, default=-1)
        4
    """
    # First key logic
    try:
        result = key1(d) if callable(key1) else (d[key1] if isinstance(d, Mapping) else getattr(d, key1))
        return result
    except (KeyError, AttributeError):
        # Second key logic if first key fails
        if key2 is not None:
            try:
                return key2(d) if callable(key2) else (d[key2] if isinstance(d, Mapping) else getattr(d, key2))
            except (KeyError, AttributeError):
                pass

    # Handle default and error cases
    if raise_key_error:
        if key2 is not None:
            raise KeyError(f"Neither key '{key1}' nor key '{key2}' exist or are valid")
        else:
            raise KeyError(f"Key '{key1}' does not exist or are not valid")
    return default


def _get__for_mapping(d, keys, default, raise_key_error, return_hit_key):
    for key in flatten_iter(keys, non_atom_types=(List,)):
        if key in d:
            value = d[key]
            if return_hit_key:
                return key, value
            else:
                return value
    if raise_key_error:
        raise KeyError(f"None of the keys '{keys}' exist")
    return default


def _get__for_non_mapping(d, keys, default, raise_key_error, return_hit_key):
    for key in flatten_iter(keys, non_atom_types=(List,)):
        if hasattr(d, key):
            value = getattr(d, key)
            if return_hit_key:
                return key, value
            else:
                return value
    if raise_key_error:
        raise KeyError(f"None of the keys '{keys}' exist")
    return default


def get__(d: Union[Mapping, Any], *keys, default=None, raise_key_error: bool = False, return_hit_key: bool = False):
    """
    Fetches value from a mapping, or fetches an attribute from an object,
    with multiple possible keys specified in `keys`. This function tries
    the specified keys in order.

    Args:
        d: the mapping to retrieve a value from.
        keys: the keys to try.
        default: returns this default value if all `keys` do not exist in the mapping,
            and `raise_key_error` is set False.
        raise_key_error: True to raise :class:`KeyError` if all `keys` do not exist,
            and in this case the `default` will be ignored.
        return_hit_key: If True, returns a tuple (key, value) where key is the key or attribute
            that resulted in a successful value retrieval.

    Returns:
        A value retrieved from the mapping or object by one of the `keys`, or the default if none found.
        If `return_hit_key` is True, returns a tuple of (key, value).


    Examples:
        Example 1: Using a dictionary to retrieve a value by key
        >>> data = {'name': 'Alice', 'age': 30}
        >>> get__(data, 'name')
        'Alice'

        Example 2: Using the same dictionary, trying multiple keys where one exists
        >>> get__(data, 'gender', 'age')
        30
        >>> get__(data, ['gender', 'age'])
        30
        >>> get__(data, ['gender', 'age'], return_hit_key=True)
        ('age', 30)
        >>> get__(data, ('gender', 'age'), raise_key_error=True)
        Traceback (most recent call last):
        ...
        KeyError: "None of the keys '(('gender', 'age'),)' exist"

        Example 3: Using default value when none of the keys are present
        >>> get__(data, 'gender', default='Unknown')
        'Unknown'

        Example 4: Raising a KeyError when keys are not found, and raise_key_error is True
        >>> get__(data, 'gender', raise_key_error=True)
        Traceback (most recent call last):
        ...
        KeyError: "None of the keys '('gender',)' exist"

        Example 5: Fetching an attribute from an object
        >>> class Person:
        ...     def __init__(self, name, age):
        ...         self.name = name
        ...         self.age = age
        ...
        >>> person = Person('Bob', 40)
        >>> get__(person, 'name')
        'Bob'

        Example 6: Using multiple keys to fetch the first existing attribute from an object
        >>> get__(person, 'gender', 'age')
        40
        >>> get__(person, ['gender', 'age'])
        40
        >>> get__(person, ['gender', 'age'], return_hit_key=True)
        ('age', 40)
        >>> get__(person, ('gender', 'age'))
        Traceback (most recent call last):
        ...
        TypeError: hasattr(): attribute name must be string
    """
    if isinstance(d, Mapping):
        return _get__for_mapping(d, keys, default, raise_key_error, return_hit_key)
    else:
        return _get__for_non_mapping(d, keys, default, raise_key_error, return_hit_key)


def _get_multiple(
        d: Union[Mapping, Any],
        keys,
        default=None,
        raise_key_error: bool = False
):
    result = {}
    if isinstance(d, Mapping):
        for key in keys:
            if isinstance(key, List):
                _key, value = _get__for_mapping(d, key, default, raise_key_error, True)
                result[_key] = value
            else:
                if key in d:
                    result[key] = d[key]
                elif raise_key_error:
                    raise KeyError(f"Key '{key}' does not exist")
                else:
                    result[key] = default
    else:
        for key in keys:
            if isinstance(key, List):
                _key, value = _get__for_non_mapping(d, key, default, raise_key_error, True)
                result[_key] = value
            else:
                if hasattr(d, key):
                    result[key] = getattr(d, key)
                elif raise_key_error:
                    raise KeyError(f"Key '{key}' does not exist")
                else:
                    result[key] = default
    return result


def get_multiple(
        d: Union[Mapping, Any],
        *keys,
        default=None,
        raise_key_error=False,
        unpack_result_for_single_key: bool = True
):
    """
    Retrieves values from a mapping or object attributes based on given keys. This function
    can fetch values for multiple keys and optionally unpack the result if only one key is provided.
    If a key is not found, it can return a default value or raise an error.

    Args:
        d: The mapping or object from which to retrieve values.
        *keys: A variable number of keys or attributes to attempt to retrieve.
        default: The default value to return if a key is not found and `raise_key_error` is False.
        raise_key_error: If True, raises a KeyError when none of the keys are found. Default is False.
        unpack_result_for_single_key: If True and only one key is provided, returns the value directly
            rather than in a dictionary. Default is True.

    Returns:
        If `unpack_result_for_single_key` is True and one key is given, returns the single value directly.
        Otherwise, returns a dictionary of keys and their corresponding values. If keys are not found,
        it returns the `default` value or raises a KeyError based on `raise_key_error`.

    Examples:
        >>> d = {'key1': 1, 'key2': 2, 'key3': 3}
        >>> get_multiple(d, 'key1')
        1
        >>> get_multiple(d, 'key1', unpack_result_for_single_key=False)
        {'key1': 1}
        >>> get_multiple(d, 'key1', 'key2')
        {'key1': 1, 'key2': 2}
        >>> get_multiple(d, 'key4', default='Not found')
        'Not found'
        >>> get_multiple(d, 'key1', 'key4', default='Not found')
        {'key1': 1, 'key4': 'Not found'}
        >>> get_multiple(d, 'key4', raise_key_error=True)
        Traceback (most recent call last):
        ...
        KeyError: "None of the keys '('key4',)' exist"
    """
    if unpack_result_for_single_key and len(keys) == 1:
        return get__(d, keys[0], default=default, raise_key_error=raise_key_error)
    return _get_multiple(
        d=d, keys=keys, default=default, raise_key_error=raise_key_error
    )


def _get_by_index(
        d: Union[Mapping, Any],
        index: Union[int, Any],
        default: Any = None,
        raise_key_error: bool = False
) -> Any:
    try:
        return d[index]
    except IndexError as err:
        if raise_key_error:
            raise err
        return default


def get_by_index_or_key(
        d: Union[Mapping, Any],
        index_or_key: Union[int, Any],
        default: Any = None,
        raise_key_error: bool = False,
        indexed_types: Union[Type, Tuple[Type, ...]] = None
) -> Any:
    """
    Retrieves a value from a given data structure by index or key. This function is flexible
    and can handle various types of data structures such as lists, tuples, dictionaries, or
    other types specified in `indexed_types`.

    Args:
        d: The input data structure, which can be a list, tuple, dictionary, or any other type
           that supports indexing.
        index_or_key: The index (if `d` is a sequence) or key (if `d` is a mapping) used to retrieve the value.
        default: The default value to return if the index or key is not found and `raise_key_error` is False.
        raise_key_error: If True, raises a KeyError when the specified index or key is not found
                         instead of returning `default`.
        indexed_types: Optional tuple of additional types that support indexing to be handled by the function.

    Returns:
        The retrieved value from the input data structure. If the index or key is not found,
        it returns `default` unless `raise_key_error` is set to True, in which case a KeyError is raised.

    Examples:
        >>> get_by_index_or_key([1, 2, 3, 4], 0)
        1
        >>> get_by_index_or_key((10, 20, 30), 1)
        20
        >>> d = {'key1': 1, 'key2': 2, 'key3': 3}
        >>> get_by_index_or_key(d, 'key1')
        1
        >>> get_by_index_or_key(d, 'key4', default='Not Found')
        'Not Found'
        >>> get_by_index_or_key(d, 'key4', raise_key_error=True)
        Traceback (most recent call last):
        ...
        KeyError: "None of the keys '('key4',)' exist"
        >>> get_by_index_or_key('hello', 0)
        'h'
        >>> get_by_index_or_key('hello', 5, default='Not in range')
        'Not in range'
    """
    if isinstance(d, Sequence) or (bool(indexed_types) and isinstance(d, indexed_types)):
        return _get_by_index(
            d=d, index=index_or_key, default=default, raise_key_error=raise_key_error
        )
    else:
        return get_multiple(d, index_or_key, default=default, raise_key_error=raise_key_error)


def get_multiple_by_indexes_or_keys(
        d: Union[Mapping, Any],
        *indexes_or_keys: Union[int, Any],
        default: Any = None,
        raise_key_error: bool = False,
        unpack_result_for_single_key: bool = True,
        indexed_types: Union[Type, Tuple[Type]] = None
) -> Any:
    """
    Retrieves values from a given data structure by multiple indexes or keys, handling both collections
    and objects with attribute access. This function is capable of returning multiple values and can handle
    errors or return a default value if specified keys or indexes are not found.

    Args:
        d: The input data structure, which can be a list, tuple, dictionary, or any other type that supports indexing.
        *indexes_or_keys: Variable number of indexes or keys used to retrieve the values.
        default: The default value to return if an index or key is not found and `raise_key_error` is False.
        raise_key_error: If True, a KeyError or IndexError will be raised for the first missing index or key.
        unpack_result_for_single_key: If True and there is only one index_or_key, return the value directly
                                      rather than in a collection.
        indexed_types: Optional tuple of additional types that support indexing, to handle custom or less common types.

    Returns:
        If `unpack_result_for_single_key` is True and only one index_or_key is specified, the single value directly.
        Otherwise, returns a list, tuple, or dictionary (depending on the type of `d`) containing the retrieved values.
        If an index or key is not found, it returns `default` for that position unless `raise_key_error` is True.

    Examples:
        >>> get_multiple_by_indexes_or_keys([1, 2, 3, 4], 0, 3)
        [1, 4]
        >>> get_multiple_by_indexes_or_keys((10, 20, 30, 40), 1, 3)
        (20, 40)
        >>> d = {'key1': 1, 'key2': 2, 'key3': 3}
        >>> get_multiple_by_indexes_or_keys(d, 'key1', 'key2')
        {'key1': 1, 'key2': 2}
        >>> get_multiple_by_indexes_or_keys(d, 'key4', default='Not Found')
        'Not Found'
        >>> get_multiple_by_indexes_or_keys(d, 'key1', 'key4', default='Not Found')
        {'key1': 1, 'key4': 'Not Found'}
        >>> get_multiple_by_indexes_or_keys('hello', 0, 4)
        ['h', 'o']
        >>> get_multiple_by_indexes_or_keys('hello', 0, 5, default='Not in range')
        ['h', 'Not in range']
        >>> get_multiple_by_indexes_or_keys([1, 2, 3], 0, raise_key_error=True)
        1
    """
    if unpack_result_for_single_key and len(indexes_or_keys) == 1:
        return get_by_index_or_key(
            d=d,
            index_or_key=indexes_or_keys[0],
            default=default,
            raise_key_error=raise_key_error
        )
    elif isinstance(d, tuple) or (indexed_types is not None and isinstance(d, indexed_types)):
        return tuple(_get_by_index(d, i, default=default, raise_key_error=raise_key_error) for i in indexes_or_keys)
    elif isinstance(d, Sequence):
        return [_get_by_index(d, i, default=default, raise_key_error=raise_key_error) for i in indexes_or_keys]
    else:
        return _get_multiple(
            d=d, keys=indexes_or_keys, default=default, raise_key_error=raise_key_error
        )


def get_values_by_path(
        data,
        key_path,
        return_path: bool = True,
        return_single_value: bool = False,
        unpack_result_for_single_value: bool = True
):
    """
    Retrieves values from a nested dictionary by following a specified key path provided as a list of keys,
    using an iterative approach with a stack. This function handles lists within the nested structure by
    retrieving values for each list element.

    Args:
        data: The nested dictionary or JSON-like object from which to retrieve values.
        key_path: The list of keys that define the path to the desired value. If a key within the path
            points to a list, the function processes each item in the list.
        return_path: If True, returns tuples with keys representing the full path to the value
            and the value itself. If False, returns only the values. Defaults to True.
        return_single_value: If True, returns at most one value (i.e. the first retrieved value). Defaults to False.
        unpack_result_for_single_value: If True and the result is a single value, unpacks the value
            from the list. Defaults to True.

    Returns:
        List of tuples: Each tuple contains a key representing the full path to the value (concatenated with '.' and indexed
                        with '-n' for list elements), and the value itself. If a key points to a list, keys are like "a.b.c-1.d",
                        "a.b.c-2.d", etc.

    Example:
        >>> data = {
        ...     'a': {
        ...         'b': {
        ...             'c': [
        ...                 {'d': 'value1'},
        ...                 {'d': 'value2'}
        ...             ]
        ...         }
        ...     }
        ... }
        >>> get_values_by_path(data, ['a', 'b', 'c', 'd'])
        [('a.b.c-0.d', 'value1'), ('a.b.c-1.d', 'value2')]
        >>> get_values_by_path(data, ['a', 'b', 'c', '*'])
        [('a.b.c-0', ['value1']), ('a.b.c-1', ['value2'])]

        >>> data = {
        ...    'a': {
        ...        'b': {
        ...            'c': [
        ...                {'d': 'value1', 'e': [1, 2, {'f': 'deep'}]},
        ...                {'d': 'value2'}
        ...            ]
        ...        },
        ...        'x': {
        ...            'y': [10, 20, 30]
        ...        }
        ...    }
        ... }
        >>> get_values_by_path(data, ['a', 'b', 'c', 1, 'd'], unpack_result_for_single_value=False)
        [('a.b.c-1.d', 'value2')]
        >>> get_values_by_path(data, ['a', 'b', 'c', 'd'], unpack_result_for_single_value=False)
        [('a.b.c-0.d', 'value1'), ('a.b.c-1.d', 'value2')]
        >>> get_values_by_path(data, ['a', 'b', 'c', 'e', 'f'])
        ('a.b.c-0.e-2.f', 'deep')
        >>> get_values_by_path(data, ['a', 'x', 'y'])
        ('a.x.y', [10, 20, 30])
        >>> get_values_by_path(data, ['a', 'x', 'y', [0, 2]])
        [('a.x.y-0', 10), ('a.x.y-2', 30)]
    """
    # Stack to hold the items to process (dictionary, remaining keys, path)
    stack = [(data, key_path, '')]
    results = []

    while stack:
        current_data, keys, current_path = stack.pop()
        if not keys:
            if return_path:
                if return_single_value:
                    return current_path, current_data
                else:
                    results.append((current_path, current_data))
            else:
                if return_single_value:
                    return current_data
                else:
                    results.append(current_data)
            continue

        if isinstance(current_data, Sequence):
            if isinstance(keys[0], int):
                index = keys[0]
                new_keys = keys[1:]
                if index < len(current_data):
                    indexed_path = f"{current_path}-{index}"
                    stack.append((current_data[index], new_keys, indexed_path))
            elif keys[0] is not None and isinstance(keys[0], Sequence) and all(isinstance(_key, int) for _key in keys[0]):
                key = keys[0]
                new_keys = keys[1:]
                for _index in range(len(key) - 1, -1, -1):
                    index = key[_index]
                    if index < len(current_data):
                        indexed_path = f"{current_path}-{index}"
                        stack.append((current_data[index], new_keys, indexed_path))
            elif keys[0] == '*':
                new_keys = keys[1:]
                for index in range(len(current_data) - 1, -1, -1):
                    item = current_data[index]
                    indexed_path = f"{current_path}-{index}"
                    stack.append((list(item.values()), new_keys, indexed_path))
            else:
                for index in range(len(current_data) - 1, -1, -1):
                    item = current_data[index]
                    indexed_path = f"{current_path}-{index}"
                    stack.append((item, keys, indexed_path))
        else:
            key = keys[0]
            new_keys = keys[1:]
            try:
                new_data = get_multiple_by_indexes_or_keys(current_data, *iter_(key, non_atom_types=List), raise_key_error=True)
            except:
                continue
            new_path = f"{current_path}.{key}" if current_path else key
            stack.append((new_data, new_keys, new_path))

    if unpack_result_for_single_value and len(results) == 1:
        return results[0]
    else:
        return results


def get_value_by_path(
        data,
        key_path,
        return_path: bool = False
):
    return get_values_by_path(
        data=data,
        key_path=key_path,
        return_path=return_path,
        return_single_value=True
    )


def get_values_by_path_hierarchical(data, key_path):
    """
    Retrieves and preserves the nested structure of values from a dictionary based on a given key path.
    Handles nested lists by maintaining the full structure.

    Args:
        data: The nested dictionary or JSON-like object from which to retrieve values.
        key_path: The list of keys that define the path to the desired value. If a key within the path
                  points to a list, the function maintains the list structure in the output.

    Returns:
        A dictionary reflecting the nested structure of the data up to the last key in the path,
        preserving the structure even if the final value is within a list.

    Example:
        >>> data = {
        ... 'a': {
        ...        'b': {
        ...            'c': [
        ...                {'d': 'value1'},
        ...                {'d': 'value2'}
        ...            ]
        ...        }
        ...    }
        ... }
        >>> get_values_by_path_hierarchical(data, ['a', 'b', 'c', 'd'])
        {'a': {'b': {'c': [{'d': 'value1'}, {'d': 'value2'}]}}}
        >>> get_values_by_path_hierarchical(data, ['a', 'b', 'c', '*'])
        {'a': {'b': {'c': [['value1'], ['value2']]}}}

        >>> data = {
        ...    'a': {
        ...        'b': {
        ...            'c': [
        ...                {'d': {'e': 'value1'}},
        ...                {'d': {'e': 'value2'}}
        ...            ],
        ...            'f': {
        ...                'g': [
        ...                    {'h': 'value3'},
        ...                    {'h': 'value4'}
        ...                ],
        ...                'i': 'value5'
        ...            }
        ...        }
        ...    },
        ...    'j': {
        ...        'k': [
        ...            {'l': 'value6'},
        ...            {'m': {'n': 'value7'}}
        ...        ]
        ...    }
        ... }
        >>> get_values_by_path_hierarchical(data, ['a', 'b', 'c', 'd', 'e'])
        {'a': {'b': {'c': [{'d': {'e': 'value1'}}, {'d': {'e': 'value2'}}]}}}
        >>> get_values_by_path_hierarchical(data, ['a', 'b', 'f', 'g', 'h'])
        {'a': {'b': {'f': {'g': [{'h': 'value3'}, {'h': 'value4'}]}}}}
        >>> get_values_by_path_hierarchical(data, ['j', 'k'])
        {'j': {'k': [{'l': 'value6'}, {'m': {'n': 'value7'}}]}}
    """

    def recursive_get(sub_data, path):
        # Base case: if the path is empty, return the sub_data
        if not path:
            return sub_data

        if isinstance(sub_data, Sequence):
            if isinstance(path[0], int):
                return recursive_get(sub_data[path[0]], path[1:])
            elif path[0] is not None and isinstance(path[0], Sequence) and all(isinstance(index, int) for index in path[0]):
                next_path = path[1:]
                return [recursive_get(sub_data[i], next_path) for i in path[0] if i < len(sub_data)]
            elif path[0] == '*':
                next_path = path[1:]
                return [recursive_get(list(item.values()), next_path) for item in sub_data]
            else:
                return [recursive_get(item, path) for item in sub_data]
        else:
            key = path[0]
            next_path = path[1:]
            return {
                key: recursive_get(value, next_path)
                for key, value
                in get_multiple(
                    sub_data,
                    key,
                    raise_key_error=True,
                    unpack_result_for_single_key=False
                ).items()
            }

    # Start the recursive processing
    return recursive_get(data, key_path)


# endregion

class KeyRequirement(int, Enum):
    NoRequirement = 0
    ExistingKey = 1
    NewKey = 2


def _check_key_for_mapping(
        d: Mapping,
        key,
        key_requirement: KeyRequirement = KeyRequirement.NoRequirement,
        raise_key_error: bool = False
):
    if key_requirement == KeyRequirement.ExistingKey and key not in d:
        if raise_key_error:
            if len(d) < 100:
                raise ValueError(
                    f"key '{key}' does not exist in the mapping; "
                    f"the current keys of the mapping are '{tuple(d)}'"
                )
            else:
                raise ValueError(
                    f"key '{key}' does not exist in the mapping; "
                    f"the top 100 keys of the mapping are '{tuple(islice(d, 100))}'"
                )
        else:
            return False
    elif key_requirement == KeyRequirement.NewKey and key in d:
        if raise_key_error:
            raise ValueError(f"key '{key}' already exists in the mapping")
        else:
            return False
    return True


def _check_key_for_obj(
        d: Mapping,
        key,
        key_requirement: KeyRequirement = KeyRequirement.NoRequirement,
        raise_key_error: bool = False
):
    if key_requirement == KeyRequirement.ExistingKey and not hasattr(d, key):
        if raise_key_error:
            raise ValueError(f"key '{key}' does not exist in object {d}")
        else:
            return False
    elif key_requirement == KeyRequirement.NewKey and hasattr(d, key):
        if raise_key_error:
            raise ValueError(f"key '{key}' already exists in the object {d}")
        else:
            return False
    return True


def set_(
        d: Union[Dict, Any],
        key,
        value,
        key_requirement: KeyRequirement = KeyRequirement.NoRequirement,
        raise_key_error: bool = False
):
    if isinstance(d, Dict):
        if _check_key_for_mapping(
                d=d,
                key=key,
                key_requirement=key_requirement,
                raise_key_error=raise_key_error
        ):
            d[key] = value
    else:
        if _check_key_for_obj(
                d=d,
                key=key,
                key_requirement=key_requirement,
                raise_key_error=raise_key_error
        ):
            setattr(d, key, value)


def set__(
        d: Union[Dict, Any],
        keys_and_values: Mapping,
        key_requirement: KeyRequirement = KeyRequirement.NoRequirement,
        raise_key_error: bool = False
):
    if isinstance(d, Dict):
        for key, value in keys_and_values.items():
            if _check_key_for_mapping(
                    d=d,
                    key=key,
                    key_requirement=key_requirement,
                    raise_key_error=raise_key_error
            ):
                d[key] = value
    else:
        for key, value in keys_and_values.items():
            if _check_key_for_obj(
                    d=d,
                    key=key,
                    key_requirement=key_requirement,
                    raise_key_error=raise_key_error
            ):
                d[key] = value


def promote_keys(d: dict, keys_to_promote: Iterable, in_place: bool = True):
    """
    Promotes the specified keys to the beginning of the dictionary, optionally in-place.
    The order of the keys in `keys_to_promote` will be maintained in the resulting dictionary.

    Args:
        d: The input dictionary.
        keys_to_promote: An iterable of keys to promote to the beginning of the dictionary.
        in_place: If True, the input dictionary is modified in-place. If False, a new
            dictionary is created and returned.

    Returns:
        The dictionary with the specified keys promoted to the beginning.

    Examples:
        >>> d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        >>> d2 = promote_keys(d, ['c', 'a'], in_place=False)
        >>> print(d)
        {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        >>> print(d2)
        {'c': 3, 'a': 1, 'b': 2, 'd': 4}

        >>> d = {'x': 10, 'y': 20, 'z': 30}
        >>> d2 = promote_keys(d, ['y'], in_place=True)
        >>> print(d)
        {'y': 20, 'x': 10, 'z': 30}
        >>> print(d2)
        {'y': 20, 'x': 10, 'z': 30}
    """
    keys_to_demote = filter(lambda x: x not in keys_to_promote, d.keys())
    if in_place:
        keys_to_demote = tuple(keys_to_demote)
        for k in keys_to_promote:
            if k in d:
                v = d[k]
                del d[k]
                d[k] = v
        for k in keys_to_demote:
            v = d[k]
            del d[k]
            d[k] = v
        return d
    else:
        new_dict = {}
        for k in keys_to_promote:
            if k in d:
                new_dict[k] = d[k]
        for k in keys_to_demote:
            new_dict[k] = d[k]
        return new_dict


def kvswap(d: Mapping, allows_duplicate_values: bool = False):
    """
    Swaps the key and values in a mapping.
    This requires the values being qualified as keys (e.g. float numbers cannot be keys).

    If `allows_duplicate_values` is False, then the values must be distinct for the key-value
    swaping to succeed; otherwise, keys of the same values will be collected into lists in the
    returned key-value swaped dictionary.

    Examples:
        >>> kvswap({1: 2, 3: 4})
        {2: 1, 4: 3}
        >>> kvswap({1: ['a', 'b'], 3: ['c', 'd']})
        {('a', 'b'): 1, ('c', 'd'): 3}
        >>> kvswap({1: 2, 3: 2, 4: 5}, allows_duplicate_values=True)
        {2: [1, 3], 5: 4}
    """
    if allows_duplicate_values:
        out = {}
        for k, v in d.items():
            if isinstance(v, list):
                v = tuple(v)
            if v in out:
                _k = out[v]
                if isinstance(_k, list):
                    _k.append(k)
                else:
                    out[v] = [_k, k]
            else:
                out[v] = k
        return out
    else:
        return {(tuple(v) if isinstance(v, list) else v): k for k, v in d.items()}


def join_mapping_by_values(
        d1: Mapping, d2: Mapping,
        allows_duplicate_values: bool = False,
        keep_original_value_for_mis_join: bool = False
):
    """
    Joins the keys of two mappings through the values.
    For example, if one mapping is `{'a': 'b'}`, and the other mapping is `{'c': 'b'}`,
    then the joined mapping is `{'a': 'c'}`.

    Args:
        d1: the first mapping.
        d2: the second mapping.
        allows_duplicate_values: True to allow keys for the same values in `d2`
            being collected into lists; for example, joining `{'a': 'b'}`, `{'c': 'b', 'd': 'b'}`
            gets `{'a': ['c', 'd']}` if this parameter is set True.
        keep_original_value_for_mis_join: True to use the original value of `d1` if a value of `d1`
            cannot be found in the values of `d2`; otherwise, keys in `d1` will be dropped if their
            values cannot be found in the values of `d2`; for example, joining `{'a': 'b'}`,
            `{'c': 'd'}` gets `{'a': 'b'}` if this parameter is set True, but gets `{}` if
            this parameter is set False.

    Returns: a mapping from keys of `d1` to the keys of `d2`,
        joined by the values of the two mappings.

    Examples:
        >>> join_mapping_by_values(
        ...   d1={1: 2, 3: 4, 5: 6, 7: 8},
        ...   d2={-2: 2, -4: 4, -6: 6}
        ... )
        {1: -2, 3: -4, 5: -6}
        >>> join_mapping_by_values(
        ...   d1={1: 2, 3: 4, 5: 6, 7: 8},
        ...   d2={-2: 2, -4: 4, -6: 6},
        ...   keep_original_value_for_mis_join=True
        ... )
        {1: -2, 3: -4, 5: -6, 7: 8}
        >>> join_mapping_by_values(
        ...   d1={1: 2, 3: 4, 5: 6, 7: 8},
        ...   d2={-2: 2, -4: 4, -6: 6, -7: 6},
        ...   allows_duplicate_values=True,
        ...   keep_original_value_for_mis_join=True
        ... )
        {1: -2, 3: -4, 5: [-6, -7], 7: 8}
        >>> join_mapping_by_values(
        ...   d1={1: [2, 3], 3: [4, 5], 6: 7},
        ...   d2={-1: [2, 3], -3: [4, 5], -6: 6}
        ... )
        {1: -1, 3: -3}
        >>> join_mapping_by_values(
        ...   d1={1: [2, 3], 3: [4, 5], 6: 7},
        ...   d2={-1: [2, 3], -3: [4, 5], -6: 6},
        ...   keep_original_value_for_mis_join=True
        ... )
        {1: -1, 3: -3, 6: 7}
    """
    d2 = kvswap(d2, allows_duplicate_values=allows_duplicate_values)
    out = {}
    for k, v in d1.items():
        if isinstance(v, list):
            v = tuple(v)

        if v in d2:
            out[k] = d2[v]
        elif keep_original_value_for_mis_join:
            out[k] = v
    return out


# region sub-mapping

def sub_map(
        d: Mapping,
        sub_keys: Iterable,
        excluded_keys: Union[Set, Iterable, Mapping] = None
) -> dict:
    """
    Create a new dictionary containing a subset of key-value pairs from the input dictionary,
    optionally excluding specified keys.

    Args:
        d (Mapping): The input dictionary.
        sub_keys (Iterable): An iterable containing the keys to include in the output dictionary.
        excluded_keys (Union[Set, Iterable, Mapping], optional):
            An iterable or mapping containing keys to exclude from the output dictionary. Defaults to None.

    Returns:
        dict: A new dictionary containing the selected key-value pairs.

    Examples:
        >>> d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        >>> sub_map(d, ['a', 'c'])
        {'a': 1, 'c': 3}

        >>> sub_map(d, ['a', 'b', 'e'])
        {'a': 1, 'b': 2}

        >>> sub_map(d, ['a', 'c'], excluded_keys=['c'])
        {'a': 1}
    """
    if not excluded_keys:
        return {key: d[key] for key in sub_keys if key in d}
    else:
        return {key: d[key] for key in sub_keys if key in d and key not in excluded_keys}

def sub_map_by_prefix(prefix: str, d: Mapping, remove_prefix: bool = True):
    if remove_prefix:
        return {k[len(prefix):]: d[k] for k in filter(lambda k: k.startswith(prefix), d)}
    else:
        return {k: d[k] for k in filter(lambda k: k.startswith(prefix), d)}


# endregion

# region mapping merge
def merge_mappings(
        mappings: Iterator[Union[Dict, Mapping]],
        in_place: bool = False,
        use_tqdm: bool = False,
        tqdm_msg: str = None
) -> Mapping:
    """
    Merges multiple mappings as one.

    Args:
        mappings: the mappings to merge; if `in_place` is True,
            then the first mapping must be writable.
        in_place: True to merge all mappings into the first one of `mappings`,
            otherwise a new dictionary will be created to save the merged mappings.
        use_tqdm: True to enable tqdm wrap to display progress of the mapping merge.
        tqdm_msg: The tqdm message to display along with the prgress.

    Returns: the merged mapping; either the first mapping of `mappings`
        if `in_place` is True, or otherwise a new dictionary object.

    """
    mappings = iter(mappings)
    if use_tqdm:
        mappings = tqdm_wrap(mappings, use_tqdm=use_tqdm, tqdm_msg=tqdm_msg)
    d = next(mappings)

    if not in_place:
        d = dict(d)

    for _d in mappings:
        d.update(_d)
    return d


def merge_list_valued_mappings(
        mappings: Iterator[Mapping[Any, List]],
        in_place: bool = False
):
    """
    Merges mappings with lists as their values.
    Lists with the same key will be merged as a single list.

    Args:
        mappings: The mappings to merge.
        in_place: True if the merge results are saved in-place
            in the first dictionary of `mappings` and returned;
            False if creating a new dictionary to store the merge results.

    Returns: a dictionary of merged lists; either the first mapping of `mappings`
        if `in_place` is True, or otherwise a new dictionary object.

    """
    output_dict = None if in_place else defaultdict(list)
    for d in mappings:
        if output_dict is None:
            output_dict = defaultdict(list, d)
        else:
            if d is output_dict:
                raise ValueError(
                    "the first mapping appears twice "
                    "and 'in_place' is set True; "
                    "in this case we have lost the original data "
                    "in the first dictionary "
                    "and hence the merge cannot proceed"
                )

            for k, v in d.items():
                output_dict[k].extend(v)
    return output_dict


def merge_set_valued_mappings(
        mappings: Iterator[Mapping[Any, Set]],
        in_place: bool = False
):
    """
    Merges mappings with sets as their values.
    Sets with the same key will be merged as a single set.

    Args:
        mappings: The mappings to merge.
        in_place: True if the merge results are saved in-place
            in the first dictionary of `mappings` and returned;
            False if creating a new dictionary to store the merge results.

    Returns: a dictionary of merged sets; either the first mapping of `mappings`
        if `in_place` is True, or otherwise a new dictionary object.
    """
    output_dict = None if in_place else defaultdict(set)
    for d in mappings:
        if output_dict is None:
            output_dict = defaultdict(set, d)
        else:
            if d is output_dict:
                raise ValueError(
                    "the first mapping appears twice "
                    "and 'in_place' is set True; "
                    "in this case we have lost the original data "
                    "in the first dictionary "
                    "and hence the merge cannot proceed"
                )
            for k, v in d.items():
                output_dict[k] = output_dict[k].union(v)
    return output_dict


def merge_counter_valued_mappings(
        mappings: Iterator[Mapping[Any, Counter]],
        in_place: bool = False
):
    """
    Merges mappings with counters as their values.
    Counters with the same key will be merged as a single Counter.

    Args:
        mappings: The mappings to merge.
        in_place: True if the merge results are saved in-place
            in the first dictionary of `mappings` and returned;
            False if creating a new dictionary to store the merge results.

    Returns: a dictionary of merged counters; either the first mapping of `mappings`
        if `in_place` is True, or otherwise a new dictionary object.

    """
    output_dict = None if in_place else {}
    for d in mappings:
        if output_dict is None:
            output_dict = d
        else:
            if d is output_dict:
                raise ValueError(
                    "the first mapping appears twice "
                    "and 'in_place' is set True; "
                    "in this case we have lost the original data "
                    "in the first dictionary "
                    "and hence the merge cannot proceed"
                )
            for k, v in d.items():
                if k in output_dict:
                    output_dict[k] += v
                else:
                    output_dict[k] = v
    return output_dict


# endregion

# region counting
def _add_count(count_dict, k, v):
    if k in count_dict:
        if hasattr(v, '__add__') or hasattr(v, '__iadd__'):
            count_dict[k] += v
        elif hasattr(v, '__or__') or hasattr(v, '__ior__'):
            count_dict[k] |= v
        elif isinstance(v, dict):
            count_or_accumulate(count_dict[k], v)
    else:
        # ! have to make deep copy here
        # ! to avoid the potential error caused by a `v` being used in two counters
        count_dict[k] = copy.deepcopy(v)


def count_or_accumulate(count_dict: dict, items: Union[dict, Iterator[Any], Any]):
    if items is not None:
        if isinstance(items, dict):
            for k, v in items.items():
                _add_count(count_dict, k, v)
        elif isinstance(items, (list, tuple)) and isinstance(items[0], dict):
            if len(items) == 4:
                _items, accu_keys, weight_key, extra_count_field = items
            elif len(items) == 3:
                _items, accu_keys, weight_key = items
                if isinstance(weight_key, bool):
                    weight_key = None
                    extra_count_field = 'count'
                else:
                    extra_count_field = False
            elif len(items) == 2:
                _items, accu_keys = items
                weight_key = None
                extra_count_field = False
            else:
                raise ValueError("unsupported format of add-up items")

            if weight_key == 'none':
                weight_key = None

            if extra_count_field is not False and extra_count_field in accu_keys:
                raise ValueError("the extra counting field should not be in the accumulation keys")

            if weight_key is None:
                for k in accu_keys:
                    _add_count(count_dict, k, _items[k])
            else:
                weight = _items[weight_key]
                for k in accu_keys:
                    if k == weight_key:
                        _add_count(count_dict, k, weight)
                    else:
                        _add_count(count_dict, k, _items[k] * weight)

            if extra_count_field:
                if extra_count_field in count_dict:
                    count_dict[extra_count_field] += 1
                else:
                    count_dict[extra_count_field] = 1

        elif nonstr_iterable(items):
            for item in items:
                if item in count_dict:
                    count_dict[item] += 1
                else:
                    count_dict[item] = 1
        elif items in count_dict:
            count_dict[items] += 1
        else:
            count_dict[items] = 1
    return count_dict


def sum_dicts(count_dicts, in_place=False):
    if isinstance(count_dicts, dict):
        return count_dicts
    if len(count_dicts) == 1:
        return count_dicts[0]
    base_count_dict = count_dicts[0] if in_place else dict(count_dicts[0])
    for i in range(1, len(count_dicts)):
        count_or_accumulate(base_count_dict, count_dicts[i])

    return base_count_dict
# endregion
