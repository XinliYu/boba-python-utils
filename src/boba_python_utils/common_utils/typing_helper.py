import json
from ast import literal_eval
from typing import Mapping, Callable, Union, List, Tuple, ClassVar, Any, Optional, Set, Iterable


# region type checking

def is_str(_obj) -> bool:
    """
    Convenience function to check if an object is a string.

    Examples:
        >>> assert not is_str(None)
        >>> assert is_str('')
        >>> assert is_str('1')
        >>> assert not is_str(1)

    """
    return isinstance(_obj, str)


def is_none_or_empty_str(_obj) -> bool:
    """
    Checks if an object is None or an empty string.

    Examples:
        >>> assert is_none_or_empty_str(None)
        >>> assert is_none_or_empty_str('')
        >>> assert not is_none_or_empty_str('1')
        >>> assert not is_none_or_empty_str(1)

    """
    return _obj is None or (isinstance(_obj, str) and _obj == '')


def str_eq(s1, s2) -> bool:
    """
    Checks if two strings equal each other. `None` is considered equal to an empty string.

    Examples:
        >>> assert str_eq(None, None)
        >>> assert str_eq('', None)
        >>> assert str_eq(None, '')
        >>> assert str_eq('1', '1')
        >>> assert not str_eq('1', None)
        >>> assert not str_eq('1', 1)
    """
    if s1 is None:
        return is_none_or_empty_str(s2)
    if s2 is None:
        return is_none_or_empty_str(s1)
    return isinstance(s1, str) and isinstance(s2, str) and s1 == s2


def is_class(_obj) -> bool:
    """
    Check whether or not an object is a class/type.

    Examples:
        >>> assert not is_class(1)
        >>> assert is_class(int)
    """
    return isinstance(_obj, type)


def is_basic_type(_obj):
    """
    Check whether an object is a python int/float/str/bool.

    Examples:
        >>> assert is_basic_type(1)
        >>> assert not is_basic_type([])
    """
    return isinstance(_obj, (int, float, str, bool))


def is_basic_type_iterable(_obj, iterable_type=(list, tuple, set)):
    """
    Check whether an object is an iterable of python int/float/str/bool.
    If `iterable_type` is specified, then the iterable object itself must be of the specified type.

    Examples:
        >>> assert not is_basic_type_iterable(1)
        >>> assert is_basic_type_iterable([1,2,3,4])
        >>> assert is_basic_type_iterable(['1','2','3','4'])
        >>> assert not is_basic_type_iterable(['1','2','3','4'], iterable_type=tuple)
    """
    if iterable_type is None:
        return iterable__(_obj) and all(is_basic_type(x) for x in _obj)
    else:
        return isinstance(_obj, iterable_type) and all(is_basic_type(x) for x in _obj)


def is_basic_type_or_basic_type_iterable(_obj, iterable_type=(list, tuple, set)):
    """
    Check whether an object is a python int/float/str/bool,
    of if the object is an iterable of python int/float/str/bool.

    See Also :func:`is_basic_type` and :func:`is_basic_type_iterable`.

    """
    return is_basic_type(_obj) or is_basic_type_iterable(_obj, iterable_type=iterable_type)


def element_type(_container, atom_types=(str,), key=0):
    """
    Gets the type of an atomic element in the provided object container.
    The container may be nested, and we use `key` to
        recursively retrieve an atomic element inside container.

    When using this function, we usually assuem the elements in the container are of the same type,
    so the retrieved element type can represent the type of all elements in the container.

    Args:
        _container: the container holding values.
        atom_types: the types that should be treated as an atomic object.
        key: the key used to recursively retrieve the inside containers.
    Returns:
        the type of an atomic element in the container.

    Examples:
        >>> assert element_type(((True, False), (True, False))) is bool
        >>> assert element_type([[['a', [0, 1, 2]], 'c']]) is str
        >>> assert element_type(['a', [0, 1, 2], 'c'], key=1) is int
    """

    try:
        while not isinstance(_container, atom_types):
            _container = _container[key]
    except:
        pass
    return type(_container)


def iterable(_obj) -> bool:
    """
    Check whether or not an object can be iterated over.

    Examples:
        >>> assert iterable([1, 2, 3, 4])
        >>> assert iterable(iter(range(5)))
        >>> assert iterable('123')
        >>> assert not iterable(123)
    """
    try:
        iter(_obj)
    except TypeError:
        return False
    return True


def iterable__(_obj, atom_types=(str,)):
    """
    A variant of `iterable` that considers types in `atom_types` as non-iterable.
    Returns `True` if the type of `obj` is not in the `atom_types`, and it is iterable.
    By default, the `atom_types` conssits of the string type.

    Examples:
        >>> assert iterable__('123', atom_types=None)
        >>> assert not iterable__('123')
        >>> assert iterable__((1, 2 ,3))
        >>> assert not iterable__((1, 2, 3), atom_types=(tuple,str))
    """
    return not (atom_types and isinstance(_obj, atom_types)) and iterable(_obj)


def nonstr_iterable(_obj) -> bool:
    """
    Checks whether the object is an iterable but is not a string.
    Equivalent to `iterable__(x, atom_types=(str,))`.

    >>> assert not nonstr_iterable('123')
    >>> assert nonstr_iterable([1,2,3])
    """
    return (not is_str(_obj)) and iterable(_obj)


def is_named_tuple(obj) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, '_fields')


def sliceable(_obj):
    """
    Checks whether an object can be sliced.
    Examples:
        >>> sliceable(2)
        False
        >>> sliceable(None)
        False
        >>> sliceable((1, 2, 3))
        True
        >>> sliceable('abc')
        True
        >>> sliceable([])
        True
    """
    if _obj is None:
        return False
    if not hasattr(_obj, '__getitem__'):
        return False
    try:
        _obj[0:1]
    except:
        return False
    return True


def of_type_all(_it, _type):
    """
    Checks if every iterated object is of the specified type.

    Examples:
        >>> assert of_type_all([1, 2, 3], int)
        >>> assert not of_type_all([1, 2, '3'], int)
    """
    if isinstance(_it, _type):
        return True
    if iterable(_it):
        return all(isinstance(item, _type) for item in _it)
    return False


def all_str(_it):
    """
    Checks if every iterated object is a python string.

    Examples:
        >>> assert all_str(['1', '2', '3'])
        >>> assert not all_str([1, 2, '3'])
    """
    return of_type_all(_it, str)


# endregion

# region type conversion
def str_(x) -> str:
    """
    Converting an object to string using format.

    This function is particularly helpful for str Enum objects,
    returning the literal string value of the Enum.

    Examples:
        >>> from enum import Enum
        >>> class Opt(str, Enum):
        ...   A = 'a'
        ...   B = 'b'
        >>> print(str(Opt.A))
        Opt.A
        >>> print(str_(Opt.A))
        a

    """
    return '{}'.format(x)


def make_list(_x, atom_types=(str,), str_sep=None, ignore_none: bool = False):
    """
    Converts a possible iterable object to a list if it is considered not an iterable
        (any type in `atom_types` is not considered as an iterable);
        otherwise make it a single-object list.
    If the input object is already a list, then the original object is returned.

    If the input object is a string and `str_sep` is provided (not False or None), then
        1. if `str_sep` is True, then return a list substrings of the input string,
            split by white spaces;
        2. otherwise, return a list substrings of the input string, split by `str_sep`.
        Empty splits are ignored and all substrings are striped of leading and ending white spaces.

    Examples:
        >>> assert make_list(None) == [None]
        >>> assert make_list(None, ignore_none=True) is None
        >>> assert make_list(3) == [3]
        >>> assert make_list((1,2,3)) == [1,2,3]
        >>> assert make_list({1, 2, 3}) == [1,2,3]
        >>> assert make_list('123') == ['123']
        >>> assert make_list(('123', '456')) == ['123', '456']
        >>> assert make_list('123', atom_types=None) == ['1','2','3']
        >>> assert make_list('1 2 3', str_sep=True) == ['1','2','3']
        >>> assert make_list('1,2,3', str_sep=',') == ['1','2','3']
    """

    if (
            (str_sep is not None) and
            (str_sep is not False) and
            isinstance(_x, str)
    ):
        if str_sep is True:
            _x = _x.split()
        else:
            _x = (__x.strip() for __x in _x.split(str_sep))
        return list(filter(None, _x))

    if ignore_none and _x is None:
        return None

    if isinstance(_x, list):
        return _x
    if iterable__(_x, atom_types=atom_types):
        return list(_x)
    else:
        return [_x]


def make_tuple(_x, atom_types=(str,), str_sep=None, ignore_none: bool = False):
    """
    Same as `make_list`, except that it returns a tuple.

    Examples:
        >>> assert make_tuple(None) == (None, )
        >>> assert make_tuple(None, ignore_none=True) is None
        >>> assert make_tuple(3) == (3,)
        >>> assert make_tuple((1,2,3)) == (1,2,3)
        >>> assert make_tuple({1, 2, 3}) == (1,2,3)
        >>> assert make_tuple('123') == ('123',)
        >>> assert make_tuple(('123', '456')) == ('123', '456')
        >>> assert make_tuple('123', atom_types=None) == ('1','2','3')
        >>> assert make_tuple('1 2 3', str_sep=True) == ('1','2','3')
        >>> assert make_tuple('1,2,3', str_sep=',') == ('1','2','3')
    """

    if (
            (str_sep is not None) and
            (str_sep is not False) and
            isinstance(_x, str)
    ):
        if str_sep is True:
            _x = _x.split()
        else:
            _x = (__x.strip() for __x in _x.split(str_sep))
        return tuple(filter(None, _x))

    if ignore_none and _x is None:
        return None

    if isinstance(_x, tuple):
        return _x
    if iterable__(_x, atom_types=atom_types):
        return tuple(_x)
    else:
        return _x,


def make_set(_x, atom_types=(str,), str_sep=None, ignore_none: bool = False):
    """
    Same as `make_list`, except that it returns a set.

    Examples:
        >>> assert make_set(None) == {None}
        >>> assert make_set(None, ignore_none=True) is None
        >>> assert make_set(3) == {3}
        >>> assert make_set((1,2,3)) == {1,2,3}
        >>> assert make_set({1, 2, 3}) == {1,2,3}
        >>> assert make_set('123') == {'123'}
        >>> assert make_set(('123', '456')) == {'123', '456'}
        >>> assert make_set(('123', '123')) == {'123'}
        >>> assert make_set('123', atom_types=None) == {'1','2','3'}
        >>> assert make_set('1 2 3', str_sep=True) == {'1','2','3'}
        >>> assert make_set('1,2,3', str_sep=',') == {'1','2','3'}
    """

    if (
            (str_sep is not None) and
            (str_sep is not False) and
            isinstance(_x, str)
    ):
        if str_sep is True:
            _x = _x.split()
        else:
            _x = (__x.strip() for __x in _x.split(str_sep))
        return set(filter(None, _x))

    if ignore_none and _x is None:
        return None

    if isinstance(_x, set):
        return _x
    if iterable__(_x, atom_types=atom_types):
        return set(_x)
    else:
        return {_x}


def make_list_(_x: Any, non_atom_types=(tuple, set)):
    """
    If the object `_x` is a type of `non_atom_types` (tuple or a set by default),
        then turn it into a list.
    If the object `_x` is a list, then returns list instance itself.
    Otherwise, returns `[_x]`.

    Unlike `make_list`, this method specifies `non_atom_types` rather than `atom_types`.

    Examples:
        >>> assert make_list_(3) == [3]
        >>> assert make_list_((1,2,3)) == [1,2,3]
        >>> assert make_list_({1, 2, 3}) == [1,2,3]
        >>> assert make_list_('123', non_atom_types=(str, tuple, set)) == ['1','2','3']

    """

    if isinstance(_x, non_atom_types):
        return list(_x)
    elif isinstance(_x, list):
        return _x
    else:
        return [_x]


def make_list_if_not_none_(
        _x: Any,
        non_atom_types: Tuple[type, ...] = (tuple, set)
) -> Optional[List[Any]]:
    """
    If the input is None, returns None;
    otherwise, call :func:`make_list_` function to convert `_x` into a list.

    Examples:
        >>> make_list_if_not_none_(None)
        >>> make_list_if_not_none_(('a', 'b'))
        ['a', 'b']
        >>> make_list_if_not_none_({'a', 'b'})
        ['a', 'b']
        >>> make_list_if_not_none_('a')
        ['a']
    """
    if _x is None:
        return
    return make_list_(_x, non_atom_types=non_atom_types)


def make_tuple_(_x, non_atom_types=(list, set)):
    """
    If the object `_x` is a type of `non_atom_types` (list or a set by default),
        then turn it into a tuple.
    If the object `_x` is a tuple, then returns list instance itself.
    Otherwise, returns `(_x,)`.

    Unlike `make_tuple`, this method specifies `non_atom_types` rather than `atom_types`.

    Examples:
        >>> assert make_tuple_(3) == (3,)
        >>> assert make_tuple_([1,2,3]) == (1,2,3)
        >>> assert make_tuple_({1, 2, 3}) == (1,2,3)
        >>> assert make_tuple_('123', non_atom_types=(str, list, set)) == ('1','2','3')

    """
    if isinstance(_x, non_atom_types):
        return tuple(_x)
    elif isinstance(_x, tuple):
        return _x
    else:
        return _x,


def make_set_(_x, non_atom_types=(tuple, list)):
    """
    If the object `_x` is a type of `non_atom_types` (list or a tuple by default),
        then turn it into a set.
    If the object `_x` is a et, then returns list instance itself.
    Otherwise, returns `{_x}`.

    Examples:
        >>> assert make_set_(3) == {3}
        >>> assert make_set_([1,2,3]) == {1,2,3}
        >>> assert make_set_([1,2,2]) == {1,2}
        >>> assert make_set_({1, 2, 3}) == {1,2,3}
        >>> assert make_set_('123', non_atom_types=(str, list, set)) == {'1','2','3'}

    """
    if isinstance(_x, non_atom_types):
        return set(_x)
    elif isinstance(_x, set):
        return _x
    else:
        return {_x}


def get_iter(
        input_path_or_iterable: Union[str, Iterable],
        default_iterator: Optional[Callable] = None
) -> Iterable:
    """
    Returns an iterator from the given input. The input can be a string representing a file path or an
    iterable object. If the input is a string and `default_iterator` is not provided, a predefined iterator
    function is used to read and iterate over the file. If `default_iterator` is provided, it is used to
    create an iterator from the string.

    Args:
        input_path_or_iterable: A string representing a file path or an iterable object.
        default_iterator: An optional callable that takes a string and returns an iterator.
                          If None, a predefined iterator function is used for file paths.

    Returns:
        An iterator derived from the input.

    Raises:
        ValueError: If the function is unable to create an iterator from the given input.

    Examples:
        >>> # Example with a file path and default json iterator
        >>> for item in get_iter('path/to/file.json'):
        ...     print(item)

        >>> # Example with a custom iterator function
        >>> def custom_iterator(path):
        ...     # Custom logic to iterate over the file
        ...     yield from custom_logic(path)
        >>> for item in get_iter('path/to/other_file.txt', custom_iterator):
        ...     print(item)

        >>> # Example with an iterable object
        >>> data = [1, 2, 3, 4]
        >>> for item in get_iter(data):
        ...     print(item)
    """
    if isinstance(input_path_or_iterable, str):
        if default_iterator is None:
            from boba_python_utils.io_utils.json_io import iter_json_objs
            return iter_json_objs(input_path_or_iterable)
        else:
            _iterable = default_iterator(input_path_or_iterable)
    else:
        _iterable = input_path_or_iterable

    if iterable(_iterable):
        return _iterable
    else:
        raise ValueError(f"Unable to obtain data iter from '{input_path_or_iterable}'")


def enumerate_(_x):
    try:
        return enumerate(_x)
    except:
        return enumerate((_x,))


_STRS_TRUE = {'true', 'yes', 'y', 'ok', '1'}
_STRS_FALSE = {'false', 'no', 'n', '0'}


def str2bool(s: str):
    """
    Converts a string to a boolean value.

    Args:
        s: The input string.

    Returns:
        The boolean value corresponding to the input string.
        If the string is one of 'true', 'yes', 'y', 'ok', or '1' (case-insensitive),
        the function returns True. Otherwise, it returns False.

    Examples:
        >>> str2bool("True")
        True

        >>> str2bool("No")
        False
    """
    return s.lower() in _STRS_TRUE


def bool_(x):
    """
    Converts a value to a boolean.

    Args:
        x: The input value. It can be of any type.

    Returns:
        The boolean value corresponding to the input.
        If the input is a string, it will be converted using `str2bool()`.
        Otherwise, the input will be cast to a boolean.

    Examples:
        >>> bool_("yes")
        True

        >>> bool_(0)
        False
    """
    if isinstance(x, str):
        return str2bool(x)
    else:
        return bool(x)


def map_iterable_elements(
        _iterable,
        _converter: Callable,
        atom_types=(str,)
):
    """
    Maps every elements in the provided iterable by applying the converter.
    Elements of type `atom_types` will not be treated as iterables.

    Examples:
        >>> map_iterable_elements([1, 2, 3], str)
        ['1', '2', '3']
        >>> map_iterable_elements('123', int)
        123
        >>> map_iterable_elements(
        ...    [{(1,2), (3,4)}, [[1, 2]], '123'], tuple, atom_types=(str, set)
        ... )
        [((1, 2), (3, 4)), [(1, 2)], ('1', '2', '3')]

    """
    try:
        if atom_types and isinstance(_iterable, atom_types):
            return _converter(_iterable)
        else:
            return type(_iterable)(
                map_iterable_elements(x, _converter, atom_types=atom_types)
                for x in _iterable
            )
    except:
        pass
    return _converter(_iterable)


def str2val_(s: str, str_format: Union[str, Callable[[str], str]] = None, success_label=False):
    """
    Parses a string as its likely equivalent value.
    Typically tries to convert to integers, floats, bools, lists, tuples, dictionaries.

    Args:
        s: the string to parse as a value.
        str_format: a formatting string,
            or a callable that takes a string and outputs a processed string.
        success_label: returns a tuple, with the first being the parsed value,
                and the second being a boolean value indicating if the the parse is successful.

    Returns: the parsed value if `success_label` is `False`,
        or a tuple with the second being the parse success flag if `success_label` is `True`.

    Examples:
        >>> assert str2val_('1') == 1
        >>> assert str2val_('2.554') == 2.554
        >>> assert str2val_("[1, 2, 'a', 'b', False]") == [1, 2, 'a', 'b', False]
        >>> assert str2val_("1, 2, 'a', 'b', False", str_format='[{}]') == [1, 2, 'a', 'b', False]
    """
    ss = s.strip()
    if not ss:
        return ss

    if str_format:
        if is_str(str_format):
            ss = str_format.format(ss)
        elif callable(str_format):
            ss = str_format(ss)
            if not is_str(ss):
                raise ValueError(f"'str_format' must outputs a string; got {ss}")
        else:
            raise ValueError("'str_format' must be a formatting string "
                             "or a callable that takes a string and outputs a processed string; "
                             f"got {str_format}")

    if success_label:
        def _literal_eval():
            try:
                return literal_eval(ss), True
            except:  # noqa: E722
                return s, False

        if ss[0] == '{':
            try:
                return json.loads(ss), True
            except:  # noqa: E722
                return _literal_eval()
        elif ss[0] == '[' or ss[0] == '(':
            return _literal_eval()
        else:
            try:
                return int(ss), True
            except:  # noqa: E722
                try:
                    return float(ss), True
                except:  # noqa: E722
                    sl = ss.lower()
                    if sl in _STRS_TRUE:
                        return True, True
                    elif sl in _STRS_FALSE:
                        return False, True
                    else:
                        try:
                            return literal_eval(ss), True
                        except:  # noqa: E722
                            return s, False
    else:
        def _literal_eval():
            try:
                return literal_eval(ss)
            except:  # noqa: E722
                return s

        if ss[0] == '{':
            try:
                return json.loads(ss)
            except:  # noqa: E722
                return _literal_eval()
        elif ss[0] == '[' or ss[0] == '(':
            return _literal_eval()
        else:
            try:
                return int(ss)
            except:  # noqa: E722
                try:
                    return float(ss)
                except:  # noqa: E722
                    sl = ss.lower()
                    if sl in _STRS_TRUE:
                        return True
                    elif sl in _STRS_FALSE:
                        return False
                    else:
                        try:
                            return literal_eval(ss)
                        except:  # noqa: E722
                            return s


def solve_obj(
        _input: Union[str, Mapping, List, Tuple, Any],
        obj_type: ClassVar = None,
        str2obj: Callable = str2val_,
        str_format: str = None,
):
    """
    Solves a string, Mapping, list or tuple as an object.

    Args:
        _input: the input, must be a string, Mapping, list or tuple.
            1. if this is a string, then `str2obj` is used to convert the input.
            2. if this is a Mapping or a tuple or a list,
                then `obj_type` is used to create the object,
                assuming the mapping consists of named arguments for the `obj_type`,
                or the tuple/list consists of arguments for the `obj_type`;
            3. otherwise, try parsing the input as `obj_type`
                assuming it is the sole argument for `obj_type`.
        obj_type: a class variable used to create object from a Mapping, list, or tuple.
        str2obj: a callable to convert a string input as the object; by default we use `str2val_`.
        str_format: providers a format as the second argument of `str2obj` if necessary.

    Returns: the parsed object.

    Examples:
        >>> assert solve_obj('1') == 1
        >>> assert solve_obj('2.554') == 2.554
        >>> assert solve_obj("[1, 2, 'a', 'b', False]") == [1, 2, 'a', 'b', False]
        >>> assert solve_obj("1, 2, 'a', 'b', False", str_format='[{}]') == [1, 2, 'a', 'b', False]

        >>> class A:
        ...    __slots__ = ('a', 'b')
        ...    def __init__(self, a, b):
        ...        self.a = a
        ...        self.b = b
        >>> obj = solve_obj("(1,2)", A)
        >>> obj.a
        1
        >>> obj.b
        2
        >>> obj = solve_obj("{'a': 2, 'b': 1}", A)
        >>> obj.a
        2
        >>> obj.b
        1
    """
    _ori_input = _input
    if isinstance(_input, str):
        if str2obj is None:
            raise ValueError("'str2obj' must be provided to parse a string input")
        if not callable(str2obj):
            raise ValueError(f"'str2obj' must be a callable; got {type(str2obj)}")

        obj = str2obj(_input) if str_format is None else str2obj(_input, str_format)
        if obj_type is None or isinstance(obj, obj_type):
            return obj
        else:
            _input = obj

    if obj_type is None:
        raise ValueError("'obj_type' must be provided to parse a non-string input")
    if not callable(obj_type):
        raise ValueError(f"the 'obj_type' must be a callable; got {type(obj_type)}")
    if isinstance(_input, Mapping):
        return obj_type(**_input)
    elif isinstance(_input, (list, tuple)):
        return obj_type(*_input)
    elif is_class(obj_type) and isinstance(_input, obj_type):
        return _input
    else:
        try:
            return obj_type(_input)
        except:  # noqa: E722
            raise ValueError(f"cannot parse '{_ori_input}' as {obj_type}")


def solve_nested_singleton_tuple_list(x, atom_types=(str,)) -> Union[Tuple, List]:
    """
    Resolving nested singleton list/tuple. For example, resolving `[[0,1,2]]` as `[0,1,2]`.

    Examples:
        >>> solve_nested_singleton_tuple_list([[0, 1, 2]])
        [0, 1, 2]
        >>> solve_nested_singleton_tuple_list([[[0, 1, 2]]])
        [0, 1, 2]
        >>> solve_nested_singleton_tuple_list([0, 1, 2])
        [0, 1, 2]
        >>> solve_nested_singleton_tuple_list([([0, 1, 2],)])
        [0, 1, 2]
    """

    while isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]
        else:
            return x

    # unpacks the element `x` as a tuple if it is considered iterable
    if iterable__(x, atom_types):
        return tuple(x)

    # otherwise, returns `x` as a singleton tuple
    return x,


def solve_atom(x, atom_types=(str,), raise_error_if_cannot_resolve_an_atom: bool = False):
    """
    Resolves an atomic object from the given input, potentially nested in singleton lists or tuples.

    Args:
        x (Any): The input object, potentially nested in singleton lists or tuples.
        atom_types (Tuple[type]): A tuple of types considered as atomic objects (default: (str,)).
        raise_error_if_cannot_resolve_an_atom (bool): If True, raises a ValueError when the function
            cannot resolve an atomic object from the input (default: False).

    Returns:
        Any: The resolved atomic object, or the original input if it cannot be resolved.

    Examples:
        >>> solve_atom([[[[1]]]])
        1
        >>> solve_atom((("a",)))
        'a'
        >>> solve_atom([[[1, 2]]])
        [1, 2]
        >>> solve_atom("abc", atom_types=(str,))
        'abc'
        >>> solve_atom("abc", atom_types=(int,))
        ('a', 'b', 'c')
    """
    _x = solve_nested_singleton_tuple_list(x, atom_types=atom_types)
    if isinstance(_x, (list, tuple)) and len(_x) == 1:
        return _x[0]
    else:
        if raise_error_if_cannot_resolve_an_atom:
            raise ValueError(f"cannot resolve an atomic object from '{x}'")
        else:
            return _x


def solve_key_value_pairs(*kvs, parse_seq_as_alternating_key_value: bool = True):
    """
    Solves the input argument(s) as a sequence of key-value pairs. The input can be
        1. a single Mapping; then it returns an iterator through the items in the mapping;
        2. a list/tuple of 2-tuples;
        3. a sequence of elements; if :param:`parse_seq_as_alternating_key_value` is True,
            then the item at the position of even index as the key,
            and the following item at the position of odd index as the value;
            otherwise each element in the sequence will be duplicated as a tuple.

    This function does not perform thorough error checking; and if all parsing fails,
    the original input is returned.

    Examples:
        >>> solve_key_value_pairs()
        ()
        >>> solve_key_value_pairs(None)
        ()
        >>> solve_key_value_pairs(())
        ()
        >>> solve_key_value_pairs('a')
        (('a', 'a'),)
        >>> solve_key_value_pairs({'a': 1, 'b': 2})
        dict_items([('a', 1), ('b', 2)])
        >>> solve_key_value_pairs([{'a': 1, 'b': 2}])
        dict_items([('a', 1), ('b', 2)])
        >>> solve_key_value_pairs(('a', 1), ('b', 2))
        (('a', 1), ('b', 2))
        >>> tuple(solve_key_value_pairs(
        ...   'a', 1,
        ...   'b', 2
        ... ))
        (('a', 1), ('b', 2))
        >>> tuple(solve_key_value_pairs(
        ...   'a', 1,
        ...   'b', 2,
        ...   parse_seq_as_alternating_key_value=False
        ... ))
        (('a', 'a'), (1, 1), ('b', 'b'), (2, 2))
    """
    if len(kvs) == 1:
        # when a single object is passed in
        if kvs[0] is None:
            return ()  # when a single None is passed in
        elif isinstance(kvs[0], Mapping):
            return kvs[0].items()  # when a single mapping is passed in
        elif iterable__(kvs[0]):
            return solve_key_value_pairs(
                *kvs[0],
                parse_seq_as_alternating_key_value=parse_seq_as_alternating_key_value
            )  # recursively resolve the single iterable object
        else:
            # resolve the single non-iterable object as a tuple
            return (kvs[0], kvs[0]),
    elif isinstance(kvs, (list, tuple)) and kvs:
        if isinstance(kvs[0], (list, tuple)) and len(kvs[0]) == 2:
            return kvs  # assume it is a list of 2-tuples
        else:
            if parse_seq_as_alternating_key_value:
                # assume it is a sequence of items, with the item at the even position as the key,
                # and the following item at the odd position as the value
                def _it(_kvs):
                    for i in range(0, len(_kvs), 2):
                        yield _kvs[i], _kvs[i + 1]

                return _it(kvs)
            else:
                # otherwise, each element as both key and value
                return ((x, x) for x in kvs)

    return kvs

# endregion
