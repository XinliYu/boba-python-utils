import inspect
from functools import reduce
from typing import Any, Callable, Mapping, Tuple, Union, List, Sequence
from typing import Iterable

from boba_python_utils.common_utils.typing_helper import iterable


def apply_arg(
        func: Callable,
        arg: Any = None,
        map_type: Union[type, Tuple[type]] = Mapping,
        seq_type: Union[type, Tuple[type]] = (list, tuple),
        allows_mixed_positional_and_named_arg: bool = False
) -> Any:
    """
    Applies a function `func` to `arg`.
    If `arg` is None, then `func` is executed without arguments.
    If `arg` is of `map_type` (default Mapping), then expand it as named arguments.
    If `arg` is of `seq_type` (default list or tuple), then expand it as positional arguments.
    Otherwise, `arg` is passed as a single argument for `func`.

    if `allows_mixed_positional_and_named_arg` is set True,
    then 'arg' must be a list or tuple of length at least 2,
    and the last element of the list or tuple can be a mapping of named arguments.

    This utility function is helpful when `func` is a function with all optional arguments,
    and we are giving flexibility to the format of the `func`'s arguments.

    Examples:
        >>> def greet(name="John", age=30):
        ...     return f"Hello, {name}! You are {age} years old."

        >>> apply_arg(greet, None)
        'Hello, John! You are 30 years old.'

        >>> apply_arg(greet, ("Alice", 25))
        'Hello, Alice! You are 25 years old.'

        >>> apply_arg(greet, {"name": "Alice", "age": 25})
        'Hello, Alice! You are 25 years old.'

        >>> apply_arg(greet, (["Alice", {"age": 25}]), allows_mixed_positional_and_named_arg=True)
        'Hello, Alice! You are 25 years old.'
    """

    if arg is not None:
        if allows_mixed_positional_and_named_arg:
            if isinstance(arg, (list, tuple)) and len(arg) > 1:
                positional_args, arg = arg[:-1], arg[-1]
                if isinstance(arg, map_type):
                    return func(*positional_args, **arg)
                elif isinstance(arg, seq_type):
                    return func(*positional_args, *arg)
                else:
                    return func(*positional_args, arg)
            else:
                raise ValueError(
                    "when 'allows_mixed_positional_and_named_arg' is set True, "
                    "'arg' must be a list or tuple of length at least 2;"
                    f"got {arg}"
                )
        else:
            if isinstance(arg, map_type):
                return func(**arg)
            elif isinstance(arg, seq_type):
                return func(*arg)
            else:
                return func(arg)
    else:
        return func()


def apply_func(
        func: Callable,
        input,
        seq_type=(list, tuple),
        mapping_type=Mapping,
        skip_if_neither_seq_or_mapping: bool = False,
        pass_seq_index: bool = False,
        pass_mapping_key: bool = False
):
    """
    Applies a function `func` to the input object.
    If `_input` is of `sequence_type`, then `func` is applied to each object in the sequence,
        and returns a sequence of the same type.
    If `_input` is of `mapping_type`, then `func` is applied to the values of the mapping,
        and returns a dictionary of the original keys
        and their corresponding values processed by `func`.
    Otherwise, applies `func` to `_input` itself if `skip_if_neither_seq_or_mapping` is set False.

    Args:
        func: the function to apply on the input object.
        input: the input object.
        seq_type: applies `func` to elements of `input`
            if `input` is of the specified sequence type.
        mapping_type: applies `func` to values of `input`
            if `input` is of the specified mapping type.
        skip_if_neither_seq_or_mapping: True to skip applying `func` if `input`
            is neither of `seq_type` or of `mapping_type`.
        pass_seq_index: True to pass sequence index to `func` as the first argument;
            if `input` is not a `seq_type` and also not a `mapping_type`,
            then `func(None, input)` is executed if this argument is True.
        pass_mapping_key: True to pass mapping key to `func` as the first argument.

    Returns: the output of applying `func` on `input`.

    """
    if isinstance(input, seq_type):
        if pass_seq_index:
            return type(input)(func(i, x) for i, x in enumerate(input))
        else:
            return type(input)(func(x) for x in input)
    elif isinstance(input, mapping_type):
        if pass_mapping_key:
            return {
                k: func(k, v) for k, v in input.items()
            }
        else:
            return {
                k: func(v) for k, v in input.items()
            }
    elif not skip_if_neither_seq_or_mapping:
        if pass_seq_index:
            return func(None, input)
        else:
            return func(input)


def has_parameter(func: Callable, para: str) -> bool:
    """
    Checks if a callable has a parameter of the specified name.

    Args:
        func: The callable to check for the parameter.
        para: The parameter name to search for.

    Returns:
        True if the callable has a parameter of the specified name; otherwise, False.

    Examples:
        >>> def example_func(a, b, c):
        ...     pass
        ...
        >>> has_parameter(example_func, 'a')
        True

        >>> has_parameter(example_func, 'x')
        False
    """
    return para in inspect.signature(func).parameters


def is_first_parameter_varpos(func: Callable) -> bool:
    """
    Checks if the first parameter of a callable is a variable positional parameter.

    Args:
        func: The callable to check for the variable positional parameter.

    Returns:
        True if the first parameter is a variable positional parameter; otherwise, False.

    Examples:
        >>> def example_func(*args, b, c):
        ...     pass
        ...
        >>> is_first_parameter_varpos(example_func)
        True

        >>> def example_func(a, b, c):
        ...     pass
        ...
        >>> is_first_parameter_varpos(example_func)
        False
    """
    return (
            next(iter(inspect.signature(func).parameters.values())).kind
            == inspect.Parameter.VAR_POSITIONAL
    )


def get_arg_names(
        func: Callable,
        include_varargs: bool = False,
        include_varkw: bool = False
) -> List[str]:
    """
    Gets the argument names of a callable.

    Args:
        func: the callable.
        include_varargs: True to include the name of the positional arguments.
        include_varkw: True to include the name of the named arguments.

    Returns: the names of arguments of the callable `func`.

    Examples:
        >>> get_arg_names(get_arg_names)
        ['func', 'include_varargs', 'include_varkw']
        >>> get_arg_names(sum)
        ['iterable', 'start']
        >>> get_arg_names(sum, include_varargs=True)
        ['iterable', 'start', None]
        >>> get_arg_names(sum, include_varargs=True, include_varkw=True)
        ['iterable', 'start', None, None]
        >>> get_arg_names(dict.__new__)
        ['type']
        >>> get_arg_names(dict.__new__, include_varargs=True, include_varkw=True)
        ['type', 'args', 'kwargs']
    """
    arg_spec = inspect.getfullargspec(func)
    out = arg_spec.args + arg_spec.kwonlyargs
    if include_varargs:
        out.append(arg_spec.varargs)
    if include_varkw:
        out.append(arg_spec.varkw)
    return out


def get_relevant_named_args(
        func: Union[Callable, Iterable[Callable]],
        include_varargs: bool = False,
        include_varkw: bool = False,
        return_other_args: bool = False,
        exclusion: Sequence[str] = None,
        **kwargs
) -> Union[Mapping, Tuple[Mapping, Mapping]]:
    """
    Extracts named arguments from `kwargs` that are relevant to the callable `func`.

    Args:
        func: the callable; if a list callable
        include_varargs: True to consider the name of the positional arguments of `func`
            when looking into `kwargs`.
        include_varkw: True to consider the name of the named arguments of `func`
            when looking into `kwargs`.
        exclusion: A sequence of argument names to be excluded from the result.
        return_other_args: returns a 2-tuple, and the second element in the tuple is the other args
            not considered relevant to `func`.
        **kwargs: extracts arguments relevant to `func` from these named arguments.

    Returns: a mapping consisting of the named arguments relevant to `func` if `return_other_args`
        is set False; otherwise, two mappings of named arguments, where the first mapping consists
        of relevant named arguments, and the second mapping consists of other named arguments.

    Examples:
        >>> get_relevant_named_args(sum, iterable=[1, 2], start=0, seed=0)
        {'iterable': [1, 2], 'start': 0}
        >>> get_relevant_named_args(sum, iterable=[1, 2], return_other_args=True, start=0, seed=0)
        ({'iterable': [1, 2], 'start': 0}, {'seed': 0})
        >>> get_relevant_named_args(
        ...    dict.__new__,
        ...    include_varargs=True,
        ...    include_varkw=False,
        ...    type=dict,
        ...    args=None,
        ...    kwargs={1:2, 3:4}
        ... )
        {'type': <class 'dict'>, 'args': None}
        >>> get_relevant_named_args(
        ...    dict.__new__,
        ...    include_varargs=True,
        ...    include_varkw=True,
        ...    type=dict,
        ...    args=None,
        ...    kwargs={1:2, 3:4}
        ... )
        {'type': <class 'dict'>, 'args': None, 'kwargs': {1: 2, 3: 4}}
    """
    if callable(func):
        arg_names = get_arg_names(
            func,
            include_varargs=include_varargs,
            include_varkw=include_varkw
        )
    elif iterable(func):
        arg_names = sum(
            (
                get_arg_names(_method, include_varargs=include_varargs, include_varkw=include_varkw)
                for _method in func
            ),
            []
        )
    else:
        raise ValueError("'func' must be a callable, or an iterable of callables")
    related_args = {
        k: v
        for k, v in kwargs.items()
        if k in arg_names and (not exclusion or k not in exclusion)
    }

    if return_other_args:
        unrelated_args = {
            k: v
            for k, v in kwargs.items()
            if k not in arg_names and (not exclusion or k not in exclusion)
        }
        return related_args, unrelated_args
    else:
        return related_args


def compose2(func2: Callable, func1: Callable) -> Callable:
    """
    A composition of two functions.

    Examples:
        >>> def f1(x):
        >>>     return x + 2
        >>> def f2(x):
        >>>     return x * 2
        >>> assert compose2(f2, f1)(5) == 14  # f2(f1(5))

    Args:
        func2: the outside function.
        func1: the inside function.
    Returns:
        the composed function.
    """

    def _composed(*args, **kwargs):
        return func2(func1(*args, **kwargs))

    return _composed


def compose(*funcs: Callable) -> Callable:
    """
    A composition of multiple functions.

    Examples:
        >>> from timeit import timeit
        >>> def f1(x):
        >>>     return x + 2
        >>> def f2(x):
        >>>     return x * 2
        >>> assert compose(f2, f1)(5) == 14

        # `compose2` is faster to compose two functions
        >>> def target1():
        >>>     compose2(f2, f1)(5)
        >>> def target2():
        >>>     compose(f2, f1)(5)
        >>> print(timeit(target1)) # about 30% faster
        >>> print(timeit(target2))

    Args:
        funcs: the functions to compose.
    Returns:
        the composed function
    """
    return reduce(compose2, funcs)


def get_func_name() -> str:
    return inspect.stack()[1][3]


def get_func_caller_name() -> str:
    return inspect.stack()[2][3]


def is_bounded_callable(f: Callable) -> bool:
    return hasattr(f, "__self__")


# region apply processors

def get_processor(processor_name: str, modules: Iterable[Any] = None, processors: Mapping = None) -> Callable:
    """
    Get a processor function based on its name.

    Args:
        processor_name (str): The name of the processor function to retrieve.
        modules (Iterable[Any], optional): An iterable containing modules to search for the processor function. Defaults to None.
        processors (Mapping, optional): A dictionary mapping processor names to functions. Defaults to None.

    Returns:
        Callable: The processor function.

    Raises:
        ValueError: If the specified processor name is not found.

    Example:
        >>> get_processor('strip', modules=[str])
        <method 'strip' of 'str' objects>
    """
    if processors is not None and processor_name in processors:
        processor = processors[processor_name]
        if callable(processor):
            return processor
    else:
        buildins = globals()['__builtins__']
        if processor_name in buildins:
            processor = buildins[processor_name]
            if callable(processor):
                return processor
        for module in modules:
            if hasattr(module, processor_name):
                processor = getattr(module, processor_name)
                if callable(processor):
                    return processor

    raise ValueError(f"processor '{processor_name}' not found")


def process(obj: Any, modules: Iterable[Any] = None, processors: Mapping = None, output_as_arg_place_holder: str = '#output', **kwargs) -> Any:
    """
    Apply processing functions to the input object.

    Args:
        obj (Any): The input object to be processed.
        modules (Iterable[Any], optional): An iterable containing modules to search for processor functions. Defaults to None.
        processors (Mapping, optional): A dictionary mapping processor names to functions. Defaults to None.
        output_as_arg_place_holder (str, optional): Placeholder string used to indicate the output argument position. Defaults to '#output'.
        **kwargs: Keyword arguments where the key is the processor name and the value is either True, False,
            a dictionary of processor arguments, or directly the arguments.

    Returns:
        Any: The processed object.


    Example:
        >>> s = "   Hello, World!   "
        >>> process(s, modules=[str], strip=True)
        'Hello, World!'
        >>> process(s, modules=[str], strip=True, lower=True)
        'hello, world!'
        >>> s = "   Hello, World!   xxx"
        >>> process(s, modules=[str], rstrip=' x')
        '   Hello, World!'
        >>> process(s, modules=[str], isdigit=True)
        False
        >>> my_list = [1, 2, 3, 4, 5]
        >>> process(my_list, reverse=True)
        [5, 4, 3, 2, 1]
        >>> my_dict = {'a': 1, 'b': 2, 'c': 3}
        >>> process(my_dict, items=True)
        dict_items([('a', 1), ('b', 2), ('c', 3)])
    """
    if modules is None and processors is None:
        modules = [type(obj), obj]
    for processor_name, processor_args in kwargs.items():
        processor = get_processor(
            processor_name=processor_name,
            modules=modules,
            processors=processors
        )
        _obj = None
        if processor_args is True:
            if is_bounded_callable(processor):
                _obj = processor()
            else:
                _obj = processor(obj)

        elif isinstance(processor_args, Mapping):
            if output_as_arg_place_holder in processor_args.values():
                processor_args = {
                    k: v if v == output_as_arg_place_holder else v
                    for k, v in processor_args.items()
                }

            if is_bounded_callable(processor):
                _obj = processor(**processor_args)
            else:
                _obj = processor(obj, **processor_args)
        elif isinstance(processor_args, List):
            if output_as_arg_place_holder in processor_args:
                processor_args = [
                    obj if v == output_as_arg_place_holder else v
                    for v in processor_args
                ]
                _obj = processor(*processor_args)
            else:
                if is_bounded_callable(processor):
                    _obj = processor(*processor_args)
                else:
                    _obj = processor(obj, *processor_args)
        else:
            if is_bounded_callable(processor):
                _obj = processor(processor_args)
            else:
                _obj = processor(obj, processor_args)
        if _obj is not None:
            obj = _obj
    return obj

# endregion
