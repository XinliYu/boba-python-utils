import itertools
from typing import Callable
from typing import Dict, List, Tuple, Union

def parse_args_from_str(
    s: str,
    main_arg_sep: str = '@',
    arg_sep: str = '|',
    named_arg_sep: str = ':'
) -> Tuple[str, Union[None, List[str]], Union[None, Dict[str, str]]]:
    """
    Parses string arguments for a string in a specified format.

    Args:
        s: The input string to parse.
        main_arg_sep: Separator for main arguments (default: '@').
        arg_sep: Separator for unnamed arguments (default: '|').
        named_arg_sep: Separator for named arguments (default: ':').

    Returns:
        A tuple containing the main argument, a list of unnamed arguments, and a dictionary of named arguments.

    Examples:
        >>> s = "csv"
        >>> parse_args_from_str(s)
        ('csv', None, None)

        >>> s = "csv@Name|Gender|Age"
        >>> parse_args_from_str(s)
        ('csv', ['Name', 'Gender', 'Age'], {})

        >>> s = "csv@Name|Gender|Age|sep:,"
        >>> parse_args_from_str(s)
        ('csv', ['Name', 'Gender', 'Age'], {'sep': ','})
    """

    if main_arg_sep:
        ss = s.split(main_arg_sep, maxsplit=1)
        if len(ss) == 2:
            main_arg, s_other_args = ss
        else:
            main_arg = s
            return main_arg, None, None
    else:
        main_arg = None
        s_other_args = s

    args = s_other_args.split(arg_sep)

    unnamed_args = []
    named_args = {}
    for arg in args:
        if arg:
            if named_arg_sep in arg:
                k, v = arg.split(named_arg_sep, maxsplit=1)
                named_args[k] = v
            else:
                unnamed_args.append(arg)
    return main_arg, unnamed_args, named_args


def apply_arg_combo(
        method: Callable,
        *args,
        unpack_for_single_result=False,
        return_first_only=False,
        **kwargs
):
    """
    Applies combinations of arguments to callable `method`.

    Args:
        method: the callable.
        *args: the positional arguments;
            multiple arguments for a position must be specified in a list or a tuple.
        unpack_for_single_result: indicates the return format
            when there is only one argument combination that produces only one result;
            True to return the single result itself;
            False to return the single-element list that contains the only result.
        return_first_only: True to only return the result of the first argument combination;
            False to return all results of all argument combination in a list.
        **kwargs: the name arguments;
            multiple arguments for a name must be specified in a list or a tuple.

    Returns: results of applying the argument combinations to the callable `method`.

    Examples:
        >>> def foo(x, y):
        ...     return x * y
        >>> apply_arg_combo(foo, [1, 2], [3, 4])
        [3, 4, 6, 8]
        >>> apply_arg_combo(foo, [1, 2], [3, 4], return_first_only=True)
        3
        >>> apply_arg_combo(foo, [1], [3], unpack_for_single_result=False)
        [3]
        >>> apply_arg_combo(foo, [1], [3], unpack_for_single_result=True)
        3
        >>> apply_arg_combo(foo, x=[1, 2], y=[3, 4])
        [3, 4, 6, 8]
    """

    output = []
    args = list(args)
    for arg_idx, arg in enumerate(args):
        if type(arg) not in (list, tuple):
            args[arg_idx] = (arg,)

    for arg_key, arg in kwargs.items():
        if type(arg) not in (list, tuple):
            kwargs[arg_key] = (arg,)

    if args:
        for a1 in itertools.product(*args):
            if kwargs:
                for a2 in itertools.product(*kwargs.values()):
                    this_arg = method(*a1, **dict(zip(kwargs, a2)))
                    if return_first_only:
                        return this_arg
                    output.append(this_arg)
            else:
                this_arg = method(*a1)
                if return_first_only:
                    return this_arg
                output.append(this_arg)
    elif kwargs:
        for a2 in itertools.product(*kwargs.values()):
            this_arg = method(**dict(zip(kwargs, a2)))
            if return_first_only:
                return this_arg
            output.append(this_arg)
    else:
        this_arg = method()
        if return_first_only:
            return this_arg
        output.append(this_arg)

    if unpack_for_single_result and len(output) == 1:
        return output[0]
    return output



