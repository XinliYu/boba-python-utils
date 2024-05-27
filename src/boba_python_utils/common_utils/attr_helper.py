from typing import Callable, Union


def getattr_(_o, name, default):
    return getattr(_o, name, None) or default


def getattr__(_o, name, default, transform: Union[Callable, str] = None, null_set=None):
    if not null_set:
        return getattr_(_o, name, default)

    _o = getattr(_o, name, None)
    return (
        transform.format(_o) if (
                transform and isinstance(transform, str)
        )
        else (
            transform(_o)
            if callable(transform)
            else _o
        )
    ) if (
            _o and _o not in null_set
    ) else default


def hasattr_(_o, name) -> bool:
    return bool(getattr(_o, name, None))


def hasattr__(_o, name, null_set=None):
    if not null_set:
        return hasattr_(_o, name)

    _o = getattr(_o, name, None)
    return _o and _o not in null_set


def setattr_if_none_or_empty(obj, attr: str, val) -> None:
    """
    The same as the build-in function `setattr`,
    with the difference that this function only set the attribute
    if it is None or does not currently exist in the object.

    Args:
        obj: the object to set the attribute.
        attr: the name of the attribute to set.
        val: the value to set for the specified attribute.

    Examples:
        >>> class Point:
        ...     def __init__(self, x, y):
        ...         self.x, self.y = x, y
        >>> p = Point(1, None)
        >>> setattr_if_none_or_empty(p, 'x', 10)
        >>> setattr_if_none_or_empty(p, 'y', 10)
        >>> assert p.x == 1
        >>> assert p.y == 10

    """
    if not getattr(obj, attr, None):
        setattr(obj, attr, val)


def setattr_if_none_or_empty_(obj, attr: str, get_val: Callable) -> None:
    """
    The same as `setattr_if_none_or_empty`,
        but the "value" is a callable `get_val`.
    If the attribute to set is None or does not currently exist in the object,
        then the callable `get_val` is executed to compute the actual value to set.

    The purpose is avoid unnecessary computation of the value.
    Sometime the value is expensive to compute,
    and here we only compute the value if the attribute
    to set does not currently exists_path or has a `None` value.

    Examples:
        >>> class Point:
        ...     def __init__(self, x, y):
        ...         self.x, self.y = x, y
        >>> from math import factorial
        >>> p = Point(1, None)
        >>> get_val = lambda: factorial(10)
        >>> setattr_if_none_or_empty_(p, 'x', get_val)  # get_val will NOT be computed in this line
        >>> setattr_if_none_or_empty_(p, 'y', get_val)  # get_val will be computed in this line
        >>> assert p.x == 1
        >>> assert p.y == get_val()
    """

    if not getattr(obj, attr, None):
        setattr(obj, attr, get_val())
