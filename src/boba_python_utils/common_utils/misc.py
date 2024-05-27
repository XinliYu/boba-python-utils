def divide_(x, y, default=0):
    if y == 0:
        return default

    return x / y


def is_close(x, y, tolerance=0.0001, is_tolerance_ratio: bool = True):
    """
    Checks if two numbers are considered close given the tolerance.
    The `tolerance` can be a ratio or abosoute value,
        set `is_tolerance_ratio` to change the behavior.

    Examples:
        >>> assert is_close(10001, 10000)
        >>> assert not is_close(10001, 10000, is_tolerance_ratio=False)
        >>> assert is_close(0.321457, 0.321466)
        >>> assert not is_close(0.321457, 0.321466, tolerance=0.000001)

    """
    if is_tolerance_ratio:
        return abs(x / y - 1) < tolerance
    else:
        return abs(x - y) < tolerance


def is_negligible(x, y, tolerance=0.001):
    return abs(x / y) < tolerance
