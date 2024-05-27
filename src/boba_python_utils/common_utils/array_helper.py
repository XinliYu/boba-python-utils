from typing import Callable, List, Sequence, Any, Tuple, Union, Iterator


# region basic array utils
def is_homogeneous_sequence(items: Sequence[Any]) -> bool:
    """
    Checks if all elements in a sequence are of the same type.

    Args:
        items: The sequence of elements to check.

    Returns:
        True if all elements are of the same type or the sequence is empty; False otherwise.

    Examples:
        >>> is_homogeneous_sequence([1, 2, 3])
        True
        >>> is_homogeneous_sequence([1, '2', 3])
        False
        >>> is_homogeneous_sequence((1.0, 2.0, 3.0))
        True
        >>> is_homogeneous_sequence([])
        True
        >>> is_homogeneous_sequence(['hello', 'world', 'test'])
        True
        >>> is_homogeneous_sequence(['hello', 'world', 3])
        False
        >>> is_homogeneous_sequence([1, 2, 3, 4.5])
        False
        >>> is_homogeneous_sequence([{'a': 1}, {'b': 2}])
        True
        >>> is_homogeneous_sequence([1, 2, [3]])
        False
    """
    if not items:
        return True
    first_type = type(items[0])
    return all(isinstance(item, first_type) for item in items)


def split_half(arr: Union[List, Tuple]):
    """
    Splits a list or a tuple into two halves.

    If the input has an odd number of elements, the extra element is added to the second half.

    Args:
        arr: The list or tuple to be split.

    Returns:
        A tuple of two lists or tuples, each representing a half of the input.

    Example:
        >>> split_half([1, 2, 3, 4, 5])
        ([1, 2], [3, 4, 5])

        >>> split_half((1, 2, 3, 4, 5, 6))
        ((1, 2, 3), (4, 5, 6))
    """
    split_pos = len(arr) // 2
    return arr[:split_pos], arr[split_pos:]


def first_half(arr: Union[List, Tuple]):
    """
    Returns the first half of a list or a tuple.

    If the input has an odd number of elements, the extra element is not included in the returned half.

    Args:
        arr: The list or tuple to be halved.

    Returns:
        A list or tuple representing the first half of the input.

    Example:
        >>> first_half([1, 2, 3, 4, 5])
        [1, 2]

        >>> first_half((1, 2, 3, 4, 5, 6))
        (1, 2, 3)
    """
    return arr[:len(arr) // 2]


def second_half(arr: Union[List, Tuple]):
    """
    Returns the second half of a list or a tuple.

    If the input has an odd number of elements, the extra element is included in the returned half.

    Args:
        arr: The list or tuple to be halved.

    Returns:
        A list or tuple representing the second half of the input.

    Example:
        >>> second_half([1, 2, 3, 4, 5])
        [3, 4, 5]

        >>> second_half((1, 2, 3, 4, 5, 6))
        (4, 5, 6)
    """
    return arr[(len(arr) // 2):]


def all_equal(arr: Union[List, Tuple], value=None):
    """
    Checks if all elements of a list or a tuple are equal, or equal to a provided value.

    If the input is empty, the function returns True.

    Args:
        arr: The list or tuple to be checked.
        value: An optional value to which all elements of the list or tuple should be compared.

    Returns:
        True if all elements in the list or tuple are equal, or equal to the provided value,
        False otherwise.

    Example:
        >>> all_equal([1, 1, 1, 1])
        True

        >>> all_equal((1, 2, 3, 4))
        False

        >>> all_equal([])
        True

        >>> all_equal([2, 2, 2], 2)
        True

        >>> all_equal((1, 2, 3, 4), 2)
        False
    """
    if not arr:
        return True
    if value is None:
        if len(arr) == 1:
            return True
        else:
            return all(arr[0] == arr[i] for i in range(1, len(arr)))
    else:
        if len(arr) == 1:
            return arr[0] == value
        else:
            return all(arr[i] == value for i in range(len(arr)))


# endregion

# region sub-array iteration utils

def iter_split_list(list_to_split: List, num_splits: int) -> Iterator[List]:
    """
    Returns an iterator that iterates through even splits of the provided `list_to_split`.

    If the size of `list_to_split` is not dividable ty `num_splits`,
    then the last split will be larger or smaller than others in size.

    If the size of the `list_to_split` is smaller than or equal to `num_splits`,
        then singleton lists will be yielded and the total number of yielded splits
        is the same as the length of the `list_to_split`.

    Args:
        list_to_split: the list to split.
        num_splits: the number of splits to yield;

    Returns: an iterator that iterates through splits of the provided list; all splits are of
        the same size, except for the last split might be larger or smaller than others in size
        if the size of `list_to_split` is not dividable ty `num_splits`.

    Examples:
        >>> list(iter_split_list([1, 2, 3, 4, 5], 1))
        [[1, 2, 3, 4, 5]]
        >>> list(iter_split_list([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4, 5]]
        >>> list(iter_split_list([1, 2, 3, 4, 5], 3))
        [[1, 2], [3, 4], [5]]
        >>> list(iter_split_list([1, 2, 3, 4, 5], 6))
        [[1], [2], [3], [4], [5]]
    """
    list_len = len(list_to_split)
    if list_len <= num_splits:
        for item in list_to_split:
            yield [item]
    else:
        list_len = len(list_to_split)
        chunk_size = int(list_len / num_splits)
        remainder = int(list_len - chunk_size * num_splits)

        if remainder > 1:
            begin, end = 0, chunk_size + 1
            for i in range(0, remainder - 1):
                yield list_to_split[begin:end]
                begin, end = end, end + chunk_size + 1
        else:
            begin, end = 0, chunk_size

        for i in range(remainder - 1, num_splits - 1):
            yield list_to_split[begin:end]
            begin, end = end, end + chunk_size
        if begin < list_len:
            yield list_to_split[begin:]


def split_list(list_to_split: List, num_splits: int) -> List[List]:
    """
    See :func:`iter_split_list`.
    """
    return list(iter_split_list(list_to_split, num_splits))


def moving_window_convert(
        arr: Sequence,
        converter: Callable[[Sequence, Sequence], Any] = None,
        hist_window_size: int = 20,
        future_window_size: int = 10,
        pre_hist_window_size: int = None,
        step_size: int = 1,
        allows_partial_future_window: bool = False
) -> List:
    """
    Convert an input list into a list of tuples, where each tuple contains the historical window,
    future window, and optionally, the pre-historical window.

    A converter function ban be applied to operate on the windows.

    Args:
        arr: Input list of numeric values.
        converter: Optional function to apply on the historical, future, and pre-historical windows.
        hist_window_size: Size of the historical window.
        future_window_size: Size of the future window.
        pre_hist_window_size: Optional size of the pre-historical window.
        step_size: Step size for moving the window.
        allows_partial_future_window: If True, allows the last historical window to have a
            partial future window.

    Returns:
        A list of tuples containing historical and future windows, and optionally,
        pre-historical windows.

    Examples:
        >>> arr = list(range(1, 101))
        >>> result = moving_window_convert(arr, hist_window_size=10, future_window_size=5)
        >>> print(result[0])  # First historical-future window pair
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15])
        >>> print(result[-1])  # Last historical-future window pair
        ([86, 87, 88, 89, 90, 91, 92, 93, 94, 95], [96, 97, 98, 99, 100])
        >>> result_with_pre_hist = moving_window_convert(
        ...    arr,
        ...    hist_window_size=10,
        ...    future_window_size=5,
        ...    pre_hist_window_size=5
        ... )
        >>> print(result_with_pre_hist[0])  # First historical-future-pre-historical window tuple
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [])
        >>> print(result_with_pre_hist[-1])  # Last historical-future-pre-historical window tuple
        ([86, 87, 88, 89, 90, 91, 92, 93, 94, 95], [96, 97, 98, 99, 100], [81, 82, 83, 84, 85])
        >>> result_with_pre_hist = moving_window_convert(
        ...    arr,
        ...    hist_window_size=10,
        ...    future_window_size=5,
        ...    pre_hist_window_size=5,
        ...    allows_partial_future_window=True
        ... )
        >>> print(result_with_pre_hist[-1])  # Last historical-future-pre-historical window tuple
        ([90, 91, 92, 93, 94, 95, 96, 97, 98, 99], [100], [85, 86, 87, 88, 89])
        >>> arr = list(range(1, 22))
        >>> result_with_pre_hist = moving_window_convert(
        ...    arr,
        ...    hist_window_size=20,
        ...    future_window_size=10,
        ...    pre_hist_window_size=10,
        ...    allows_partial_future_window=True
        ... )
        >>> print(result_with_pre_hist)
        [([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [21], [])]
        >>> result_with_pre_hist = moving_window_convert(
        ...    arr, hist_window_size=20,
        ...    future_window_size=10,
        ...    pre_hist_window_size=10,
        ...    allows_partial_future_window=False
        ... )
        >>> print(result_with_pre_hist)
        []
    """
    out = []
    arr_len = len(arr)
    total_window_size = hist_window_size + future_window_size

    end_i = (
        (arr_len - hist_window_size)
        if allows_partial_future_window else
        (arr_len - total_window_size + 1)
    )

    if pre_hist_window_size:
        for i in range(0, end_i, step_size):
            i0 = max(0, i - pre_hist_window_size)
            i2 = i + hist_window_size
            i3 = i + total_window_size
            window = (arr[i:i2], arr[i2:i3], arr[i0:i])
            if converter is None:
                out.append(window)
            else:
                out.append(converter(*window))
    else:
        for i in range(0, end_i, step_size):
            i2 = i + hist_window_size
            i3 = i + total_window_size
            window = (arr[i:i2], arr[i2:i3])
            if converter is None:
                out.append(window)
            else:
                out.append(converter(*window))
    return out


# endregion

# region array cartesian product

def iter_cartesian_product(
        arr: Sequence[Any],
        arr_sort_func: Callable = None,
        bidirection_product: bool = False,
        include_self_product: bool = False
) -> Iterator[Tuple[Any, Any]]:
    """
    Computes the Cartesian product of elements in a sequence and returns an iterator.

    Args:
        arr: The input sequence.
        arr_sort_func: A function to sort the
            input sequence before computing the Cartesian product. Defaults to None.
        bidirection_product: If True, include both (a, b) and (b, a) in the result.
            Defaults to False.
        include_self_product: If True, include pairs with identical elements (a, a)
            in the result. Defaults to False.

    Returns: An iterator for the Cartesian product of the input sequence's elements.

    Examples:
        >>> input_arr = ["a", "b", "c"]
        >>> cart_product = iter_cartesian_product(input_arr, bidirection_product=False,
        ...                                      include_self_product=False)
        >>> list(cart_product)
        [('a', 'b'), ('a', 'c'), ('b', 'c')]

        >>> input_arr = ["a", "a", "b", "c"]
        >>> cart_product = iter_cartesian_product(input_arr, bidirection_product=False,
        ...                                      include_self_product=False)
        >>> list(cart_product)
        [('a', 'a'), ('a', 'b'), ('a', 'c'), ('a', 'b'), ('a', 'c'), ('b', 'c')]

        >>> input_arr = [1, 2, 3]
        >>> cart_product = iter_cartesian_product(input_arr, bidirection_product=True,
        ...                                      include_self_product=True)
        >>> list(cart_product)
        [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    """
    if arr_sort_func is None:
        arr_sort_func = sorted
    arr = arr_sort_func(arr)

    if include_self_product:
        if bidirection_product:
            return ((arr[i], arr[j]) for i in range(len(arr)) for j in range(len(arr)))
        else:
            return ((arr[i], arr[j]) for i in range(len(arr)) for j in range(i, len(arr)))
    else:
        if bidirection_product:
            return ((arr[i], arr[j]) for i in range(len(arr)) for j in range(len(arr)) if i != j)
        else:
            return ((arr[i], arr[j]) for i in range(len(arr)) for j in range(i + 1, len(arr)))


def get_cartesian_product(
        arr: Sequence[Any],
        arr_sort_func: Callable = None,
        bidirection_product: bool = False,
        include_self_product: bool = False
) -> List[Tuple[Any, Any]]:
    """
    See `iter_cartesian_product`.
    """
    return list(iter_cartesian_product(
        arr,
        arr_sort_func=arr_sort_func,
        bidirection_product=bidirection_product,
        include_self_product=include_self_product
    ))


# endregion

# region quick save list of lists to csv

def save_to_csv(data: Sequence[Sequence], filename: str, delimiter: str = ',', **kwargs) -> None:
    """Save data to a CSV file.

    Args:
        data (Sequence[Sequence]): The data to be written to the CSV file.
        filename (str): The filename of the CSV file.
        delimiter (str, optional): The character used to separate fields in the CSV file.
            Defaults to ','.
        **kwargs: Additional keyword arguments to pass to the csv.writer.

    Returns:
        None

    Example:
        >>> import os
        >>> data = [['Name', 'Age'], ['John', 30], ['Alice', 25]]
        >>> filename = 'test.csv'
        >>> save_to_csv(data, filename)
        >>> # Verify file content with default delimiter
        >>> with open(filename, 'r') as file:
        ...     content = file.read()
        >>> expected_content = "Name,Age\\nJohn,30\\nAlice,25\\n"
        >>> assert content == expected_content
        >>> os.remove(filename)
        >>>
        >>> # Test with a different delimiter
        >>> data = [['Name', 'Age'], ['John', 30], ['Alice', 25]]
        >>> filename = 'test2.csv'
        >>> save_to_csv(data, filename, delimiter=';')
        >>> # Verify file content with ';' delimiter
        >>> with open(filename, 'r') as file:
        ...     content = file.read()
        >>> expected_content = "Name;Age\\nJohn;30\\nAlice;25\\n"
        >>> assert content == expected_content
        >>> os.remove(filename)
    """
    import csv
    from boba_python_utils.path_utils.common import ensure_dir_existence
    from os import path
    ensure_dir_existence(path.dirname(filename))
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter, **kwargs)
        writer.writerows(data)

# endregion