# region reading
import random
from itertools import islice
from typing import Callable, Union, Iterable, Optional, Iterator, Any

from boba_python_utils.common_utils.iter_helper import tqdm_wrap
from boba_python_utils.common_utils.typing_helper import str2val_
from boba_python_utils.io_utils.common import open_
from boba_python_utils.general_utils.iter_utils import with_names, chunk_iter
from boba_python_utils.path_utils.path_string_operations import make_ext_name
from boba_python_utils.path_utils.path_listing import sort_paths
from boba_python_utils.string_utils.common import strip_
from os import path


def read_all_text(file_path: str, encoding: Optional[str] = None) -> str:
    """
    Reads all text from a file.
    Args:
        file_path (str): the path to the file.
        encoding (Optional[str]): provides encoding for the file.

    Returns:
        str: the text read from the file.
    """
    with open(file_path, encoding=encoding) as f:
        return f.read()


def read_all_text_(file_path_or_content: str, encoding: Optional[str] = None) -> str:
    if path.isfile(file_path_or_content):
        if file_path_or_content:
            if path.isfile(file_path_or_content):
                return read_all_text(file_path_or_content, encoding)
            else:
                return file_path_or_content


def write_all_text(text: str, file_path: str, encoding: Optional[str] = None, create_dir: bool = True) -> None:
    """
    Writes the given text to a file, replacing any existing content.
    Args:
        text (str): the text to write to the file.
        file_path (str): the path to the file.
        encoding (Optional[str]): the encoding to use when writing to the file.
        create_dir (bool): If True, creates the parent directory if it does not exist.

    Returns:
        None
    """
    with open_(file_path, mode='w', encoding=encoding, create_dir=create_dir) as f:
        f.write(text)


def _iter_all_lines(
        file_path: str,
        use_tqdm: bool = False,
        description: str = None,
        lstrip: bool = False,
        rstrip: bool = True,
        encoding: str = None,
        parse: Union[str, Callable] = False,
        sample_rate: float = 1.0,
        verbose: bool = __debug__
):
    """
    Iterates through all lines of a file, applying optional transformations
    such as stripping leading/trailing characters, parsing, and sampling.

    Args:
        file_path: The file path to be read.
        use_tqdm: A flag indicating whether to use tqdm for progress display. Defaults to False.
        description: A description for the tqdm progress bar.
            If not provided, it will be generated.
        lstrip: A flag indicating whether to remove leading characters from the left side of each line.
        rstrip: A flag indicating whether to remove trailing characters from the right side of each line.
        encoding: The encoding to be used for opening the file.
        parse: A callable or string that specifies the parsing function to apply to each line.
            If set to True, the default `str2val_` function is used.
            Defaults to False (no parsing).
        sample_rate: A float in the range [0, 1] specifying the probability of including each line in the output.
            Defaults to 1.0 (all lines are included).
        verbose: A flag indicating whether to display tqdm progress bar or not. Defaults to __debug__.

    Yields:
        The modified lines after applying the specified transformations.

    Examples:
        >>> import tempfile
        >>> test_file = tempfile.NamedTemporaryFile(delete=False)
        >>> assert test_file.write(b"   Line 1   \\n   Line 2   \\n   Line 3   ")
        >>> test_file.close()

        >>> list(_iter_all_lines(test_file.name,
        ...    lstrip=True,
        ...    rstrip=True,
        ...    use_tqdm=False,
        ...    verbose=False
        ... ))
        ['Line 1', 'Line 2', 'Line 3']

        >>> x = list(_iter_all_lines(test_file.name,
        ...    lstrip=True,
        ...    rstrip=True,
        ...    sample_rate=0.5,
        ...    use_tqdm=False,
        ...    verbose=False
        ... ))
        >>> assert len(x) <= 3
    """
    assert isinstance(sample_rate, float)
    assert 0.0 <= sample_rate <= 1.0

    description = (description or 'read from file at {path}').format(path=file_path)

    if parse is False:
        with open_(
                file_path, 'r',
                encoding=encoding,
                use_tqdm=use_tqdm,
                description=description,
                verbose=verbose
        ) as fin:
            yield from (
                strip_(line, lstrip=lstrip, rstrip=rstrip)
                for line in fin
                if (sample_rate == 1.0 or random.uniform(0, 1) < sample_rate)
            )
    else:
        if parse is True:
            parse = str2val_
        with open_(
                file_path, 'r',
                encoding=encoding,
                use_tqdm=use_tqdm,
                description=description,
                verbose=verbose
        ) as fin:
            yield from (
                parse(strip_(line, lstrip=lstrip, rstrip=rstrip))
                for line in fin
                if (sample_rate == 1.0 or random.uniform(0, 1) < sample_rate)
            )


def iter_all_lines(
        file_path: Union[str, Iterable[str]],
        use_tqdm: bool = False,
        description: str = None,
        lstrip: bool = False,
        rstrip: bool = True,
        encoding: str = None,
        parse: Union[str, Callable] = False,
        sample_rate: float = 1.0,
        sort_path: Union[str, bool] = False,
        sort_by_basename: bool = False,
        verbose: bool = __debug__
):
    """
    Iterates through all lines of a file or multiple files, applying optional transformations such as
    stripping leading/trailing characters, parsing, and sampling.

    Args:
        file_path: One or more file paths to read.
        use_tqdm: True if using tqdm to display progress; otherwise False.
        description: The message to display with the tqdm; the message can be a format pattern of a single
            parameter 'path', e.g., 'read from file at {path}'.
        lstrip: True if the spaces at the beginning of each read line should be stripped; otherwise False.
        rstrip: True if the spaces at the end of each read line should be stripped; otherwise False.
        encoding: Specifies encoding for the file, e.g., 'utf-8'.
        parse: True to parse each line as its most likely value, e.g., '2.178' to float 2.178,
            '[1,2,3,4]' to list [1,2,3,4]; or specify a customized callable for the string parsing.
        sample_rate: Randomly samples a ratio of lines and skips the others.
        sort_path: True to sort `file_path` if `file_path` contains multiple paths.
            See `sort_paths` function.
        sort_by_basename: True to sort the `file_path` using path base name
            rather than the whole path string. See `sort_paths` function.
        verbose: True if details of the execution should be printed on the terminal.

    Yields:
        The modified lines after applying the specified transformations.

    Examples:
        >>> import tempfile
        >>> test_file = tempfile.NamedTemporaryFile(delete=False)
        >>> assert test_file.write(b"   Line 1   \\n   Line 2   \\n   Line 3   ")
        >>> test_file.close()

        >>> list(iter_all_lines(test_file.name,
        ...    lstrip=True,
        ...    rstrip=True,
        ...    use_tqdm=False,
        ...    verbose=False
        ... ))
        ['Line 1', 'Line 2', 'Line 3']

        >>> test_file2 = tempfile.NamedTemporaryFile(delete=False)
        >>> assert test_file2.write(b"   Line 4   \\n   Line 5   \\n   Line 6   ")
        >>> test_file2.close()

        >>> list(iter_all_lines([test_file.name, test_file2.name],
        ...    lstrip=True,
        ...    rstrip=True,
        ...    use_tqdm=False,
        ...    verbose=False
        ... ))
        ['Line 1', 'Line 2', 'Line 3', 'Line 4', 'Line 5', 'Line 6']
    """
    if isinstance(file_path, str):
        yield from _iter_all_lines(
            file_path=file_path,
            use_tqdm=use_tqdm,
            description=description,
            lstrip=lstrip,
            rstrip=rstrip,
            encoding=encoding,
            parse=parse,
            sample_rate=sample_rate,
            verbose=verbose
        )
    else:
        file_path = sort_paths(file_path, sort=sort_path, sort_by_basename=sort_by_basename)
        for _file_path in file_path:
            yield from _iter_all_lines(
                file_path=_file_path,
                use_tqdm=use_tqdm,
                description=description,
                lstrip=lstrip,
                rstrip=rstrip,
                encoding=encoding,
                parse=parse,
                sample_rate=sample_rate,
                verbose=verbose
            )


def read_all_lines(
        file_path: Union[str, Iterable[str]],
        use_tqdm: bool = False,
        description: str = None,
        lstrip: bool = False,
        rstrip: bool = True,
        encoding: str = None,
        parse: Union[str, Callable] = False,
        sample_rate: float = 1.0,
        sort_path: Union[str, bool] = False,
        sort_by_basename: bool = False,
        verbose: bool = __debug__
):
    """
    Works in the same way as :func:`iter_all_lines` but returns everything all at once.
    """

    return list(iter_all_lines(
        file_path=file_path,
        use_tqdm=use_tqdm,
        description=description,
        lstrip=lstrip,
        rstrip=rstrip,
        encoding=encoding,
        parse=parse,
        sample_rate=sample_rate,
        sort_path=sort_path,
        sort_by_basename=sort_by_basename,
        verbose=verbose
    ))


# endregion


def iter_all_lines_from_all_files(
        input_paths,
        sample_rate=1.0,
        lstrip=False,
        rstrip=True,
        use_tqdm=False,
        tqdm_msg=None,
        verbose=__debug__,
        sort=False,
        sort_by_basename=False
):
    """
    Iterates through all lines of a collection of input paths,
    with the options to sort input files, sub-sample lines,
    and strip the whitespaces at the start or the end of each line.
    """
    if isinstance(input_paths, str):
        input_paths = (input_paths,)
    else:
        input_paths = sort_paths(input_paths, sort=sort, sort_by_basename=sort_by_basename)
    if sample_rate >= 1.0:
        for file in input_paths:
            with open_(
                    file,
                    use_tqdm=use_tqdm,
                    description=tqdm_msg,
                    verbose=verbose
            ) as f:
                yield from (strip_(line, lstrip=lstrip, rstrip=rstrip) for line in f)
    else:
        for file in input_paths:
            with open_(
                    file,
                    use_tqdm=use_tqdm,
                    description=tqdm_msg,
                    verbose=verbose
            ) as f:
                for line in f:
                    if random.uniform(0, 1) < sample_rate:
                        yield strip_(line, lstrip=lstrip, rstrip=rstrip)


def read_all_lines_from_all_files(input_path, *args, **kwargs):
    return list(iter_all_lines_from_all_files(input_path, *args, **kwargs))


def _get_input_file_stream(
        file: Union[str, Iterable, Iterator],
        encoding: str,
        top: int,
        use_tqdm: bool,
        display_msg: str,
        verbose: bool
):
    # Check if 'file' is a file path (string) or a file-like object
    if isinstance(file, str):
        # If it's a file path, open the file with the given encoding
        fin = open(file, encoding=encoding)
    else:
        # If it's a file-like object, use it as is
        fin = file
        # Try to get the file name, if available, or set it ot 'an iterator' as a fallback
        if hasattr(file, 'name'):
            file = file.name
        else:
            file = 'an iterator'

    # Wrap the file stream with tqdm for progress display,
    # and use 'islice' to limit the number of lines read if 'top' is specified
    return tqdm_wrap(
        (islice(fin, top) if top else fin),
        use_tqdm=use_tqdm,
        tqdm_msg=display_msg.format(file) if display_msg else None,
        verbose=verbose
    ), fin


# region write lines
def write_all_lines_to_stream(
        fout,
        iterable: Iterator[str],
        to_str: Callable[[Any], str] = None,
        remove_blank_lines: bool = False,
        avoid_repeated_new_line: bool = True
):
    """
    Writes all lines from an iterable to a given output stream.

    Args:
        fout: The output stream.
        iterable: An iterable of strings.
        to_str: A function to convert items in iterable to string. If not provided, str() is used.
        remove_blank_lines: If True, blank lines will not be written.
        avoid_repeated_new_line: If True, avoids writing a new line if the text already ends with a new line.

    Example:
        >>> with open('output.txt', 'w') as f:
        ...     write_all_lines_to_stream(f, ['Line 1', 'Line 2', ''])
    """

    def _write_text(text):
        if len(text) == 0:
            if not remove_blank_lines:
                fout.write('\n')
        else:
            fout.write(text)
            if not avoid_repeated_new_line or text[-1] != '\n':
                fout.write('\n')

    if to_str is None:
        to_str = str

    for item in iterable:
        _write_text(to_str(item))

    fout.flush()


def write_all_lines(
        iterable: Iterator,
        output_path: str,
        to_str: Callable = None,
        use_tqdm: bool = False,
        display_msg: str = None,
        append: bool = False,
        encoding: str = None,
        verbose: bool = __debug__,
        create_dir: bool = True,
        remove_blank_lines: bool = False,
        avoid_repeated_new_line: bool = True,
        chunk_size: int = None,
        chunk_name_format: str = 'part_{:05}',
        chunked_file_ext_name: str = '.txt'
):
    """
    Writes all lines from an iterable to a file. If chunk size is provided, it splits the iterable into chunks
    and writes each chunk to a separate file.

    Args:
        iterable: An iterable of strings.
        output_path: The path where to write the output.
        to_str: A function to convert items in iterable to string. If not provided, str() is used.
        use_tqdm: If True, tqdm progress bar is displayed.
        display_msg: The message to display in the tqdm progress bar.
        append: If True, data is appended to existing files. Otherwise, it overwrites any existing files.
        encoding: The encoding of the output file(s).
        verbose: If True, additional progress details are displayed.
        create_dir: If True, creates the parent directory if it does not exist.
        remove_blank_lines: If True, blank lines will not be written.
        avoid_repeated_new_line: If True, avoids writing a new line if the text already ends with a new line.
        chunk_size: If provided, splits the iterable into chunks of this size and writes each chunk to a separate file.
        chunk_name_format: The format of the chunk file names.
        chunked_file_ext_name: The extension name for the chunked files.

    Example:
        >>> write_all_lines(['Line 1', 'Line 2', ''], 'output.txt')
    """
    iterable = tqdm_wrap(iterable, use_tqdm=use_tqdm, tqdm_msg=display_msg, verbose=verbose)
    if chunk_size is None:
        with open_(
                output_path,
                'a+' if append else 'w+',
                encoding=encoding,
                create_dir=create_dir
        ) as wf:
            write_all_lines_to_stream(
                fout=wf,
                iterable=iterable,
                to_str=to_str,
                remove_blank_lines=remove_blank_lines,
                avoid_repeated_new_line=avoid_repeated_new_line
            )
            wf.flush()
    else:
        chunked_file_ext_name = make_ext_name(chunked_file_ext_name)
        if path.isfile(output_path):
            output_path, _chunked_file_ext_name = path.splitext(output_path)
            if not chunked_file_ext_name:
                chunked_file_ext_name = _chunked_file_ext_name
            _chunked_file_main_name = path.basename(output_path)
            if _chunked_file_main_name:
                chunk_name_format = _chunked_file_main_name + '-' + chunk_name_format
            output_path = path.dirname(output_path)

        for chunk_name, chunk in with_names(
                chunk_iter(iterable, chunk_size=chunk_size),
                name_format=chunk_name_format,
                name_suffix=chunked_file_ext_name
        ):
            with open_(
                    path.join(output_path, chunk_name),
                    'a+' if append else 'w+',
                    encoding=encoding,
                    create_dir=create_dir
            ) as wf:
                write_all_lines_to_stream(
                    fout=wf,
                    iterable=chunk,
                    to_str=to_str,
                    remove_blank_lines=remove_blank_lines,
                    avoid_repeated_new_line=avoid_repeated_new_line
                )
                wf.flush()
# endregion
