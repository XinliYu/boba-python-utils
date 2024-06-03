from os import path
import os
from typing import Optional, Tuple, Callable, Any
import shutil
from boba_python_utils.common_utils.iter_helper import tqdm_wrap
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.path_utils.common import ensure_parent_dir_existence
from boba_python_utils.path_utils.path_with_date_time_info import add_date_time_to_path
from boba_python_utils.time_utils.common import solve_date_time_format_by_granularity


def _solve_paras_from_io_mode(
        file: str,
        mode: str,
        use_tqdm: bool,
        description: str,
        verbose: bool
) -> Tuple[bool, str, bool]:
    """
     Solves arguments for `open_`. This function disables the tqdm wrap for writing
     (because tqdm does not support that), generates the `description` for the tqdm progress bar
     if it is not specified, and determines if the IO operation requires the existence
     of the parent path.

     Args:
         file: The file path to be opened.
         mode: The file access mode, such as 'r', 'w', 'a', 'x', etc.
         use_tqdm: A flag indicating whether to use tqdm for progress display.
             Disabled for writing modes.
         description: A description for the tqdm progress bar.
             If not provided, it will be generated.
         verbose: A flag indicating whether to display tqdm progress bar or not.

     Returns:
         tuple: A tuple containing the modified values of `use_tqdm`, `description`,
         and a boolean indicating whether the
         parent path needs to exist for the specified IO operation.

     Examples:
         >>> import tempfile
         >>> test_file = tempfile.NamedTemporaryFile(delete=False)
         >>> assert test_file.write(b"Test content")
         >>> test_file.close()

         >>> mode = "r"
         >>> use_tqdm = True
         >>> description = None
         >>> verbose = True
         >>> x = _solve_paras_from_io_mode(test_file.name, mode, use_tqdm, description, verbose)
         >>> assert x[0] and not x[2]
         >>> assert x[1].startswith('read from file ')

         >>> mode = "w"
         >>> x = _solve_paras_from_io_mode(test_file.name, mode, use_tqdm, description, verbose)
         >>> assert not x[0] and x[2]
         >>> assert x[1].startswith('overwrite file ')

     """
    need_dir_exist = False
    if description is None and (use_tqdm or verbose):
        binary = 'binary ' if 'b' in mode else ''
        if 'r' in mode:
            description = f'read from {binary}file {file}'
        elif 'w' in mode:
            if path.exists(file):
                description = f'overwrite {binary}file {file}'
            else:
                description = f'write to {binary}file {file}'
            need_dir_exist = True
            use_tqdm = False
        elif 'a' in mode:
            description = f'append to {binary}file {file}'
            need_dir_exist = True
            use_tqdm = False
        elif 'x' in mode:
            description = f'write to {binary}file {file}'
            need_dir_exist = True
            use_tqdm = False
    else:
        need_dir_exist = 'w' in mode or 'a' in mode or 'x' in mode
    return use_tqdm, description, need_dir_exist


class open_:
    """
    Provides more options for opening a file, including creating the parent directory, tqdm wrap,
        and enforced flusing upon exit.

    Args:
        file: The file path to be opened.
        mode: The file access mode, such as 'r', 'w', 'a', 'x', etc. Defaults to None.
        append: A flag indicating whether to open the file in append mode. If True, the file is
            opened in 'a+' mode. If False, the file is opened in 'w+' mode. Cannot be used with `mode`. Defaults to None.
        encoding: The encoding to be used for opening the file. Defaults to None.
        use_tqdm: A flag indicating whether to use tqdm for progress display. Disabled for writing modes.
            Defaults to False.
        description: A description for the tqdm progress bar. If not provided, it will be generated.
            Defaults to None.
        verbose: A flag indicating whether to display tqdm progress bar or not. Defaults to __debug__.
        create_dir: A flag indicating whether to create the parent directory if it does not exist.
            Defaults to True.
        *args: Additional positional arguments to be passed to the built-in `open` function.
        **kwargs: Additional keyword arguments to be passed to the built-in `open` function.

    Examples:
        >>> import tempfile
        >>> test_file = tempfile.NamedTemporaryFile(delete=False)
        >>> assert test_file.write(b"Test content")
        >>> test_file.close()

        >>> with open_(test_file.name, mode="r", use_tqdm=False, verbose=False) as f:
        ...     print(f.read())
        Test content

        >>> with open_(test_file.name, mode="r", use_tqdm=True, verbose=False) as f:
        ...     list(f)
        ['Test content']

        >>> with open_(test_file.name, append=True, verbose=False) as f:
        ...     assert f.write("Appended text")

    """

    def __init__(
            self,
            file: str,
            mode: str = None,
            append: Optional[bool] = None,
            encoding: str = None,
            use_tqdm: bool = False,
            description: str = None,
            verbose: bool = __debug__,
            create_dir: bool = True,
            *args, **kwargs
    ):
        self._file = file

        if append is True:
            if mode is not None:
                raise ValueError('cannot specify `mode` and `append` at the same time')
            mode = 'a+'
        elif append is False:
            if mode is not None:
                raise ValueError('cannot specify `mode` and `append` at the same time')
            mode = 'w+'
        elif mode is None:
            mode = 'r'

        use_tqdm, description, need_dir_exist = _solve_paras_from_io_mode(
            file=file,
            mode=mode,
            use_tqdm=use_tqdm,
            description=description,
            verbose=verbose
        )

        if create_dir and need_dir_exist:
            os.makedirs(path.dirname(self._file), exist_ok=True)

        self._f = open(self._file, mode=mode, encoding=encoding, *args, **kwargs)
        self._f_with_tqdm_wrap = tqdm_wrap(
            self._f,
            use_tqdm=use_tqdm,
            tqdm_msg=description,
            verbose=verbose
        )

    def flush(self):
        self._f.flush()

    def __enter__(self):
        return self._f_with_tqdm_wrap

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._f.flush()
        self._f.close()


def read_text_or_file(text_or_file: str, read_text_func: Callable[[str], Any], read_file_func: Callable[[str], Any]):
    if not isinstance(text_or_file, str):
        return text_or_file
    if path.exists(text_or_file):
        return read_file_func(text_or_file)
    else:
        return read_text_func(text_or_file)


def create_empty_file(file_path: str) -> None:
    """
    Create an empty file at the specified path.

    Args:
        file_path: The path where the empty file should be created.
    """
    with open(file_path, "w") as file:
        pass


def backup(
        input_path: str,
        backup_path: str,
        datetime_granularity: str = "day",
        date_format: str = '%Y%m%d',
        time_format: str = '',
        unix_timestamp: bool = False
) -> None:
    """
    Create a backup copy of the input file or directory with a date/time string appended to the backup path.

    Args:
        input_path: Path to the file or directory to backup.
        backup_path: Path where the backup will be created.
        date_format: Format for date string. Defaults to 'YYYYMMDD'.
        time_format: Format for time string. If empty, no time is appended.
        datetime_granularity: Granularity of the date/time string, e.g. "year", "month", "date", "hour". Default is "second".
        unix_timestamp: If True, append Unix timestamp instead of date and time. Default is False.

    Raises:
        FileNotFoundError: If the input file or directory does not exist.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The input file or directory {input_path} does not exist.")

    # Get date and time format based on the granularity
    solved_date_format, solved_time_format = solve_date_time_format_by_granularity(
        datetime_granularity,
        date_format,
        time_format
    )

    # Get the backup path with date/time string appended
    backup_path_with_datetime = add_date_time_to_path(
        backup_path,
        date_format=solved_date_format,
        time_format=solved_time_format,
        unix_timestamp=unix_timestamp
    )

    ensure_parent_dir_existence(backup_path_with_datetime)

    if os.path.isdir(input_path):
        # Copy directory
        shutil.copytree(input_path, backup_path_with_datetime)
    else:
        # Copy file
        shutil.copy2(input_path, backup_path_with_datetime)

    hprint_message('backup created at', backup_path_with_datetime)
