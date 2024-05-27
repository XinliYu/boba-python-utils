import os
import shutil
from os import path
from typing import Union, List

from boba_python_utils.general_utils.console_util import hprint_pairs, hprint
from boba_python_utils.general_utils.messages import msg_skip_non_local_dir, msg_create_dir, msg_arg_not_a_dir, msg_clear_dir
from boba_python_utils.path_utils.path_listing import iter_files_by_pattern


def print_basic_path_info(*path_or_paths):
    for item in path_or_paths:
        if isinstance(item, str):
            hprint_pairs(("path", item), ("is file", path.isfile(item)), ("exists_path", path.exists(item)))
        else:
            hprint_pairs((item[0], item[1]), ("is file", path.isfile(item[1])), ("exists_path", path.exists(item[1])))


def ensure_parent_dir_existence(*dir_path_or_paths, clear_dir=False, verbose=__debug__):
    ensure_dir_existence(*(path.dirname(p) for p in dir_path_or_paths), clear_dir=clear_dir, verbose=verbose)
    return dir_path_or_paths[0] if len(dir_path_or_paths) == 1 else dir_path_or_paths


def ensure_dir_existence(
        *dir_path_or_paths,
        clear_dir=False,
        verbose=__debug__
) -> Union[str, List[str]]:
    """
    Creates a directory if the path does not exist. Optionally, set `clear_dir` to `True` to clear an existing directory.

import boba_python_utils.path_utils.common    >>> import utix.pathex as pathx
    >>> import os
    >>> path1, path2 = 'test/_dir1', 'test/_dir2'
    >>> boba_python_utils.path_utils.common.print_basic_path_info(path1)
    >>> boba_python_utils.path_utils.common.print_basic_path_info(path2)

    Pass in a single path.
    ----------------------
    >>> pathx.ensure_dir_existence(path1)
    >>> os.remove(path1)

    Pass in multiple paths.
    -----------------------
    >>> pathx.ensure_dir_existence(path1, path2)
    >>> os.remove(path1)
    >>> os.remove(path2)

    Pass in multiple paths as a tuple.
    ----------------------------------
    >>> # this is useful when this method is composed with another function that returns multiple paths.
    >>> def foo():
    >>>     return path1, path2
    >>> pathx.ensure_dir_existence(foo())

    :param dir_path_or_paths: one or more paths to check.
    :param clear_dir: clear the directory if they exist.
    :return: the input directory paths; this function has guaranteed their existence.
    """
    if len(dir_path_or_paths) == 1 and not isinstance(dir_path_or_paths[0], str):
        dir_path_or_paths = dir_path_or_paths[0]

    for dir_path in dir_path_or_paths:
        if '://' in dir_path:
            msg_skip_non_local_dir(dir_path)
            continue
        if not path.exists(dir_path):
            if verbose:
                hprint(msg_create_dir(dir_path))
            os.umask(0)
            os.makedirs(dir_path, mode=0o777, exist_ok=True)
        elif not path.isdir(dir_path):
            raise ValueError(msg_arg_not_a_dir(path_str=dir_path, arg_name='dir_path_or_paths'))
        elif clear_dir is True:
            if verbose:
                hprint(msg_clear_dir(dir_path))
            shutil.rmtree(dir_path)
            os.umask(0)
            os.makedirs(dir_path, mode=0o777, exist_ok=True)
        elif isinstance(clear_dir, str) and bool(clear_dir):
            for file in iter_files_by_pattern(dir_or_dirs=dir_path, pattern=clear_dir, recursive=False):
                os.remove(file)

        if verbose:
            print_basic_path_info(dir_path)

    return dir_path_or_paths[0] if len(dir_path_or_paths) == 1 else dir_path_or_paths
