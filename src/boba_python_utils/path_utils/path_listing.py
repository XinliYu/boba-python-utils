# mypy: ignore-errors
import os
import re
from enum import IntEnum
from os import path
from pathlib import Path
from typing import List, Union, Iterator


class FullPathMode(IntEnum):
    RelativePath = 0
    FullPath = 1
    BaseName = 2
    FullPathRelativePathTuple = 3


def sort_paths(
        paths: List[str],
        sort: Union[bool, str],
        sort_by_basename: bool = False
) -> List[str]:
    """
    Sorts a list of file paths according to the specified criteria: alphabetically, by base name,
    or by numerical indices extracted from the paths.

    Args:
        paths: A list of file paths to be sorted.
        sort: Determines the sorting criteria. If True or 'alphabetic', sorts paths alphabetically.
              If 'index', sorts paths based on numerical indices found in the paths or filenames.
        sort_by_basename: If True, sorts by the base name (filename) of each path.
                          If False, sorts by the entire path.

    Returns:
        The list of paths sorted according to the specified criteria.

    Examples:
        >>> paths = ["file3.txt", "file1.txt", "file2.txt"]
        >>> sort_paths(paths, sort=True)
        ['file1.txt', 'file2.txt', 'file3.txt']

        >>> paths = ["/path/to/file3.txt", "/path/to/file1.txt", "/path/to/file2.txt"]
        >>> sort_paths(paths, sort='alphabetic', sort_by_basename=True)
        ['/path/to/file1.txt', '/path/to/file2.txt', '/path/to/file3.txt']

        >>> paths = ["/path/to/file3_3.txt", "/path/to/file1_1.txt", "/path/to/file2_2.txt"]
        >>> sort_paths(paths, sort='index')
        ['/path/to/file1_1.txt', '/path/to/file2_2.txt', '/path/to/file3_3.txt']

        Note: For 'index' sorting, the function expects to find numerical indices in the paths or filenames.
              It sorts based on the first numerical index found in each path or filename.
    """
    if sort_by_basename:
        if sort is True or sort == 'alphabetic':
            return sorted(paths, key=lambda x: path.basename(x))
        elif sort == 'index':
            return sorted(paths, key=lambda x: int(re.search(r'[0-9]+', path.basename(x)).group()))
    else:
        if sort is True or sort == 'alphabetic':
            return sorted(paths)
        elif sort == 'index':
            return sorted(paths, key=lambda x: int(re.search(r'[0-9]+', x).group()))
    return paths


def iter_files_by_pattern(dir_or_dirs: str, pattern: str = '*', full_path: Union[FullPathMode, bool] = True, recursive=True):
    """
    Iterate through the paths or file names of all files in a folder at path `dir_path` of a specified `pattern`.
    :param dir_or_dirs: the path to the folder.
    :param pattern: only iterate through files of this pattern, e.g. '*.json'; a pattern starts with `**/` indicates to recursively search all sub folders.
    :param full_path: `True` if the full path to each file should be returned; otherwise, only the file name will be returned.
    :param recursive: `True` if recursively searching all sub folders, equivalent to adding prefix `**/` in front of `pattern`; `False` has no actual effect.
    :return: an iterator that returns each file of the specified pattern in a folder at path `dir_path`.
    """

    def _iter_files(dir_path):
        nonlocal pattern
        if path.isdir(dir_path):
            if recursive and not pattern.startswith('**/'):
                pattern = '**/' + pattern
            if full_path is True or full_path == FullPathMode.FullPath:
                p = Path(path.abspath(dir_path))
                for f in p.glob(pattern):
                    if path.isfile(f):
                        yield str(f)
            elif full_path is False or full_path == FullPathMode.BaseName:
                p = Path(path.abspath(dir_path))
                for f in p.glob(pattern):
                    if path.isfile(f):
                        yield path.basename(f)
            elif full_path == FullPathMode.RelativePath:
                dir_path = path.abspath(dir_path)
                len_dir_path = len(dir_path)
                p = Path(path.abspath(dir_path))
                for f in p.glob(pattern):
                    if path.isfile(f):
                        f = str(f)
                        yield f[len_dir_path + 1:] if f[len_dir_path] == os.sep else f[len_dir_path:]
            elif full_path == FullPathMode.FullPathRelativePathTuple:
                dir_path = path.abspath(dir_path)
                len_dir_path = len(dir_path)
                p = Path(path.abspath(dir_path))
                for f in p.glob(pattern):
                    if path.isfile(f):
                        f = str(f)
                        yield (f, f[len_dir_path + 1:]) if f[len_dir_path] == os.sep else (f, f[len_dir_path:])

    if isinstance(dir_or_dirs, str):
        yield from _iter_files(dir_or_dirs)
    else:
        for dir_path in dir_or_dirs:
            yield from _iter_files(dir_path)


def get_paths_by_pattern(
        dir_or_dirs: Union[str, Iterator],
        pattern: str = '*',
        full_path: Union[FullPathMode, bool] = True,
        recursive=True,
        sort=False,
        sort_use_basename=False,
        path_filter='file'
):
    """
    Retrieves paths matching a given pattern within specified directory or directories,
    with options for full or relative paths, recursion, sorting, and filtering by file or directory.

    Args:
        dir_or_dirs: A single directory path or an iterator of directory paths to search within.
        pattern: Pattern to match the files or directories against, using Unix shell-style wildcards.
                 Defaults to '*', matching everything. Prepending with '**/' searches recursively,
                 similar to setting `recursive=True`.
        full_path: Determines the format of the returned paths. When True (default), returns full paths.
                   If False, returns only base names. Can also be an instance of FullPathMode for more options.
        recursive: If True, searches directories recursively. Equivalent to patterns starting with '**/'. Defaults to True.
        sort: If True, returns sorted list of paths. Defaults to False.
        sort_use_basename: If True and `sort` is also True, sorts the paths by their base names. Defaults to False.
        path_filter: Specifies whether to filter for 'file' or 'directory', or a custom function for filtering.
                     Defaults to 'file', filtering for files only.

    Returns:
        A list of paths matching the specified pattern and filters, formatted according to `full_path`.

    Examples:
        Suppose we have a directory structure as follows, and `dir_or_dirs` is set to '/path/to/directory':
        /path/to/directory/
        ├── file1.txt
        ├── file2.json
        └── subdirectory/
            └── file3.txt

        get_paths_by_pattern('/path/to/directory', '*.txt')
        # Returns ['/path/to/directory/file1.txt', '/path/to/directory/subdirectory/file3.txt']
        # if `recursive=True` and `full_path=True`.

        get_paths_by_pattern('/path/to/directory', '*.txt', full_path=False)
        # Returns ['file1.txt'] if `recursive=False`.

        get_paths_by_pattern(['/path/to/directory1', '/path/to/directory2'], '*.json', sort=True)
        # Returns sorted list of .json files from both directories if they exist.
    """

    if isinstance(path_filter, str):
        path_filter = getattr(path, f'is{path_filter}', None)

    def _proc1(f, len_dir_path):
        f = str(f)
        return f[len_dir_path + 1:] if f[len_dir_path] == os.sep else f[len_dir_path:]

    def _proc2(f, len_dir_path):
        f = str(f)
        return (f, f[len_dir_path + 1:]) if f[len_dir_path] == os.sep else (f, f[len_dir_path:])

    def _get_files(dir_path):
        nonlocal pattern
        if path.isdir(dir_path):
            if recursive and not pattern.startswith('**/'):
                pattern = '**/' + pattern

            if full_path is True or full_path == FullPathMode.FullPath:
                p = Path(path.abspath(dir_path))
                results = [str(f) for f in p.glob(pattern) if ((not path_filter) or path_filter(f))]
            elif full_path is False or full_path == FullPathMode.BaseName:
                p = Path(dir_path)
                results = [path.basename(f) for f in p.glob(pattern) if ((not path_filter) or path_filter(f))]
            elif full_path == FullPathMode.RelativePath:
                dir_path = path.abspath(dir_path)
                p = Path(dir_path)
                len_dir_path = len(dir_path)
                results = [_proc1(f, len_dir_path) for f in p.glob(pattern) if ((not path_filter) or path_filter(f))]
            elif full_path == FullPathMode.FullPathRelativePathTuple:
                dir_path = path.abspath(dir_path)
                p = Path(dir_path)
                len_dir_path = len(dir_path)
                results = [_proc2(f, len_dir_path) for f in p.glob(pattern) if ((not path_filter) or path_filter(f))]
            return sort_paths(results, sort=sort, sort_by_basename=sort_use_basename)
        else:
            return []

    return (
        _get_files(dir_or_dirs)
        if isinstance(dir_or_dirs, str)
        else sum([_get_files(dir_path) for dir_path in dir_or_dirs], [])
    )


def iter_all_sub_dirs(dir_path: str):
    """
    Iterate through all subdirectories of a given directory.

    This will yield nested subdirectories. The `os.walk` is a built-in Python function
    that generates a file and directory tree by walking through a given directory and
    all of its subdirectories, including nested subdirectories.

    Args:
        dir_path: Path to the parent directory containing the subdirectories.

    Returns:
        An iterator yielding the subdirectories within the given parent directory.
    """
    return map(lambda x: x[0], os.walk(dir_path))


def get_all_sub_dirs(dir_path: str) -> List[str]:
    """
    Get all subdirectories of a given directory.

    This will return nested subdirectories. The `os.walk` is a built-in Python function
    that generates a file and directory tree by walking through a given directory and
    all of its subdirectories, including nested subdirectories.

    Args:
        dir_path: Path to the parent directory containing the subdirectories.

    Returns:
        A list of subdirectories within the given parent directory.
    """
    return [x[0] for x in os.walk(dir_path)]


def get_files_by_pattern(
        dir_or_dirs: Union[str, Iterator],
        pattern: str = '*',
        full_path: Union[FullPathMode, bool] = True,
        recursive=True,
        sort: bool = False,
        sort_use_basename: bool = False
):
    """
    Get the paths or file names of all files in a folder at path(s) specified by `dir_or_dirs` of
    a specified `pattern`.

    Args:
        dir_or_dirs: Path to one or more folders where this method is going to search for files of
                     the provided pattern.
        pattern: Search for files of this pattern, e.g. '*.json'; a pattern starts with
                            `**/` indicates to recursively search all subfolders, or equivalently
                            set the parameter `recursive` as `True`. Default is '*'.
        full_path: If True, the full path to each file should be returned; otherwise,
                              only the file name will be returned. Default is True.
        recursive: If True, recursively search all subfolders, equivalent to adding
                              prefix `**/` in front of `pattern`. Default is True.
        sort: If True, sort the returned file paths (by their strings). Default is False.
        sort_use_basename: If True, sort by basename. Default is False.

    Returns:
        A list of file paths of the specified file name pattern in the folder or folders specified
        in `dir_or_dirs`.
    """

    def _proc1(f, len_dir_path):
        # Process the file path to return the relative path
        f = str(f)
        return f[len_dir_path + 1:] if f[len_dir_path] == os.sep else f[len_dir_path:]

    def _proc2(f, len_dir_path):
        # Process the file path to return a tuple of full path and relative path
        f = str(f)
        return (f, f[len_dir_path + 1:]) if f[len_dir_path] == os.sep else (f, f[len_dir_path:])

    def _get_files(dir_path):
        nonlocal pattern
        if path.isdir(dir_path):
            if recursive and not pattern.startswith('**/'):
                pattern = '**/' + pattern

            if full_path is True or full_path == FullPathMode.FullPath:
                # Get files with full path
                p = Path(path.abspath(dir_path))
                results = [str(f) for f in p.glob(pattern) if path.isfile(f)]
            elif full_path is False or full_path == FullPathMode.BaseName:
                # Get files with basename only
                p = Path(dir_path)
                results = [path.basename(f) for f in p.glob(pattern) if path.isfile(f)]
            elif full_path == FullPathMode.RelativePath:
                # Get files with relative path
                dir_path = path.abspath(dir_path)
                p = Path(dir_path)
                len_dir_path = len(dir_path)
                results = [_proc1(f, len_dir_path) for f in p.glob(pattern) if path.isfile(f)]
            elif full_path == FullPathMode.FullPathRelativePathTuple:
                # Get files with a tuple of full path and relative path
                dir_path = path.abspath(dir_path)
                p = Path(dir_path)
                len_dir_path = len(dir_path)
                results = [_proc2(f, len_dir_path) for f in p.glob(pattern) if path.isfile(f)]
            return sort_paths(results, sort=sort, sort_by_basename=sort_use_basename)
        else:
            return []

    # Handle single directory and multiple directories cases
    return (
        _get_files(dir_or_dirs)
        if isinstance(dir_or_dirs, str)
        else sum([_get_files(dir_path) for dir_path in dir_or_dirs], [])
    )


def get_sorted_files_from_all_sub_dirs(dir_path: str, pattern: str, full_path: bool = True):
    """
    Get sorted files from all subdirectories (including nested subdirectories)
    of a given directory and matching a specified pattern.

    Args:
        dir_path: Path to the parent directory containing the subdirectories.
        pattern: Search for files of this pattern, e.g., '*.json'.
        full_path (optional): If True, the full path to each file should be returned; otherwise,
                              only the file name will be returned. Default is True.

    Returns:
        A sorted list of file paths of the specified file name pattern in all subdirectories of
        the given parent directory.

    """
    files = []
    sub_dirs = get_all_sub_dirs(dir_path)
    sub_dirs.sort()
    for sub_dir in sub_dirs:
        sub_dir_files = get_files_by_pattern(
            dir_or_dirs=sub_dir,
            pattern=pattern,
            full_path=full_path,
            recursive=False
        )
        sub_dir_files.sort()
        files.extend(sub_dir_files)
    return files
