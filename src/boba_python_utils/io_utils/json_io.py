import json
from itertools import chain, islice
from os import path
from typing import Union, Iterable, Iterator, Dict, Mapping, Type

from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.io_utils.common import open_, read_text_or_file
from boba_python_utils.io_utils.text_io import _get_input_file_stream, write_all_lines, read_all_text
from boba_python_utils.path_utils.path_string_operations import get_main_name, get_ext_name
from boba_python_utils.path_utils.path_listing import get_files_by_pattern, get_sorted_files_from_all_sub_dirs
from boba_python_utils.path_utils.common import ensure_dir_existence


def iter_all_json_strs(json_obj_iter, process_func=None, indent=None, ensure_ascii=False, **kwargs):
    if process_func:
        for json_obj in json_obj_iter:
            try:
                yield json.dumps(process_func(json_obj), indent=indent, ensure_ascii=ensure_ascii, **kwargs)
            except Exception as ex:
                print(json_obj)
                raise ex
    else:
        for json_obj in json_obj_iter:
            try:
                yield json.dumps(json_obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
            except Exception as ex:
                print(json_obj)
                raise ex


def _iter_json_objs(
        json_input: Union[str, Iterable, Iterator],
        use_tqdm: bool = True,
        disp_msg: str = None,
        verbose: bool = __debug__,
        encoding: str = None,
        ignore_error: bool = False,
        top: int = None,
        selection: Union[str, Iterable[str]] = None,
        result_type: Union[str, Type] = dict
) -> Iterator[Dict]:
    def _iter_single_input(json_input):

        if not (
                (result_type is dict)
                or (result_type is list)
                or (result_type is tuple)
                or result_type in ('dict', 'list', 'tuple')
        ):
            raise ValueError("'result_type' must be one of dict, list or tuple")

        # get an iterator for the json input
        line_iter, fin = _get_input_file_stream(
            file=json_input,
            encoding=encoding,
            top=top,
            use_tqdm=use_tqdm,
            display_msg=disp_msg or 'read json object from {}',
            verbose=verbose
        )
        # iterate through the json input
        for line in line_iter:
            if line:
                try:
                    json_obj = json.loads(line)
                    if selection:
                        if result_type is dict or result_type == 'dict':
                            json_obj = {
                                k: json_obj[k] for k in iter__(selection)
                            }
                        elif (
                                (result_type is list)
                                or (result_type is tuple)
                                or result_type == 'list'
                                or result_type == 'tuple'
                        ):
                            json_obj = result_type(json_obj[k] for k in iter__(selection))
                    elif (
                            (result_type is list)
                            or (result_type is tuple)
                            or result_type == 'list'
                            or result_type == 'tuple'
                    ):
                        json_obj = result_type(json_obj.values())
                    yield json_obj
                except Exception as ex:
                    if ignore_error is True:
                        print(line)
                        print(ex)
                    elif ignore_error == 'silent':
                        continue
                    else:
                        print(line)
                        raise ex
        fin.close()

    if isinstance(json_input, str):
        if path.isfile(json_input):
            # If input is a file, iterate over JSON objects in the file
            yield from _iter_single_input(json_input)
        else:
            # assuming otherwise the input is a directory,
            # then iterate over JSON objects in the json files under the directory
            for _json_input in get_files_by_pattern(
                    json_input,
                    pattern='*.json',
                    full_path=True,
                    recursive=False,
                    sort=True
            ):
                yield from _iter_single_input(_json_input)
    else:
        # otherwise, just try iterating the input,
        # assuming the input is itself an iterator or iterable
        yield from _iter_single_input(json_input)


def read_single_line_json_file(json_input: Union[str, Iterable, Iterator]):
    """
    Reads JSON objects from a file where each line contains a JSON object.

    Args:
        json_input: A path to a file, an iterable of file paths, or an iterator yielding lines of JSON.

    Returns:
        A list of JSON objects read from the input file(s).

    Example:
        >>> import tempfile
        >>> import os
        >>> tmpfile = tempfile.NamedTemporaryFile(delete=False)
        >>> tmpfile.write(b'[{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]\\n')
        >>> tmpfile.close()
        >>> read_single_line_json_file(tmpfile.name)
        >>> os.unlink(tmpfile.name)

    """
    if isinstance(json_input, str):
        return json.loads(read_all_text(json_input))
    else:
        out = []
        for _input_path in json_input:
            loaded_jobj_or_jobj_list = json.loads(read_all_text(_input_path))
            if isinstance(loaded_jobj_or_jobj_list, Mapping):
                out.append(loaded_jobj_or_jobj_list)
            else:
                out.extend(loaded_jobj_or_jobj_list)

        return out


def read_json(json_text_or_file: str):
    return read_text_or_file(
        text_or_file=json_text_or_file,
        read_text_func=json.loads,
        read_file_func=read_single_line_json_file
    )


def read_jsonl(jsonl_text_or_file: str):
    return read_text_or_file(
        text_or_file=jsonl_text_or_file,
        read_text_func=lambda _: [json.loads(json_line) for json_line in jsonl_text_or_file.split('\n') if json_line],
        read_file_func=lambda _: list(iter_json_objs(jsonl_text_or_file))
    )


def iter_json_objs(
        json_input: Union[str, Iterable, Iterator],
        use_tqdm: bool = True,
        disp_msg: str = None,
        verbose: bool = __debug__,
        encoding: str = None,
        ignore_error: bool = False,
        top: int = None,
        selection: Union[str, Iterable[str]] = None,
        result_type: Union[str, Type] = dict
) -> Iterator[Dict]:
    """
    Iterate through all JSON objects in a file, all JSON objects in all '.json' files in a directory,
    or a text line iterator.

    Args:
        json_input: Path to a JSON file, or a text line iterator.
        use_tqdm: If True, use tqdm to display reading progress. Default is True.
        disp_msg: Message to display for this reading. Default is None.
        verbose: If True, print out the `display_msg` regardless of `use_tqdm`. Default is __debug__.
        encoding: File encoding to use when reading from a file, such as 'utf-8'.
        ignore_error: If True, ignore errors when parsing JSON objects. Default is False.
        top: Maximum number of JSON objects to read. Default is None (read all).
        selection: one or more keys to select fields from the returned json objs.
        result_type: can be one of dict, list or tuple; if 'list' or 'tuple' is specified,
            then only values are returned as a list or a tuple.
    Returns: JSON object iterator.

    """

    if isinstance(json_input, (tuple, list, set)):
        for _json_input in json_input:
            yield from _iter_json_objs(
                json_input=_json_input,
                use_tqdm=use_tqdm,
                disp_msg=disp_msg,
                verbose=verbose,
                encoding=encoding,
                ignore_error=ignore_error,
                top=top,
                selection=selection,
                result_type=result_type
            )
    else:
        yield from _iter_json_objs(
            json_input=json_input,
            use_tqdm=use_tqdm,
            disp_msg=disp_msg,
            verbose=verbose,
            encoding=encoding,
            ignore_error=ignore_error,
            top=top,
            selection=selection,
            result_type=result_type
        )


def _iter_all_json_objs_from_all_sub_dirs(
        input_path: str,
        pattern: str = '*.json',
        use_tqdm: bool = False,
        display_msg: str = None,
        verbose: bool = __debug__,
        encoding: str = None,
        ignore_error: bool = False,
        top: int = None,
        selection: Union[str, Iterable[str]] = None,
        result_type: Union[str, Type] = dict
) -> Iterator[Dict]:
    """
    Iterate through all JSON objects from all subdirectories (including nested subdirectories)
    of a given directory, matching a specified pattern.

    Args:
        input_path: Path to the parent directory containing the subdirectories or path to the JSON file.
        pattern: Search for files of this pattern, e.g., '*.json'. Default is '*.json'.
        use_tqdm: If True, use tqdm to display reading progress. Default is False.
        display_msg: Message to display for this reading.
        verbose: If True, print out the display_msg regardless of use_tqdm. Default is __debug__.
        encoding: File encoding, such as 'utf-8'.
        ignore_error: If True, ignore JSON decoding errors and continue. Default is False.
        top: Number of lines to read from each file.
        selection: one or more keys to select fields from the returned json objs.
        result_type: can be one of dict, list or tuple; if 'list' or 'tuple' is specified,
            then only values are returned as a list or a tuple.
    Returns:
        An iterator yielding JSON objects found in all subdirectories of the given parent directory.

    """
    if path.isfile(input_path):
        all_files = [input_path]
    else:
        all_files = get_sorted_files_from_all_sub_dirs(dir_path=input_path, pattern=pattern)

    for json_file in all_files:
        yield from iter_json_objs(
            json_input=json_file,
            use_tqdm=use_tqdm,
            disp_msg=display_msg,
            verbose=verbose,
            encoding=encoding,
            ignore_error=ignore_error,
            top=top,
            selection=selection,
            result_type=result_type
        )


def iter_all_json_objs_from_all_sub_dirs(
        input_path_or_paths: Union[str, Iterable[str]],
        pattern: str = '*.json',
        use_tqdm: bool = False,
        display_msg: str = None,
        verbose: bool = __debug__,
        encoding: str = None,
        ignore_error: bool = False,
        top: int = None,
        top_per_input_path: int = None,
        selection: Union[str, Iterable[str]] = None,
        result_type: Union[str, Type] = dict
) -> Iterator[Dict]:
    """
    Iterate through all JSON objects from all subdirectories of a given directory or directories,
    matching a specified pattern.

    Args:
        input_path_or_paths: Path to the parent directory containing the subdirectories or path to the
                             JSON file(s), or a list of such paths.
        pattern: Search for files of this pattern, e.g., '*.json'. Default is '*.json'.
        use_tqdm: If True, use tqdm to display reading progress. Default is False.
        display_msg: Message to display for this reading.
        verbose: If True, print out the display_msg regardless of use_tqdm. Default is __debug__.
        encoding: File encoding. Default is None.
        ignore_error: If True, ignore JSON decoding errors and continue. Default is False.
        top: Total number of JSON objects to read from all input files.
        top_per_input_path: Number of JSON objects to read from each input path;
            not effective if there is only one input path.
        selection: one or more keys to select fields from the returned json objs.
        result_type: can be one of dict, list or tuple; if 'list' or 'tuple' is specified,
            then only values are returned as a list or a tuple.

    Returns:
        An iterator yielding JSON objects found in all subdirectories of the given parent directory
        or directories.

    """

    if isinstance(input_path_or_paths, str):
        return _iter_all_json_objs_from_all_sub_dirs(
            input_path=input_path_or_paths,
            pattern=pattern,
            use_tqdm=use_tqdm,
            display_msg=display_msg,
            verbose=verbose,
            encoding=encoding,
            ignore_error=ignore_error,
            top=top,
            selection=selection,
            result_type=result_type
        )
    else:
        _it = chain(
            *(
                _iter_all_json_objs_from_all_sub_dirs(
                    input_path=input_path,
                    pattern=pattern,
                    use_tqdm=use_tqdm,
                    display_msg=display_msg,
                    verbose=verbose,
                    encoding=encoding,
                    ignore_error=ignore_error,
                    top=top_per_input_path,
                    selection=selection,
                    result_type=result_type
                )
                for input_path in input_path_or_paths
            )
        )
        if top:
            _it = islice(_it, top)
        return _it


def write_json_objs(
        json_obj_iter,
        output_path,
        process_func=None,
        use_tqdm=False,
        disp_msg=None,
        append=False,
        encoding='utf-8',
        ensure_ascii=False,
        indent=None,
        chunk_size: int = None,
        chunk_name_format: str = 'part_{:05}',
        chunked_file_ext_name: str = '.json',
        verbose=__debug__,
        create_dir=True,
        pid=None,
        **kwargs
):
    if pid is not None:
        output_path = path.join(
            path.dirname(output_path),
            get_main_name(output_path),
            f'{pid}.{get_ext_name(output_path)}'
        )

    write_all_lines(
        iterable=iter_all_json_strs(json_obj_iter, process_func, indent=indent, ensure_ascii=ensure_ascii, **kwargs),
        output_path=output_path,
        use_tqdm=use_tqdm,
        display_msg=disp_msg,
        append=append,
        encoding=encoding,
        verbose=verbose,
        create_dir=create_dir,
        chunk_size=chunk_size,
        chunk_name_format=chunk_name_format,
        chunked_file_ext_name=chunked_file_ext_name
    )


def write_json(obj, file_path: str, append=False, indent=None, create_dir=True, encoding='utf-8', ensure_ascii=False, **kwargs):
    if obj is None:
        return
    if create_dir:
        ensure_dir_existence(path.dirname(file_path), verbose=False)
    if isinstance(obj, Mapping):
        with open_(file_path, 'a' if append else 'w', encoding=encoding) as fout:
            fout.write(
                json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
                if encoding
                else json.dumps(obj, indent=indent, **kwargs)
            )
            fout.flush()
    elif hasattr(obj, '__dict__'):
        with open_(file_path, 'a' if append else 'w', encoding=encoding) as fout:
            fout.write(
                json.dumps(vars(obj), indent=indent, ensure_ascii=ensure_ascii, **kwargs)
                if encoding
                else json.dumps(vars(obj), indent=indent, **kwargs)
            )
            fout.flush()
    else:
        write_json_objs(
            json_obj_iter=obj,
            output_path=file_path,
            append=append,
            indent=indent,
            encoding=encoding,
            ensure_ascii=ensure_ascii,
            **kwargs
        )
