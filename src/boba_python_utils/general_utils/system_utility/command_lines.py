import os
import uuid
from os import path
from time import sleep
from typing import Callable

from boba_python_utils.common_utils.iter_helper import tqdm_wrap
from boba_python_utils.general_utils.console_util import hprint_message


def execute_command(
        *commands,
        execution_method: Callable = os.system,
        non_blocking: bool = False,
        non_blocking_completion_label_dir: str = None,
        non_blocking_wait_for_completion: bool = False,
        non_blocking_completion_check_interval: float = 0.5,
        show_prog: bool = True,
        verbose: bool = True,
        **kwargs
):
    out = []
    if non_blocking:
        if non_blocking_wait_for_completion and (not non_blocking_completion_label_dir):
            non_blocking_completion_label_dir = path.join(path.expanduser('~'), '_completion_labels')

        completion_labels = []
        for command in tqdm_wrap(commands, use_tqdm=show_prog, tqdm_msg="executing commands (non-blocking)"):
            if non_blocking_completion_label_dir:
                completion_label = str(uuid.uuid4())
                command += f' && mkdir -p {non_blocking_completion_label_dir} && touch {path.join(non_blocking_completion_label_dir, completion_label)}'
            else:
                completion_label = None
            command += ' &'
            if verbose:
                hprint_message(
                    'executing', command,
                    'completion_label', completion_label
                )
            completion_labels.append(completion_label)
            out.append(execution_method(command, **kwargs))
        if non_blocking_wait_for_completion:
            for completion_label in tqdm_wrap(completion_labels, use_tqdm=show_prog, tqdm_msg="waiting for command execution completion"):
                while True:
                    if path.exists(path.join(non_blocking_completion_label_dir, completion_label)):
                        break
                    sleep(non_blocking_completion_check_interval)
    else:
        for command in tqdm_wrap(commands, use_tqdm=show_prog, tqdm_msg="executing commands"):
            if verbose:
                hprint_message('executing', command)
            out.append(execution_method(command))

    if len(out) == 1:
        out = out[0]
    return out


def make_dir_cmd(pathstr, mode=''):
    cmd = f'mkdir -p {pathstr}'
    if mode:
        cmd += f' && chmod -R 777 {pathstr}'
    return cmd


def rm_cmd(pathstr, recursive: bool = None, ignore_non_exist: bool = True, relative_to_current_path: bool = False):
    if relative_to_current_path:
        pathstr = path.join('.', pathstr)
    if recursive is None:
        recursive = path.isdir(pathstr)
    cmdlets = ['rm']
    if recursive:
        cmdlets.append('-r')
    if ignore_non_exist:
        cmdlets.append('-f')
    cmdlets.append(pathstr)
    return ' '.join(cmdlets)


def tar_compress_cmd(
        input_path: str,
        output_path: str = None,
        options: str = 'zvf',
        relative: bool = False
) -> str:
    if 'x' in options:
        raise ValueError("'x' cannot be in the tar option for compression")
    if 'c' not in options:
        options = 'c' + options

    if not output_path:
        output_path = input_path + '.tar.gz'

    if relative:
        cmd = f'cd {path.dirname(input_path)} && ' \
              f'tar -czvf {path.basename(output_path)} {path.join(".", path.basename(input_path))}'
    else:
        cmd = f'tar -{options} {output_path} {input_path}'
    return cmd


def tar_decompress_cmd(
        input_path: str,
        output_path: str = None,
        options: str = 'vf'
):
    if 'c' in options:
        raise ValueError("'c' cannot be in the tar option for decompression")
    if 'x' not in options:
        options = 'x' + options
    if output_path is None:
        cmd = f'tar -{options} {input_path}'
    else:
        cmd = f'tar -{options} {input_path} --directory {output_path}'
    return cmd


def tar_compress(
        input_path: str,
        output_path: str = None,
        options: str = 'czvf',
        relative: bool = False,
        overwrite: bool = False,
        non_blocking=False,
        non_blocking_completion_label_dir: str = None,
        verbose: bool = True,
        **kwargs
):
    if not output_path:
        output_path = input_path + '.tar.gz'
    _output_exists = path.exists(output_path)

    if overwrite or not _output_exists:
        if _output_exists:
            os.remove(output_path)

        cmd = tar_compress_cmd(
            input_path=input_path,
            output_path=output_path,
            options=options,
            relative=relative
        )

        execute_command(
            cmd,
            execution_method=os.system,
            non_blocking=non_blocking,
            non_blocking_completion_label_dir=non_blocking_completion_label_dir,
            show_prog=False,
            verbose=verbose,
            **kwargs
        )

    return output_path
