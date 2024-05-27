import os
import subprocess
import uuid
from os import path
from time import sleep
from typing import Callable, List, Mapping

from boba_python_utils.common_utils.iter_helper import tqdm_wrap
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.path_utility.path_string_operations import remove_ending_path_sep, solve_root_path, add_ending_path_sep
from boba_python_utils.general_utils.system_utility.command_lines import make_dir_cmd, rm_cmd, tar_decompress_cmd, tar_compress


def execute_ssh_command(
        command: str,
        cluster_ips: List[str],
        execution_method=os.system,
        no_strict_host_key_checking=True,
        non_blocking=False,
        non_blocking_completion_label_dir_path: str = None,
        non_blocking_wait_for_completion: bool = False,
        non_blocking_completion_check_interval: float = 1,
        show_prog=False,
        verbose: bool = True,
        **kwargs
) -> Mapping:
    """
    Executes commands on a cluster through SSH.
    Allows non-blocking parallel execution and command completion checking.

    Args:
        command: the command line to execute on the cluster.
        cluster_ips: provides a list of cluster IP addresses.
        execution_method: the method to execute the command line, e.g. `os.system`, `os.popen`, etc.
        no_strict_host_key_checking: disables strict host key checking; set this as True to avoid mantual interaction.
        non_blocking: enables non-blocking execution, so the command runs in parallel on all cluster nodes.
        non_blocking_completion_label_dir_path: specifies a path for a local path on each worker for command completion labels.
        non_blocking_wait_for_completion: True to wait for command completion on all cluster nodes before return.
        non_blocking_completion_check_interval: number of seconds to wait between two command completion checkings on cluster nodes.
        show_prog: True to show command execution progress.
        verbose: True to display rich command execution information.
        **kwargs: arguments for the `execution_method`

    Returns: a mapping of command execution results returned by the `execution_method`,
        with cluster IP addresses as the keys and the results as the values.

    """
    if non_blocking_wait_for_completion and (not non_blocking_completion_label_dir_path):
        non_blocking_completion_label_dir_path = path.join(path.expanduser('~'), '_completion_labels')

    if non_blocking_completion_label_dir_path:
        completion_label = str(uuid.uuid4())
        non_blocking_completion_label_path = path.join(non_blocking_completion_label_dir_path, completion_label)
        command += f' && mkdir -p {non_blocking_completion_label_dir_path} && touch {non_blocking_completion_label_path}'
    else:
        completion_label = None

    if verbose:
        hprint_message(
            'executing', command,
            'num_cluster_ips', len(cluster_ips),
            'completion_label', completion_label
        )

    results = {}
    for ip in tqdm_wrap(cluster_ips, use_tqdm=show_prog, tqdm_msg=f'executing command on cluster{" (non_blocking)" if non_blocking else ""}', verbose=verbose):
        cmd = f'ssh {ip} '
        if no_strict_host_key_checking:
            cmd += '-o StrictHostKeyChecking=no '
        cmd += f"\"{command}\""
        if non_blocking:
            cmd += ' &'
        results[ip] = execution_method(cmd, **kwargs)

    def _has_err(p):
        _err = p.stderr.read().decode('utf-8').lower()
        return _err and not ('contact your system administrator' in str(_err) and 'no such file or directory' not in _err)

    if non_blocking_wait_for_completion:
        unfinished_workers = set(cluster_ips)
        while len(unfinished_workers) == len(cluster_ips):
            for ip in unfinished_workers:
                p = subprocess.Popen(
                    ['ssh', ip, '-o StrictHostKeyChecking=no', f'ls {non_blocking_completion_label_path}'],
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                if not _has_err(p):
                    unfinished_workers.remove(ip)
                    break
                sleep(non_blocking_completion_check_interval * 3)
        while unfinished_workers:
            for ip in tqdm_wrap(list(unfinished_workers), use_tqdm=show_prog, tqdm_msg=f'waiting for command execution completion', verbose=verbose):
                p = subprocess.Popen(
                    ['ssh', ip, '-o StrictHostKeyChecking=no', f'ls {non_blocking_completion_label_path}'],
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                if not _has_err(p):
                    unfinished_workers.remove(ip)
                sleep(non_blocking_completion_check_interval)
            if len(unfinished_workers) == 1 or (len(unfinished_workers) / len(cluster_ips) <= 0.1):
                hprint_message('unfinished workers', unfinished_workers)
            sleep(non_blocking_completion_check_interval * 3)

    return results


def broadcast_files(
        input_path: str,
        cluster_ips,
        output_path: str,
        intermediate_path: str = None,
        put_file_method: Callable = None,
        put_dir_method: Callable = None,
        get_file_method: Callable = None,
        get_dir_method: Callable = None,
        overwrite: bool = False,
        compress: bool = True
):
    if intermediate_path is None:
        raise NotImplemented("We do not support direct SSH file broadcast yet.")

    input_path = remove_ending_path_sep(input_path)
    input_path_basename = path.basename(input_path)
    output_root_path, output_path = solve_root_path(output_path, default_basename=input_path_basename)
    intermediate_root_path, intermediate_path = solve_root_path(intermediate_path, default_basename=input_path_basename)

    hprint_message(
        'input_path', input_path,
        'input_path_basename', input_path_basename,
        'output_path', output_root_path,
        'intermediate_root_path', intermediate_root_path
    )

    if compress:
        _compressed_filepath = tar_compress(
            input_path=input_path,
            relative=True,
            overwrite=overwrite
        )

        put_file_method(
            src_path_local=_compressed_filepath,
            dst_path=add_ending_path_sep(intermediate_root_path),
            overwrite=overwrite
        )

        _compressed_filename = path.basename(_compressed_filepath)
        _intermediate_path_compressed = path.join(intermediate_root_path, _compressed_filename)
        _output_path_compressed = path.join(output_root_path, _compressed_filename)
        _compressed_filename = path.join('.', _compressed_filename)
        ssh_command = f"{make_dir_cmd(output_root_path, mode='777')} && " \
                      f"{rm_cmd(output_path, recursive=True, ignore_non_exist=True)} && " + \
                      f'{get_file_method(src_path=_intermediate_path_compressed, dst_path=_output_path_compressed)} && ' \
                      f'cd {output_root_path} && ' \
                      f"{tar_decompress_cmd(_compressed_filename)} && " \
                      f"chmod -R 777 {path.join('.', path.basename(output_path))} &&" \
                      f"{rm_cmd(_compressed_filename)}"

        execute_ssh_command(
            command=ssh_command,
            cluster_ips=cluster_ips,
            non_blocking=True,
            non_blocking_wait_for_completion=True,
            show_prog=True,
            verbose=True
        )

    elif path.isdir(input_path):
        put_dir_method(
            src_path_local=input_path,
            dst_path=intermediate_path,
            overwrite=overwrite
        )

        ssh_command = f"{make_dir_cmd(output_root_path)} && " \
                      f"{rm_cmd(output_path, recursive=True, ignore_non_exist=True)} &&" \
                      f"{get_dir_method(src_path=intermediate_path, dst_path=output_root_path)}"

        execute_ssh_command(
            command=ssh_command,
            cluster_ips=cluster_ips,
            non_blocking=True,
            non_blocking_wait_for_completion=True,
            show_prog=True,
            verbose=True
        )
    else:
        put_file_method(
            src_path_local=input_path,
            dst_path=intermediate_path,
            overwrite=overwrite
        )

        ssh_command = f"{make_dir_cmd(output_root_path)} && " \
                      f"{rm_cmd(output_path, recursive=False, ignore_non_exist=True)} &&" \
                      f"{get_file_method(src_path=intermediate_path, dst_path=output_path)}"

        execute_ssh_command(
            command=ssh_command,
            cluster_ips=cluster_ips,
            non_blocking=True,
            non_blocking_wait_for_completion=True,
            show_prog=True,
            verbose=True
        )
    return output_path
