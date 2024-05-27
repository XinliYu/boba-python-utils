import os
import subprocess
from functools import partial
from os import path

from boba_python_utils.common_utils.function_helper import get_relevant_named_args
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.io_utils.pickle_io import pickle_load
from boba_python_utils.production_utils.s3 import mlps3_client, mlp_s3_load_success, URL_SCHEME_SEP, CMD_MLPS3, CMD_MLPS3_WITH_ENV_MOD
from boba_python_utils.production_utils.s3.common import get_s3_bucket_and_key, sanitize_s3url
from boba_python_utils.production_utils.s3.existence import exists_file, exists_path, exists_dir
from boba_python_utils.production_utils.s3.upload import put_file, put_dir


def get_file_cmd(src_path, dst_path, mlps3=None, cmd='get', overwrite: bool = True):
    if mlps3 is None:
        mlps3 = CMD_MLPS3_WITH_ENV_MOD

    if overwrite:
        return ' '.join((mlps3, cmd, '-f', src_path, dst_path))
    else:
        return ' '.join((mlps3, cmd, src_path, dst_path))


def get_file(src_path, dst_path, mlps3=None, cmd='get', verbose=True, **kwargs):
    src_path, dst_path = sanitize_s3url(src_path), sanitize_s3url(dst_path)
    use_mlp_s3 = (mlp_s3_load_success and cmd == 'get')
    if verbose:
        hprint_message('cmd', cmd)
        hprint_message('src_path', src_path)
        hprint_message('dst_path', dst_path)
        hprint_message('use_mlp_s3', use_mlp_s3)

    if use_mlp_s3:
        if mlps3 is None:
            mlps3 = mlps3_client
        bucket, key = get_s3_bucket_and_key(src_path)
        if exists_file(src_path, mlps3=mlps3, **kwargs):
            if path.isdir(dst_path):
                dst_path = path.join(dst_path, path.basename(key))
            mlps3.download_file(bucket, key, dst_path)
        else:
            if key[-1] != os.sep:
                key += os.sep
            if path.isfile(dst_path):
                dst_path += os.sep
            if not path.exists(dst_path):
                os.makedirs(dst_path, exist_ok=True)
            mlps3.download_files(
                bucket, key, dst_path,
                **get_relevant_named_args(mlps3.download_files, **kwargs)
            )
    else:
        if mlps3 is None:
            mlps3 = CMD_MLPS3
        mlps3_env = os.environ.copy()
        mlps3_env['PYTHONPATH'] = ''
        mlps3_env['LD_LIBRARY_PATH'] = ''
        if src_path[-1] != os.sep:
            src_path += os.sep
        if dst_path[-1] != os.sep:
            dst_path += os.sep
        run_command = partial(subprocess.run, env=mlps3_env)
        run_command([mlps3, cmd, '-f', src_path, dst_path])


def load_pickle_file(src_path, mlps3=None, cmd='get', verbose=True, **kwargs):
    if not exists_file(src_path):
        return None
    file_name = path.basename(src_path)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_file_path = path.join(tmpdirname, file_name)
        get_file(
            src_path=src_path,
            dst_path=tmp_file_path,
            mlps3=mlps3,
            cmd=cmd,
            verbose=verbose,
            **kwargs
        )
        return pickle_load(
            tmp_file_path,
            **get_relevant_named_args(
                func=pickle_load,
                **kwargs
            )
        )


def get_dir_cmd(src_path, dst_path, mlps3=None, cmd='get', overwrite: bool = True):
    if mlps3 is None:
        mlps3 = CMD_MLPS3_WITH_ENV_MOD

    if overwrite:
        return ' '.join((mlps3, cmd, '-r', '-f', src_path, dst_path))
    else:
        return ' '.join((mlps3, cmd, '-r', src_path, dst_path))


def get_dir(src_path, dst_path, mlps3=None, cmd='get', verbose=True, **kwargs):
    src_path, dst_path = sanitize_s3url(src_path), sanitize_s3url(dst_path)
    use_mlp_s3 = (mlp_s3_load_success and cmd == 'get')
    if verbose:
        hprint_message('cmd', cmd)
        hprint_message('src_path', src_path)
        hprint_message('dst_path', dst_path)
        hprint_message('use_mlp_s3', use_mlp_s3)

    if use_mlp_s3:
        if mlps3 is None:
            mlps3 = mlps3_client
        bucket, key = get_s3_bucket_and_key(src_path)

        if key[-1] != os.sep:
            key += os.sep
        if dst_path[-1] != os.sep:
            dst_path += os.sep
        if not path.exists(dst_path):
            os.makedirs(dst_path, exist_ok=True)
        mlps3.download_files(bucket, key, dst_path, **kwargs)
    else:
        if mlps3 is None:
            mlps3 = CMD_MLPS3
        mlps3_env = os.environ.copy()
        mlps3_env['PYTHONPATH'] = ''
        mlps3_env['LD_LIBRARY_PATH'] = ''
        if src_path[-1] != os.sep:
            src_path += os.sep
        if dst_path[-1] != os.sep:
            dst_path += os.sep
        run_command = partial(subprocess.run, env=mlps3_env)
        run_command([mlps3, cmd, '-rf', src_path, dst_path])


def sync_path(path1, path2, mlps3=None, verbose=True, **kwargs):
    """
    Checks existence of `path1` or `path2`,
    and copies one to the other if one exists and the other does not.
    At least one path must be a s3 path.

    NOTE this method only checks path existence but does not compare content.

    Args:
        path1: one path to sync.
        path2: the other path to sync.
        mlps3: provides the mlps3 method; specify None to use default.
        verbose: True to print out extra information when executing this method.
        **kwargs: extra arguments for s3 operation methods.

    """
    path1_is_s3 = (URL_SCHEME_SEP in path1)
    path2_is_s3 = (URL_SCHEME_SEP in path2)
    if path1_is_s3 and path2_is_s3:
        # the case when both paths are s3
        path1_exists, path2_exists = exists_path(path1), exists_path(path2)
        needs_sync = (path1_exists != path2_exists)
        if verbose:
            hprint_message('path1_exists', path1_exists)
            hprint_message('path2_exists', path2_exists)
            hprint_message('needs_sync', needs_sync)
        if needs_sync:
            if path2_exists:
                path1, path2 = path2, path1
            if exists_file(path1):
                get_file(path1, path2, mlps3=mlps3, cmd='cp', verbose=verbose)
            elif exists_dir(path1):
                get_dir(path1, path2, mlps3=mlps3, cmd='cp', verbose=verbose)

    elif path1_is_s3 != path2_is_s3:
        # the case when one path is s3
        if path2_is_s3:
            path1, path2 = path2, path1

        path1_exists, path2_exists = exists_path(path1), path.exists(path2)
        needs_sync = (path1_exists != path2_exists)
        if verbose:
            hprint_message('path1_exists', path1_exists)
            hprint_message('path2_exists', path2_exists)
            hprint_message('needs_sync', needs_sync)
        if needs_sync:
            if path2_exists:
                # path2 exists
                # path2 is local path, we upload the path from local to s3
                if path.isfile(path2):
                    put_file(path2, path1, mlps3=mlps3, **kwargs)
                elif path.isdir(path2):
                    put_dir(path2, path1, mlps3=mlps3, **kwargs)
            else:
                # path1 exists
                # path1 is s3 path, we download the path from s3 to local
                if exists_file(path1):
                    get_file(path1, path2, mlps3=mlps3, verbose=verbose)
                elif exists_dir(path1):
                    get_dir(path1, path2, mlps3=mlps3, verbose=verbose)
    else:
        raise ValueError(f"at least one path must be s3 path; got '{path1}' and '{path2}'")
