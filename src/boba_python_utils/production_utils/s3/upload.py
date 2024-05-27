import subprocess
import sys
from os import path, listdir, linesep

from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.io_utils.pickle_io import pickle_save
from boba_python_utils.time_utils.tictoc import tic, toc
from boba_python_utils.production_utils.s3 import CMD_MLPS3_WITH_ENV_MOD
from boba_python_utils.production_utils.s3.common import sanitize_s3url
from boba_python_utils.production_utils.s3.existence import exists_path
from boba_python_utils.production_utils.s3.list import ls_dir


def put_file_cmd(src_path_local, dst_path, mlps3=None, overwrite=False):
    dst_path = sanitize_s3url(dst_path)
    mlps3 = mlps3 or CMD_MLPS3_WITH_ENV_MOD
    dst_path = sanitize_s3url(dst_path)
    if path.isfile(src_path_local):
        if dst_path.endswith('/'):
            dst_path = path.join(dst_path, path.basename(src_path_local))
        print('sending file {} to s3 path {}'.format(src_path_local, dst_path))
        cmd = "{} put -f {} {}" if overwrite else "{} put {} {}"
    else:
        raise ValueError(f"source path {src_path_local} is a directory.")
    return cmd.format(mlps3, src_path_local, dst_path)


def put_file(src_path_local, dst_path, mlps3=None, overwrite=False, retry=5):
    if (not overwrite) and exists_path(dst_path, mlps3=mlps3):
        return
    cmd = put_file_cmd(
        src_path_local=src_path_local,
        dst_path=dst_path,
        mlps3=mlps3,
        overwrite=overwrite
    )
    results = None
    while retry > 0:
        try:
            results = subprocess.check_output(
                cmd, shell=True
            ).decode(
                'UTF-8' if sys.stdout.encoding is None else sys.stdout.encoding
            ).strip().split(linesep)
            retry = 0
        except:
            retry -= 1
            continue
    return results


def put_pickle_file(obj, dst_path, mlps3=None, overwrite=False, retry=5):
    file_name = path.basename(dst_path)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_file_path = path.join(tmpdirname, file_name)
        pickle_save(obj, tmp_file_path)
        return put_file(
            src_path_local=tmp_file_path,
            dst_path=dst_path,
            mlps3=mlps3,
            overwrite=overwrite,
            retry=retry
        )


def put_dir(
        src_path_local,
        dst_path,
        mlps3=None,
        verbose=True,
        overwrite=True,
        skip_private_files=False
):
    dst_path = sanitize_s3url(dst_path)
    mlps3 = mlps3 or CMD_MLPS3_WITH_ENV_MOD
    if not dst_path[-1] == path.sep:
        dst_path += path.sep
    if not src_path_local[-1] == path.sep:
        src_path_local += path.sep
    uploaded_paths = ls_dir(dst_path, print_out=verbose, skip_private_files=skip_private_files)

    if not uploaded_paths:
        if path.isfile(src_path_local):
            raise ValueError(f"source path {src_path_local} is a file.")

        cmd_get = f'{mlps3} put -r {src_path_local} {dst_path}'
        if verbose:
            tic(f"Uploading directory {src_path_local} to S3")
            hprint_message("target", dst_path)
        subprocess.call(cmd_get, shell=True)

        if not ls_dir(dst_path, mlps3=mlps3, print_out=verbose):
            raise RuntimeError(f"Uploading from '{src_path_local}' to '{dst_path}' failed.")

        put_dir(src_path_local=src_path_local, dst_path=dst_path, mlps3=mlps3, verbose=verbose, overwrite=overwrite)  # double check in case of partial upload
        if verbose:
            toc("Done!")
    else:
        uploaded_file_basenames = [
            path.basename(_path)
            for _path in uploaded_paths
            if _path[-1] != path.sep
        ]  # a dir s3 path ends with '/', so its basename is empty anyway
        local_basenames = listdir(src_path_local)
        _basenames_to_check = set(local_basenames) - set(uploaded_file_basenames)
        if _basenames_to_check:
            for _basename in _basenames_to_check:
                local_path = path.join(src_path_local, _basename)
                s3_path = path.join(dst_path, _basename)
                if path.isfile(local_path):
                    put_file(src_path_local=local_path, dst_path=s3_path)
                else:
                    put_dir(src_path_local=local_path, dst_path=s3_path, mlps3=mlps3, verbose=verbose, overwrite=overwrite)
    return dst_path
