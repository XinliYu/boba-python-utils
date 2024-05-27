import subprocess
from os import path, linesep
from boba_python_utils.production_utils.s3 import CMD_MLPS3_WITH_ENV_MOD, S3_URL_SCHEME
import sys

from boba_python_utils.production_utils.s3.common import sanitize_s3url


def _proc_file_ext(file_ext):
    if file_ext and file_ext[0] == '.':
        return file_ext[1:]
    return file_ext


def list_path(s3url, file_ext=None, mlps3=CMD_MLPS3_WITH_ENV_MOD, top=False, full_path=True, print_out=False, skip_private_files=True):
    s3url, file_ext = sanitize_s3url(s3url), _proc_file_ext(file_ext)
    cmd = "{} ls {}".format(mlps3, path.join(s3url, '*.' + file_ext)) if file_ext else "{} ls {}".format(mlps3, s3url)
    try:
        results = subprocess.check_output(cmd, shell=True).decode('UTF-8' if sys.stdout.encoding is None else sys.stdout.encoding).strip().split(linesep)
    except:
        return []

    if not results[0].startswith('ObjectNotFoundException:'):
        if top:
            if skip_private_files:
                result = ''
                for i in range(len(results)):
                    if results[i][-1] == '/' or path.basename(results[i])[0] != '_':
                        result = results[i]
                        break
            else:
                result = results[0]

            if result:
                if full_path:
                    result = S3_URL_SCHEME + path.join(s3url.split(path.sep)[2], result)
                if print_out:
                    print("{}{} file at {}:".format(('Single' if len(results) == 1 else 'Top'), (' ' + file_ext if file_ext else ''), s3url))
                    print(result)
            else:
                print("No {} file is found at {}:".format((' ' + file_ext if file_ext else ''), s3url))
            return result
        else:
            if skip_private_files:
                results2 = []
                for i in range(len(results)):
                    if results[i][-1] == '/' or path.basename(results[i])[0] != '_':
                        results2.append(results[i])
                results = results2
            if not results:
                print("No {} file is found at {}:".format((' ' + file_ext if file_ext else ''), s3url))
            else:
                if full_path:
                    bucket = s3url.split(path.sep)[2]
                    for i in range(len(results)):
                        results[i] = S3_URL_SCHEME + path.join(bucket, results[i])
                if print_out:
                    print("{}{} file{} at `{}`:".format(len(results), (' ' + file_ext if file_ext else ''), ('' if len(results) == 1 else 's'), s3url))
            return results
    elif print_out:
        print(results[0])


def ls_dir(s3url, file_ext=None, mlps3=CMD_MLPS3_WITH_ENV_MOD, top=False, full_path=True, print_out=True, skip_private_files=True):
    if s3url[-1] != path.sep:
        s3url += path.sep
    return list_path(
        s3url=s3url,
        file_ext=file_ext,
        mlps3=mlps3,
        full_path=full_path,
        top=top,
        print_out=print_out,
        skip_private_files=skip_private_files
    )
