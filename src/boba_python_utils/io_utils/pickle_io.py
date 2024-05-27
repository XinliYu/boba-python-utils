import gzip
import pickle
import sys

from boba_python_utils.path_utils.common import ensure_parent_dir_existence


def pickle_load(file_path: str, compressed: bool = False, encoding=None):
    with open(file_path, 'rb') if not compressed else gzip.open(file_path, 'rb') as f:
        if encoding is None or sys.version_info < (3, 0):
            return pickle.load(f)

        else:
            return pickle.load(f, encoding=encoding)


def pickle_save(data, file_path: str, compressed: bool = False, ensure_dir_exists=True):
    if ensure_dir_exists:
        ensure_parent_dir_existence(file_path)
    with open(file_path, 'wb+') if not compressed else gzip.open(file_path, 'wb+') as f:
        pickle.dump(data, f)
        # ! must flush and close to ensure data completeness
        # ! when this function is called in multi-processing or Spark
        f.flush()
        f.close()
