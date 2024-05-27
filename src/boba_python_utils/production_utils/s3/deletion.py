import os

from boba_python_utils.production_utils.s3 import mlps3_client
from boba_python_utils.production_utils.s3.common import get_s3_bucket_and_key
from boba_python_utils.production_utils.s3.existence import exists_dir


def delete_path(s3path, mlps3=None, **kwargs):
    if mlps3 is None:
        mlps3 = mlps3_client
    bucket, key = get_s3_bucket_and_key(s3path)
    if key[-1] != os.sep:
        if mlps3.exists(bucket, key, **kwargs):
            try:
                mlps3.delete_object(bucket, key, **kwargs)
            except:
                mlps3.delete_prefix(bucket, key, **kwargs)
    else:
        key = key[:-1]
        if exists_dir(s3path, mlps3=mlps3, **kwargs):
            mlps3.delete_prefix(bucket, key, **kwargs)
