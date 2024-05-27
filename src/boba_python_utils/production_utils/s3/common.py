import urllib
from typing import Tuple

from boba_python_utils.production_utils.s3 import S3_URL_SCHEME, URL_SCHEME_SEP


def get_s3_bucket_and_key(s3path: str) -> Tuple[str, str]:
    url = urllib.parse.urlparse(s3path)
    return url.netloc, url.path[1:]


def sanitize_s3url(s3url):
    if URL_SCHEME_SEP in s3url:
        s3url = S3_URL_SCHEME + s3url.split(URL_SCHEME_SEP)[1]
    return s3url
