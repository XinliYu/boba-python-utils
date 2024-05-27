import warnings

try:
    import mlp_s3

    mlps3_client = mlp_s3.client()
    mlp_s3_load_success = True
except:
    mlps3_client = None
    mlp_s3_load_success = False
    warnings.warn("failed to load 'mlp_s3' module")

CMD_MLPS3 = "/apollo/env/HoverboardDefaultMLPS3Tool/bin/mlps3"
CMD_MLPS3_WITH_ENV_MOD = f"PYTHONPATH='' LD_LIBRARY_PATH='' {CMD_MLPS3}"
URL_SCHEME_SEP = '://'
S3_URL_SCHEME = 's3' + URL_SCHEME_SEP
