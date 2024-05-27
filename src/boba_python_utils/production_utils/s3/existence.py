import os
import posixpath
from typing import Callable, Union, Tuple, Iterable, Mapping, List

from pyspark.sql import DataFrame

from boba_python_utils.common_utils.function_helper import get_relevant_named_args
from boba_python_utils.common_utils.iter_helper import zip_longest__
from boba_python_utils.common_utils.typing_helper import of_type_all
from boba_python_utils.production_utils.s3 import mlps3_client
from boba_python_utils.production_utils.s3.common import get_s3_bucket_and_key

FILENAME_PYSPARK_SUCCESS = '_SUCCESS'


def exists_path(s3path, mlps3=None, **kwargs):
    if mlps3 is None:
        mlps3 = mlps3_client
    bucket, key = get_s3_bucket_and_key(s3path)
    if key[-1] != os.sep:
        if mlps3.exists(bucket, key, **kwargs):
            return True
    else:
        key = key[:-1]
    results: List[str] = mlps3.list_prefixes(bucket, key, **kwargs)
    key += os.sep
    return len(results) != 0 and any(x.startswith(key) for x in results)


def exists_dir(s3path, mlps3=None, **kwargs):
    if mlps3 is None:
        mlps3 = mlps3_client
    bucket, key = get_s3_bucket_and_key(s3path)
    if key[-1] == os.sep:
        key = key[:-1]
    results: List[str] = mlps3.list_prefixes(bucket, key, **kwargs)
    key += os.sep
    return len(results) != 0 and any(x.startswith(key) for x in results)


def exists_file(s3path, mlps3=None, **kwargs):
    if mlps3 is None:
        mlps3 = mlps3_client
    bucket, key = get_s3_bucket_and_key(s3path)
    return mlps3.exists(bucket, key, **get_relevant_named_args(mlps3.exists, **kwargs))


def exists_spark_success(s3dir: str, mlps3=None, **kwargs):
    if not s3dir.endswith(FILENAME_PYSPARK_SUCCESS):
        s3dir = posixpath.join(s3dir, FILENAME_PYSPARK_SUCCESS)
    return exists_file(s3dir, mlps3=mlps3, **kwargs)


def _write_spark_dataframe(spark_write_method, df_agg, s3path_output, compression):
    if compression is None:
        spark_write_method(df_agg, s3path_output)
    else:
        spark_write_method(df_agg, s3path_output, compression)


def execute_spark_aggregation_if_not_exist_spark_success(
        s3path: Union[str, Iterable[str]],
        spark_aggregation_method: Callable[[], DataFrame],
        spark_read_method: Callable[[str], DataFrame],
        spark_write_method: Union[
            Callable[[DataFrame, str, bool], None],
            Callable[[DataFrame, str], None]
        ],
        spark_aggregation_name: str = None,
        log_message_method: Callable[[str], None] = None,
        force=False,
        save_aggregation: bool = True,
        return_dataframe: bool = False,
        compression=None,
) -> Union[str, Tuple[str], DataFrame, Tuple[DataFrame]]:
    if isinstance(s3path, str):
        # ! if the input is a single string,
        # !   then this function returns a signal dataframe if `return_dataframe` is set true,
        # !   or the single-string path itself
        s3path_list = [s3path]
        is_input_s3path_str = True
    else:
        # ! otherwise, we return a tuple of dataframes or correspond to the input paths,
        # !   or this path tuple itself;
        # !   this applies even if the input path list contains a single path
        is_input_s3path_str = False
        if isinstance(s3path, Mapping):
            s3path_list = tuple(s3path.values())
        elif isinstance(s3path, (list, tuple)):
            s3path_list = s3path
        else:
            s3path_list = tuple(s3path)

    all_spark_aggregations_exist = True
    for _s3path_output in s3path_list:
        if exists_spark_success(_s3path_output):
            if log_message_method is not None:
                log_message_method(f"spark aggregation already exists at path '{_s3path_output}'")
        else:
            all_spark_aggregations_exist = False

    if all_spark_aggregations_exist:
        if log_message_method is not None:
            log_message_method(
                "all get_spark aggregations {}already exist".format(
                    f"of '{spark_aggregation_name}'" if spark_aggregation_name else ''
                )
            )
        if force:
            if is_input_s3path_str:
                if log_message_method is not None:
                    log_message_method(
                        "spark aggregation {}".format(
                            f'{spark_aggregation_name}' if spark_aggregation_name else ''
                        ) +
                        f"already exists at path '{s3path}'; force overwrite"
                    )
            else:
                if log_message_method is not None:
                    log_message_method(
                        "spark aggregations {}".format(
                            f"of '{spark_aggregation_name}'" if spark_aggregation_name else ''
                        ) +
                        f"already exist at paths '{s3path_list}'; force overwrite"
                    )
        else:
            if return_dataframe:
                if is_input_s3path_str:
                    return spark_read_method(s3path)
                else:
                    if return_dataframe is not None and return_dataframe is not False:
                        if isinstance(return_dataframe, int):
                            return spark_read_method(s3path_list[return_dataframe])
                        elif (
                                isinstance(return_dataframe, (list, tuple)) and
                                of_type_all(return_dataframe, int)
                        ):
                            s3path_list = [s3path_list[i] for i in return_dataframe]
                    return tuple(
                        spark_read_method(_s3path_output_index)
                        for _s3path_output_index in s3path_list
                    )
            else:
                return s3path if is_input_s3path_str else s3path_list

    df_agg = spark_aggregation_method()
    df_agg_list = (df_agg,) if is_input_s3path_str else df_agg

    for _s3path_output, _df_agg, _save_aggregation, _compression in zip_longest__(
            s3path_list, df_agg_list, save_aggregation, compression,
            fill_none_by_previous_values=[False, False, True, True]
    ):
        if (
                _df_agg is not None and
                _s3path_output is not None and
                (_save_aggregation or _save_aggregation is None)
        ):
            _write_spark_dataframe(
                spark_write_method, _df_agg, _s3path_output, _compression
            )

    if return_dataframe:
        return df_agg if is_input_s3path_str else df_agg_list
    else:
        if is_input_s3path_str:
            df_agg.unpersist()
            return s3path
        else:
            for _df_agg in df_agg:
                if _df_agg is not None:
                    _df_agg.unpersist()
            return s3path_list
