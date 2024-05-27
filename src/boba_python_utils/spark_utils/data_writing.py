import logging
from functools import partial
from typing import Callable

from boba_python_utils.common_utils.function_helper import get_relevant_named_args
from boba_python_utils.general_utils.console_util import hprint_pairs, hprint_message
from boba_python_utils.spark_utils import VERBOSE
from boba_python_utils.spark_utils.common import *


def get_write_method(
        df: DataFrame,
        format: str = 'json',
        overwrite: bool = True,
        compress: Union[bool, str] = 'gzip',
        **kwargs
) -> Callable:
    """
    Retrieve the writing method from get_spark according to the `format` and other arguments.
    Args:
        df: the dataframe to save to files.
        format: a string like 'json', 'parquet', 'csv', if that is supported by spark.
        overwrite: True to overwrite data at the existing path; otherwise False.
        compress: True to compress the output by gzip; or specify a supported compression format
                like 'bzip2', 'gzip', 'lz4'; otherwise, False.
        **kwargs: other named argument for the write method (specific to the `format`).

    Returns: a function that can write the datafram `df` to files.

    """
    if compress is True:
        compress = 'gzip'
    _df_write = df.write.mode('overwrite') if overwrite else df.write

    if hasattr(_df_write, format):
        df_write = getattr(_df_write, format)
    else:
        raise ValueError('format `{}` is not supported yet'.format(format))

    _related_kwargs, _unrelated_kwargs = get_relevant_named_args(df_write, return_other_args=True, **kwargs)

    if compress:
        _related_kwargs['compression'] = compress

    for k, v in _unrelated_kwargs.items():
        _df_write = _df_write.option(k, v)

    df_write = getattr(_df_write, format)

    return partial(df_write, **_related_kwargs) if _related_kwargs else df_write


def write_df(
        df: DataFrame,
        output_path: Union[str, Iterable[str]],
        num_files: int = 100,
        compress: Union[bool, str] = True,
        show_counts: bool = False,
        cache_before_writing: bool = False,
        repartition: bool = True,
        overwrite: bool = True,
        format: str = 'json',
        unpersist: bool = False,
        name: str = None,
        logger: logging.Logger = None,
        verbose: bool = VERBOSE,
        **kwargs
) -> DataFrame:
    """
    Write a get_spark dataframe to files.

    Args:
        df: the dataframe to save to files.
        output_path: the output path to write to.
        num_files: split the data into this specified number of files.
        compress: True to compress the output by gzip; or specify a supported compression format
                like 'bzip2', 'gzip', 'lz4'; otherwise, False.
        show_counts: True to print out the size of the dataframe on the terminal; otherwise, False.
        cache_before_writing: True to cache the dataframe before writing; otherwise, False.
        repartition: True to repartition the data before writing;
            enable this for better writing performance.
        overwrite: True to overwrite data at the existing path; otherwise False.
        kwargs: other named argument for the write method (specific to the `format`).
        format: a string like 'json', 'parquet', 'csv', if that is supported by spark.

    Returns: the input dataframe `df`.

    """

    if isinstance(output_path, (list, tuple)):
        for i, _output_opath in enumerate(output_path):
            write_df(
                df=df,
                output_path=_output_opath,
                num_files=num_files,
                compress=compress,
                show_counts=show_counts,
                cache_before_writing=cache_before_writing and (i == 0),
                repartition=repartition and (i == 0),
                overwrite=overwrite,
                format=format,
                unpersist=False,
                name=name,
                logger=logger,
                verbose=verbose,
                **kwargs
            )

    else:
        if not isinstance(df, DataFrame):
            raise ValueError(f"the input for 'write_df' must be a Spark dataframe; got {df}")

        # region enforced verbosity
        if name:
            hprint_message(f"write dataframe '{name}' to", output_path, logger=logger)
        else:
            hprint_message('write dataframe to', output_path, logger=logger)
        # endregion

        if not format:
            format = 'json'

        if num_files:
            df = df.repartition(num_files) if repartition else df.coalesce(num_files)

        if cache_before_writing or show_counts:
            if cache_before_writing:
                df = df.cache()
            num_rows = df.count()
        else:
            num_rows = 'unknown'

        write_method = get_write_method(
            df=df,
            format=format,
            overwrite=overwrite,
            compress=compress,
            **kwargs
        )

        # region verbosity
        if verbose:
            hprint_pairs(
                ('name', name),
                ('output_path', output_path),
                ('num_files', num_files),
                ('compress', compress),
                ('write_method', write_method),
                ('num_rows', num_rows),
                logger=logger
            )
        # endregion

        write_method(output_path)

    if unpersist:
        df.unpersist()
    return df


def repartition_data_files(
        input_path: str,
        target_num_files: int,
        spark: SparkSession,
        output_path: str = None,
        tmp_path: str = None,
        data_format: str = None,
        compress: bool = True,
        **write_df_kwargs
):
    if not output_path:
        output_path = input_path
    if not tmp_path:
        tmp_path = output_path + '---tmp'

    from boba_python_utils.spark_utils.data_loading import solve_input

    df = solve_input(
        input_path,
        input_format=data_format,
        spark=spark
    )

    write_df(
        df,
        tmp_path,
        repartition=True,
        num_files=target_num_files,
        format=data_format,
        compress=compress,
        **write_df_kwargs
    )

    df = solve_input(
        tmp_path,
        input_format=data_format,
        spark=spark
    )

    write_df(
        df,
        output_path,
        repartition=True,
        num_files=target_num_files,
        format=data_format,
        compress=compress,
        **write_df_kwargs
    )
