import logging
from os import path

from pyspark import StorageLevel

from boba_python_utils.general_utils.argex import (
    parse_args_from_str
)
from boba_python_utils.common_utils.iter_helper import iter__, all__
from boba_python_utils.common_utils.typing_helper import is_basic_type_or_basic_type_iterable, all_str
from boba_python_utils.general_utils.console_util import (
    hprint_message, hprint_pairs,
)
from boba_python_utils.general_utils.general import (
    has_parameter,
    is_first_parameter_varpos,
    get_relevant_named_args,
)
from boba_python_utils.io_utils.text_io import read_all_lines
from boba_python_utils.spark_utils import VERBOSE
from boba_python_utils.spark_utils.analysis import show_count
from boba_python_utils.spark_utils.common import *
from boba_python_utils.spark_utils.common import CacheOptions
from boba_python_utils.spark_utils.data_writing import write_df


def _proc_dataframe(df: DataFrame, spark: SparkSession, verbose: bool = VERBOSE, **kwargs):
    from boba_python_utils.spark_utils.join_and_filter import (
        exclude_by_anti_join_on_columns, filter_by_inner_join_on_columns, where
    )
    from boba_python_utils.spark_utils.data_transform import (
        explode_as_flat_columns, with_columns, rename, fillna
    )
    _df_cached = df if df.is_cached else None
    for _arg_name, _arg in kwargs.items():
        if _arg_name == 'explode':
            if _arg is not None:
                df = explode_as_flat_columns(df, _arg, overwrite_exist_column=True)
        elif _arg_name == 'rename':
            df = rename(df, _arg)
        elif _arg_name == 'with_columns':
            # arg emptiness already handled in `with_columns` function
            df = with_columns(df, _arg)
        elif _arg_name == 'where':
            # specifying 'null_cond_tolerance' might force the dataframe to be cached;
            # and in this case we make sure the dataframe is repartitioned in case data is skewed
            if (
                    'null_cond_tolerance' in kwargs and
                    (
                            'cache_option_for_null_cond_check' not in kwargs or
                            kwargs['cache_option_for_null_cond_check'] != CacheOptions.NO_CACHE
                    )
            ):
                df = repartition(df, spark=spark)
            # arg emptiness already handled in `where` function
            df = where(df, _arg, verbose=verbose, **get_relevant_named_args(where, **kwargs))
        elif _arg_name == 'filter':
            df = filter_by_inner_join_on_columns(df, _arg)
        elif _arg_name == 'exclusion':
            df = exclude_by_anti_join_on_columns(df, _arg)
        elif _arg_name == 'select':
            if _arg:  # handle arg emptiness
                df = df.select(*iter__(_arg))
        elif _arg_name == 'fillna':
            if isinstance(_arg, str):
                df = df.fillna(_arg)
            else:
                df = fillna(df, _arg)
        elif _arg_name == 'transformation':
            df = _arg(df)
        elif _arg_name == 'drop':
            if isinstance(_arg, str):
                df = df.drop(_arg)
            elif _arg is not None:
                df = df.drop(*_arg)
        elif _arg_name == 'distinct':
            if _arg:
                if isinstance(_arg, bool):
                    df = df.distinct()
                else:
                    df = df.dropDuplicates(_arg)
        elif _arg_name == 'sample':
            if _arg:
                df = df.sample(_arg)
        elif _arg_name == 'repartition':
            if _arg is True:
                df = repartition(df, spark=spark)
            elif isinstance(_arg, (int, float)):
                df = repartition(df, _arg, spark=spark)
            elif isinstance(_arg, (list, tuple)):
                if all_str(_arg):
                    df = repartition(df, None, *_arg, spark=spark)
                elif isinstance(_arg[0], (int, float)) and all_str(_arg[1:]):
                    df = repartition(df, *_arg, spark=spark)
                else:
                    raise ValueError(f"invalid repartition argument {_arg}")
            else:
                raise ValueError(f"invalid repartition argument {_arg}")
        elif _arg_name == 'join':
            if _arg:
                df = df.join(
                    _arg[0],
                    _arg[1],
                    how=('inner' if len(_arg) == 2 else _arg[2])
                )

        if df.is_cached:
            if _df_cached is None:
                # input dataframe is not cached, but the current transformation cached the dataframe,
                # then set `_df_cached` to the current dataframe is cached
                _df_cached = df
            elif _df_cached is not df:
                # input dataframe is cached,
                # or some previous transformation cached the dataframe,
                # and the current transformation also cached the dataframe,
                # then we unpersist the previously cached dataframe to release memory,
                # and set `_df_cached` to the current dataframe
                _df_cached.unpersist()
                _df_cached = df

    if (
            _df_cached is not None and
            _df_cached is not df and
            (not df.is_cached)
    ):
        df = df.cache()
        df.count()
        _df_cached.unpersist()

    return df


def create_one_column_dataframe_from_iterable(
        items: Iterable[Any],
        spark: SparkSession,
        colname: str = 'col'
):
    if spark is None:
        raise ValueError("Spark session object must be provided for "
                         "creating a dataframe from an iterable")
    return spark.createDataFrame(
        [
            (
                (
                    x if is_basic_type_or_basic_type_iterable(x, iterable_type=list)
                    else str(x)
                ),  # ! do not omit the comma
            ) for x in items
        ],
        [colname]
    )


def read_csv(spark: SparkSession, input_path, schema=None, header=None, delimiter='\t', **kwargs):
    read_obj = spark.read
    if header is True:
        read_obj = read_obj.option("header", True)
    elif header is False:
        read_obj = read_obj.option("header", False)
    elif isinstance(header, str):
        header = [header]

    if schema is None:
        read_obj = read_obj.option("inferSchema", True)

    read_obj = read_obj.option("delimiter", delimiter)

    df = read_obj.csv(input_path, schema=schema, **kwargs)
    return df.toDF(*header) if isinstance(header, (list, tuple)) else df


def read_df(
        input_path: Union[str, List[str]],
        spark: SparkSession, schema=None,
        format='json',
        name: str = None,
        logger: logging.Logger = None,
        verbose=VERBOSE,
        **kwargs
) -> DataFrame:
    """
    Reads a get_spark dataframe from files.

    Args:
        spark: provides a get_spark session object.
        input_path: the input path to read from.
        schema: provides dataframe schema.
        format: a string like 'json', 'parquet', 'csv', if that is supported by spark.
        kwargs: other named argument for the reading method (specific to the `format`).

    Returns: the dataframe read from the files.

    """

    # region enforced verbosity
    if name:
        hprint_message(f"read dataframe '{name}' from", input_path, logger=logger)
    else:
        hprint_message('read dataframe from', input_path, logger=logger)
    # endregion

    (
        format, _args_parsed_from_format_str, _named_args_parsed_from_format_str
    ) = parse_args_from_str(format)

    # region verbosity
    def _verbose():
        if verbose:
            hprint_pairs(
                ('format', format),
                ('read_method', read_method),
                *(
                    (k, v) for k, v in kwargs.items()
                ),
                logger=logger
            )

    # endregion

    spark_read = spark.read

    def _set_spark_read():
        nonlocal spark_read
        for k, v in kwargs.items():
            spark_read = spark_read.option(k, v)

    if format == 'json':
        _kwargs, kwargs = get_relevant_named_args(
            spark_read.json,
            return_other_args=True,
            **kwargs
        )
        _set_spark_read()
        kwargs = _kwargs
        read_method = spark.read.json
        _verbose()
        return read_method(input_path, schema=schema, **kwargs)
    elif format == 'parquet':
        _kwargs, kwargs = get_relevant_named_args(
            spark_read.parquet,
            return_other_args=True,
            **kwargs,
        )
        _set_spark_read()
        kwargs = _kwargs
        read_method = spark.read.parquet
        _verbose()
        if isinstance(input_path, str):
            return read_method(input_path, **kwargs)
        else:
            return read_method(*input_path, **kwargs)
    elif format == 'csv':
        _kwargs, kwargs = get_relevant_named_args(
            [read_csv, spark_read.csv],
            return_other_args=True,
            **kwargs
        )
        _set_spark_read()
        kwargs = _kwargs
        if _args_parsed_from_format_str:
            kwargs['header'] = _args_parsed_from_format_str
        if _named_args_parsed_from_format_str:
            kwargs.update(_named_args_parsed_from_format_str)
        read_method = read_csv
        _verbose()
        return read_method(spark=spark, input_path=input_path, schema=schema, **kwargs)
    elif format == 'list':
        if path.exists(input_path):
            list_data = read_all_lines(input_path)
        elif _named_args_parsed_from_format_str and 'sep' in _named_args_parsed_from_format_str:
            list_data = input_path.split(_named_args_parsed_from_format_str['sep'])
        else:
            list_data = input_path.split()
        kwargs = {}
        read_method = create_one_column_dataframe_from_iterable
        _verbose()
        return read_method(
            list_data,
            colname='value' if (not _args_parsed_from_format_str)
            else _args_parsed_from_format_str[0],
            spark=spark
        )
    elif hasattr(spark_read, format):
        read_method = getattr(spark_read, format)
        _kwargs, kwargs = get_relevant_named_args(
            read_method,
            return_other_args=True,
            **kwargs
        )
        _set_spark_read()
        kwargs = _kwargs
        read_method = getattr(spark_read, format)
        kwargs = get_relevant_named_args(read_method, **kwargs)
        _verbose()
        has_schema_arg = has_parameter(read_method, 'schema')
        first_parameter_varpos = is_first_parameter_varpos(read_method)
        if has_schema_arg:
            if isinstance(input_path, str) or not first_parameter_varpos:
                return read_method(input_path, schema=schema, **kwargs)
            else:
                return read_method(*input_path, schema=schema, **kwargs)
        else:
            if isinstance(input_path, str) or not first_parameter_varpos:
                return read_method(input_path, **kwargs)
            else:
                return read_method(*input_path, **kwargs)
    else:
        raise ValueError(f"format '{format}' is not supported")


def _hprint_dataframe_info(df: DataFrame, name: str, logger: logging.Logger = None):
    _msg_pairs = []
    if name:
        _msg_pairs.append(('name', name))
    _msg_pairs.extend(
        (
            ('partitions', df.rdd.getNumPartitions()),
            ('storage level', df.rdd.getStorageLevel()),
            ('columns', df.columns)
        )
    )
    hprint_pairs(
        *_msg_pairs,
        logger=logger
    )


def _solve_input(
        index,
        input: Union[str, List[str], DataFrame],
        spark: SparkSession,
        input_format=None,
        schema=None,
        name: str = None,
        logger: logging.Logger = None,
        verbose=VERBOSE,
        **kwargs
):
    if name and index is not None:
        name = f'{name} ({index})'

    if isinstance(input, DataFrame):
        df_out = input
    else:
        if callable(input):
            if has_parameter(input, 'spark'):
                kwargs['spark'] = spark
            if has_parameter(input, 'input_format'):
                kwargs['input_format'] = input_format
            if has_parameter(input, 'schema'):
                kwargs['schema'] = schema
            return input(**kwargs)

        if input_format is None:
            input_format = 'json'
            if all__(input, cond=lambda _x: 'parquet' in _x):
                input_format = 'parquet'
            elif all__(input, cond=lambda _x: 'csv' in _x):
                input_format = 'csv'

        if spark is None:
            raise ValueError(
                f"spark session instance must be provided to "
                f"load a {input_format} file at '{input}'"
            )

        df_out = read_df(
            input_path=input,
            spark=spark,
            schema=schema,
            format=input_format,
            name=name,
            logger=logger,
            **kwargs
        )

    # region verbosity
    if verbose:
        _hprint_dataframe_info(df_out, name=name, logger=logger)
    # endregion

    df_out_processed = _proc_dataframe(df_out, spark=spark, verbose=verbose, **kwargs)

    # region verbosity
    _is_df_out_processed = (df_out_processed is not df_out)
    if verbose:
        hprint_message(
            (f"is dataframe '{name}' processed" if name else 'is dataframe processed'),
            _is_df_out_processed,
            logger=logger
        )
    if _is_df_out_processed:
        if verbose:
            _hprint_dataframe_info(
                df_out_processed,
                name=f'{name} (processed)' if name else 'dataframe processed',
                logger=logger
            )
    # endregion

    return df_out_processed


def solve_input(
        input: Union[str, List[str], DataFrame, Mapping],
        spark: SparkSession,
        input_format=None,
        schema=None,
        name: str = None,
        logger: logging.Logger = None,
        verbose=VERBOSE,
        **kwargs
):
    df_out = apply_func(
        func=partial(
            _solve_input,
            spark=spark,
            input_format=input_format,
            schema=schema,
            name=name,
            logger=logger,
            verbose=verbose,
            **kwargs
        ),
        input=input,
        seq_type=tuple,
        mapping_type=Mapping,
        pass_seq_index=True,
        pass_mapping_key=True
    )
    return df_out


def cache__(
        df_or_path: Union[str, List[str], DataFrame],
        name: str = None,
        unpersist: Union[DataFrame, Tuple[DataFrame, ...], List[DataFrame], bool] = None,
        spark: SparkSession = None,
        input_format: str = None,
        schema: object = None,
        dump_path: str = None,
        dump_num_files: int = 100,
        dump_compress: bool = True,
        dump_format: Union[bool, str] = None,
        cache_retry: int = 1,
        storage_level=StorageLevel.MEMORY_AND_DISK,
        cache_option: CacheOptions = CacheOptions.IMMEDIATE,
        return_count: bool = False,
        logger: logging.Logger = None,
        verbose: bool = VERBOSE,
        **kwargs: object
) -> Union[
    DataFrame,
    Mapping[str, DataFrame],
    Tuple[DataFrame, int],
    Mapping[str, Tuple[DataFrame, int]]
]:
    """

    Args:
        df_or_path: different data loading behavior for the following input types -
            1) data frame,
            2) path,
            3) list of paths
            4) tuple of dataframes/paths
            5) a mapping from data alias to a dataframe or path or a list of paths
        name:
        unpersist:
        spark:
        input_format:
        schema:
        dump_path:
        dump_num_files:
        dump_compress:
        dump_format:
        cache_retry:
        cache_option:
        return_count:
        logger:
        verbose:
        **kwargs:

    Returns:

    """
    if df_or_path is None:
        raise ValueError("'df' cannot be None")

    df: DataFrame = solve_input(
        input=df_or_path,
        spark=spark,
        input_format=input_format,
        schema=schema,
        name=name,
        logger=logger,
        verbose=verbose,
        **kwargs
    )

    def _cache(index_or_key, df_to_cache: DataFrame):
        if name and index_or_key is not None:
            _name = f'{name} ({index_or_key})'
        else:
            _name = name

        cnt = None
        if cache_option != CacheOptions.NO_CACHE:
            if not df_to_cache.is_cached:
                df_to_cache = df_to_cache.persist(storageLevel=storage_level)
                df_to_cache = df_to_cache.cache()
            if cache_option == CacheOptions.IMMEDIATE:
                nonlocal cache_retry
                while cache_retry >= 0:
                    try:
                        if _name is None:
                            cnt = df_to_cache.count()
                        else:
                            cnt = show_count(df_to_cache, _name)
                        break
                    except:  # noqa: E722
                        cache_retry -= 1

        if dump_path is not None:
            if index_or_key is None:
                _dump_path = dump_path
            else:
                if isinstance(index_or_key, int):
                    _dump_path = path.join(dump_path, f'{index_or_key:05d}')
                else:
                    _dump_path = path.join(dump_path, index_or_key)
            write_df(
                df_to_cache,
                _dump_path,
                num_files=dump_num_files,
                compress=dump_compress,
                format=dump_format,
            )

        if return_count:
            if cnt is None:
                cnt = df_to_cache.count()
            return df_to_cache, cnt
        else:
            return df_to_cache

    df_out = apply_func(
        func=_cache,
        input=df,
        seq_type=tuple,
        mapping_type=Mapping,
        pass_seq_index=True,
        pass_mapping_key=True
    )

    def _not_in_df_out(_df: DataFrame):
        if isinstance(df_out, DataFrame):
            return _df is not df_out
        elif isinstance(df_out, (list, tuple)):
            return all(_df is not _df_out for _df_out in df_out)
        elif isinstance(df_out, Mapping):
            return all(_df is not _df_out for _df_out in df_out.values())

    if unpersist is not None:
        if isinstance(unpersist, DataFrame):
            if _not_in_df_out(unpersist):
                if verbose:
                    if unpersist.is_cached:
                        hprint_message("dataframe already uncached", unpersist)
                    else:
                        hprint_message("dataframe to be uncached", unpersist)
                unpersist.unpersist()
        elif unpersist is True:
            if isinstance(df_or_path, DataFrame) and _not_in_df_out(df_or_path):
                df_or_path.unpersist()
        else:
            for _df in unpersist:
                if isinstance(_df, DataFrame) and _not_in_df_out(_df):
                    if verbose:
                        if not _df.is_cached:
                            hprint_message("dataframe already uncached", _df)
                        else:
                            hprint_message("dataframe to be uncached", _df)
                    _df.unpersist()

    return df_out
