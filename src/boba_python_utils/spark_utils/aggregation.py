import uuid
from os import path
from typing import Callable, Iterable, List, Mapping, Optional, Tuple, Union

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.column import Column

from boba_python_utils.common_utils.function_helper import compose2, compose
from boba_python_utils.common_utils.iter_helper import iter_
from boba_python_utils.common_utils.typing_helper import all_str, make_tuple_
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.general_utils.general import (
    make_list_,
)
from boba_python_utils.general_utils.nlp_utility.punctuations import remove_punctuation_except_for_hyphen_and_underscore
from boba_python_utils.general_utils.strex import add_prefix
from boba_python_utils.spark_utils import VERBOSE
from boba_python_utils.spark_utils.common import (
    INTERNAL_USE_COL_NAME_PART_SEP,
    INTERNAL_USE_COL_NAME_PREFIX,
    get_colname,
    get_internal_colname,
    has_col,
    solve_name_and_column,
    solve_names_and_columns,
)
from boba_python_utils.spark_utils.data_transform import rename, explode_as_flat_columns
from boba_python_utils.spark_utils.typing import (
    AliasAndColumn,
    AliasesAndColumns,
    NameOrColumn, ColumnsOrAliasedColumns, ColumnOrAliasedColumn,
)


# region aggregations

def with_aggregations(
        df: DataFrame,
        group_cols: Union[NameOrColumn, Iterable[NameOrColumn]],
        agg_cols: Union[NameOrColumn, Iterable[NameOrColumn]],
        how: str = 'left'
):
    """
    Aggregates the DataFrame on specified columns and joins the aggregated result back to the
    original DataFrame.

    Args:
        df: The input DataFrame.
        group_cols: The columns to group the DataFrame by before performing aggregation.
        agg_cols: The columns to perform aggregation on.
        how: The type of join to perform. Defaults to 'left'.

    Returns: A new DataFrame with the aggregated result joined back.

    Examples:
        >>> data = [("A", 1), ("A", 2), ("B", 3), ("B", 4)]
        >>> schema = StructType([
                StructField("key", StringType(), True),
                StructField("value", IntegerType(), True)
            ])
        >>> df = spark.createDataFrame(data, schema=schema)
        >>> from pyspark.sql.functions import sum
        >>> result = with_aggregations(df, group_cols="key", agg_cols=sum("value").alias("sum_value"))
        >>> result.show()
        +---+-----+---------+
        |key|value|sum_value|
        +---+-----+---------+
        |  A|    1|        3|
        |  A|    2|        3|
        |  B|    3|        7|
        |  B|    4|        7|
        +---+-----+---------+
    """
    from boba_python_utils.spark_utils.join_and_filter import join_on_columns
    group_cols = make_list_(group_cols)
    return join_on_columns(
        df,
        df.groupBy(
            group_cols
        ).agg(
            *(iter_(agg_cols))
        ),
        group_cols,
        how=how
    )


def add_group_cols(df: DataFrame, group_cols: Iterable[NameOrColumn]):
    _group_cols = []
    for _col in group_cols:
        if isinstance(_col, str):
            if has_col(df, _col):
                _group_cols.append(_col)
            else:
                raise ValueError(f"column name '{_col}' cannot be found in dataframe {df}")
        else:
            colname = remove_punctuation_except_for_hyphen_and_underscore(get_colname(df, _col))
            df = df.withColumn(colname, _col)
            _group_cols.append(colname)
    return df, _group_cols


def add_agg_cols(df: DataFrame, group_cols, agg_cols, by_join=True):
    group_cols = make_list_(group_cols)

    if by_join:
        return df.join(
            df.groupBy(*group_cols).agg(*agg_cols),
            group_cols
        )
    else:
        group_col_names: List[str] = (
            group_cols if
            all_str(group_cols)
            else get_colname(df, *group_cols)
        )

        _tmp_colname = get_internal_colname('grouped')
        return explode_as_flat_columns(df.groupBy(*group_cols).agg(
            *agg_cols,
            F.collect_list(
                F.struct(
                    *(
                        _col for _col in df.columns
                        if _col not in group_col_names
                    )
                )
            ).alias(_tmp_colname)
        ), col_to_explode=_tmp_colname)


def get_count_and_avg_agg_cols(
        df: DataFrame,
        count_col: Optional[Union[NameOrColumn, AliasAndColumn]] = 'count',
        avg_cols: ColumnsOrAliasedColumns = None,
        avg_by_count_col: bool = False,
        weighted_avg_by_count_col: bool = False,
        count_colname_prefix: str = '',
        avg_colname_prefix: str = '',
        verbose: bool = VERBOSE
) -> Tuple[Column, ...]:
    """
    Gets aggregation columns for the average aggregations and the count.

    Args:
        df: the dataframe.
        count_col: specifies the counting column for the average aggregation.
            If this counting column already resolvable with respect to the dataframe
            (i.e. `df.select(count_col)` works), then we will sum over this this counting column
            for the aggregation; otherwise a counting aggregation column `F.count('*')`
            will be used.

            When `avg_by_count_col` or `weighted_avg_by_count_col` is set True,
            it indicates we must average over the sum of an already resolvable counting column;
            and in this case `count_col_name` must be specified and must be already be resolvable
            with respect to the dataframe.

            See :func:`solve_name_and_column` for allowed column specification format.
        avg_cols: specifies the columns for the average aggregation.
        avg_by_count_col: True to average over the sum of an existing or a resolvable counting
            column specified by `count_col`.
        weighted_avg_by_count_col: True to perform weighted average over
                the sum of an existing or a resolvable counting column specified by `count_col`,
                weighted by the counting column itself.
        count_colname_prefix: adds a prefix to the name of the counting column.
        avg_colname_prefix: adds a prefix to each column name of `avg_cols`.

    Returns: a tuple of average aggregation column and the counting column.

    """
    avg_by_count_col = avg_by_count_col or weighted_avg_by_count_col
    count_colname, count_col = solve_name_and_column(count_col)
    count_colname = add_prefix(count_colname, prefix=count_colname_prefix)

    if avg_by_count_col:
        if not has_col(df, count_col):
            raise ValueError(
                "when 'avg_by_count_col' or 'weighted_avg_by_count_col' is set True, "
                "it indicates we want to average over the sum of an existing counting column;"
                "in this case, "
                "'count_col' must be specified and must be resolvable with respect to the dataframe; "
                f"got {df} and {count_col}"
            )

        if verbose:
            hprint_message(
                "average by existing counting column", count_col,
                'output counting column', count_colname,
                'avg_cols', avg_cols,
                'weighted_avg_by_count_col', weighted_avg_by_count_col
            )

        # assumes the dataframe already contains a count column,
        # and we sum over this count column to obtain a new count column
        agg_cols = (
            (F.sum(count_col).alias(count_colname),)
            if count_col is not None
            else ()
        )
        if weighted_avg_by_count_col:
            # the average is weighted by the count column,
            # i.e. by 'sum(col*count)/sum(count)'
            agg_cols += tuple(
                (
                        F.sum(_col * count_col) / F.sum(count_col)
                ).alias(avg_colname_prefix + _col_name)
                for _col_name, _col in solve_names_and_columns(avg_cols)
            )
        else:
            # the average is not weighted by the count column,
            # i.e. by 'sum(col)/sum(count)'
            agg_cols += tuple(
                (F.sum(_col) / F.sum(count_col)).alias(avg_colname_prefix + _col_name)
                for _col_name, _col in solve_names_and_columns(avg_cols)
            )
    else:
        if verbose:
            hprint_message(
                'creating a new counting column', count_colname,
                'avg_cols', avg_cols
            )

        agg_cols = (
            (
                # if `count_colname` already exists in the dataframe, then sum it over
                (F.sum(count_col).alias(count_colname),)
                if has_col(df, count_col)
                # otherwise, creates the count column
                else (F.count('*').alias(count_colname),)
            )
            if count_col is not None
            else ()
        )
        if avg_cols is not None:
            agg_cols += tuple(
                F.avg(_col).alias(avg_colname_prefix + _col_name)
                for _col_name, _col in solve_names_and_columns(avg_cols)
            )

    return agg_cols


def _get_agg_cols(
        agg_func: Union[str, Callable],
        cols: ColumnsOrAliasedColumns,
        colname_prefix: Optional[str],
        conflict_colnames: Iterable[str] = None
):
    return tuple(
        (
            getattr(F, agg_func) if isinstance(agg_func, str) else agg_func
        )(_col).alias(add_prefix(_col_name, prefix=colname_prefix, avoid_repeat=True))
        for _col_name, _col in solve_names_and_columns(cols)
        if (not conflict_colnames) or (_col_name not in conflict_colnames)
    )


def aggregate(
        df: DataFrame,
        group_cols: Iterable[NameOrColumn],
        count_col: Optional[ColumnOrAliasedColumn] = 'count',
        avg_cols: ColumnsOrAliasedColumns = None,
        avg_by_count_col: bool = False,
        weighted_avg_by_count_col: bool = False,
        max_cols: ColumnsOrAliasedColumns = None,
        min_cols: ColumnsOrAliasedColumns = None,
        sum_cols: ColumnsOrAliasedColumns = None,
        collect_list_cols: Union[NameOrColumn, Mapping[str, Iterable[NameOrColumn]]] = None,
        collect_set_cols: Union[NameOrColumn, Mapping[str, Iterable[NameOrColumn]]] = None,
        collect_list_max_size: int = None,
        collect_set_max_size: int = None,
        concat_list_cols: ColumnsOrAliasedColumns = None,
        concat_set_cols: ColumnsOrAliasedColumns = None,
        concat_list_max_size: int = None,
        concat_set_max_size: int = None,
        other_agg_cols: Iterable[NameOrColumn] = None,
        count_colname_prefix: str = '',
        avg_colname_prefix: str = '',
        max_colname_prefix: str = 'max_',
        min_colname_prefix: str = 'min_',
        sum_colname_prefix: str = 'sum_',
        ignore_agg_cols_of_conflict_names: bool = False,
        verbose: bool = VERBOSE
):
    """
    Performs various commonly used aggregations in one place, grouped by `group_cols`.

    For counting and average, this function calls :func:`get_count_and_avg_agg_cols` to generate
    the counting column and average aggregation columns. The related parameters include `count_col`,
    `avg_cols`, `avg_by_count_col`, `weighted_avg_by_count_col`, `count_colname_prefix` and
    `avg_colname_prefix`; the :func:`get_count_and_avg_agg_cols` function supports an existing
    counting column, and weighted average by the counting column, which is useful if this is
    a further aggregation over existing count, sum and average statistics.

    For max, min, sum, collect-list, collect-set aggregations, this function calls the build-in
    Spark functions. The related parameters include `max_cols`, `min_cols`, `sum_cols`,
    `collect_list_cols`, `collect_set_cols`, `max_colname_prefix`, `min_colname_prefix` and
    `sum_colname_prefix`.

    The function supports list-concat aggregation specified by `concat_list_cols`.

    For other customized aggregations, provide it through the `other_agg_cols` parameter.

    """

    agg_cols_conflict_names = (
        set(get_colname(df, x) for x in group_cols)
        if ignore_agg_cols_of_conflict_names
        else None
    )

    # import pdb; pdb.set_trace()

    agg_cols = get_count_and_avg_agg_cols(
        df=df,
        count_col=count_col,
        avg_cols=avg_cols,
        avg_by_count_col=avg_by_count_col,
        weighted_avg_by_count_col=weighted_avg_by_count_col,
        count_colname_prefix=count_colname_prefix,
        avg_colname_prefix=avg_colname_prefix
    )

    if max_cols:
        agg_cols += _get_agg_cols(
            agg_func='max', cols=max_cols, colname_prefix=max_colname_prefix,
            conflict_colnames=agg_cols_conflict_names
        )
    if min_cols:
        agg_cols += _get_agg_cols(
            agg_func='min', cols=min_cols, colname_prefix=min_colname_prefix,
            conflict_colnames=agg_cols_conflict_names
        )
    if sum_cols:
        agg_cols += _get_agg_cols(
            agg_func='sum', cols=sum_cols, colname_prefix=sum_colname_prefix,
            conflict_colnames=agg_cols_conflict_names
        )
    if concat_list_cols:
        if concat_list_max_size is None:
            agg_func = compose2(F.flatten, F.collect_list)
        else:
            agg_func = lambda _col: F.slice(
                F.flatten(F.slice(F.collect_list(_col), 1, concat_list_max_size)),
                1, concat_list_max_size
            )
        agg_cols += _get_agg_cols(
            agg_func=agg_func,
            cols=concat_list_cols,
            colname_prefix=None,
            conflict_colnames=agg_cols_conflict_names
        )
    if concat_set_cols:
        if concat_set_max_size is None:
            agg_func = compose(F.array_distinct, F.flatten, F.collect_list)
        else:
            agg_func = lambda _col: F.slice(
                F.array_distinct(F.flatten(F.slice(F.collect_list(_col), 1, concat_set_max_size))),
                1, concat_set_max_size
            )
        agg_cols += _get_agg_cols(
            agg_func=agg_func,
            cols=concat_set_cols,
            colname_prefix=None,
            conflict_colnames=agg_cols_conflict_names
        )

    if collect_list_cols:
        if collect_list_max_size is None:
            agg_func = lambda _col: F.collect_list(
                F.struct(*_col)
                if isinstance(_col, (list, tuple))
                else _col
            )
        else:
            agg_func = lambda _col: F.slice(F.collect_list(
                F.struct(*_col)
                if isinstance(_col, (list, tuple))
                else _col
            ), 1, collect_list_max_size)
        agg_cols += _get_agg_cols(
            agg_func=agg_func,
            cols=collect_list_cols,
            colname_prefix=None,
            conflict_colnames=agg_cols_conflict_names
        )
    if collect_set_cols:
        if collect_set_max_size is None:
            agg_func = lambda _col: F.collect_set(
                F.struct(*_col)
                if isinstance(_col, (list, tuple))
                else _col
            )
        else:
            agg_func = lambda _col: F.slice(F.collect_set(
                F.struct(*_col)
                if isinstance(_col, (list, tuple))
                else _col
            ), 1, collect_set_max_size)
        agg_cols += _get_agg_cols(
            agg_func=agg_func,
            cols=collect_set_cols,
            colname_prefix=None,
            conflict_colnames=agg_cols_conflict_names
        )

    if other_agg_cols is not None:
        agg_cols += make_tuple_(other_agg_cols)

    if verbose:
        hprint_message(
            'group_cols', group_cols,
            'agg_cols', agg_cols,
            title='aggregation arguments'
        )

    df_grouped = df.groupBy(*group_cols).agg(*agg_cols)

    return df_grouped


def _solve_group_and_order_cols(
        group_cols: List[Union[str, Column]],
        order_cols: List[Union[str, Column]],
        reverse_order: bool,
        return_colnames=False
) -> Union[
    Tuple[List[Union[str, Column]], List[Union[str, Column]], List[str], List[str]],
    Tuple[List[Union[str, Column]], List[Union[str, Column]]]
]:
    """
    Solves columns for dataframe transformation with both grouping and sorting.

    The implementation of grouping and sorting is specific to each function
    that calls this internal function; but we assume it will do the following,
        - The dataframe will be grouped by columns specified in `group_cols`;
          meanwhile the dataframe will also by sorted by the columns specified in `order_cols`;
          if `order_cols` is not specified, then `group_cols` will be used for sorting;
          `reverse_order` determines if the sorting is reversed (i.e. descendingly).

    If 'group_cols' or 'order_cols' are assigned a single column,
    they will be converted to ta singleton list.

    Returns the solved 'group_cols' and 'order_cols'.

    """
    group_cols = make_list_(group_cols)
    order_cols = group_cols if (not order_cols) else make_list_(order_cols)

    if return_colnames:
        order_colnames = [colname for colname in order_cols if isinstance(colname, str)]
        group_colnames = [colname for colname in group_cols if isinstance(colname, str)]

    if reverse_order:
        order_cols = [
            (F.col(_col).desc() if isinstance(_col, str) else _col.desc()) for _col in order_cols
        ]

    if return_colnames:
        return group_cols, order_cols, group_colnames, order_colnames
    else:
        return group_cols, order_cols


def add_group_order_index(
        df,
        order_index_col_name,
        group_cols,
        order_cols=None,
        reverse_order=False,
        order_index_offset=0,
):
    """
    Adds order index of rows within each group.
    The dataframe identifies groups with `group_cols` as group keys;
        within each group the rows are sorted by `order_cols`;
        a new column of name `order_index_col_name`
        will be created to save the row order index within each group.
    The first order index within each group is defined by `order_index_offset`.

    Args:
        df: the dataframe.
        order_index_col_name: the name of created order index column.
        group_cols: group the dataframe by these specified columns.
        order_cols: order rows by these columns within each group.
        reverse_order: True to reverse the order (i.e. order descendingly);
        order_index_offset: the starting number of the order index; by default it is 0.

    Returns: a dataframe with a new index column starting from `order_index_offset`;
        the column saves row order index within each group
        identified by grouping key columns `group_cols`.

    """
    group_cols, order_cols = _solve_group_and_order_cols(group_cols, order_cols, reverse_order)
    if not order_index_col_name:
        order_index_col_name = _make_default_col_name(group_cols, order_cols, suffix='rank')
    w = Window().partitionBy(*group_cols).orderBy(*order_cols)
    df = df.withColumn(order_index_col_name, F.row_number().over(w) + order_index_offset - 1)
    return df


def add_group_rank_index(
        df,
        rank_index_col_name,
        group_cols,
        order_cols=None,
        reverse_order=False,
        dense=False,
        rank_index_offset=1,
):
    """
    Adds rank of rows within each group.
    The dataframe identifies groups with `group_cols` as group keys;
        within each group the rows are sorted by `order_cols`;
        a new column of name `rank_index_col_name`
        will be created to save the row rank index within each group.
    The first rank index within each group is defined by `rank_index_offset`.

    Args:
        df: the dataframe.
        rank_index_col_name: the name of created rank index column.
        group_cols: group the dataframe by these specified columns.
        order_cols: order rows by these columns within each group.
        reverse_order: True to reverse the order (i.e. order descendingly).
        dense: True to apply dense rank.
            See `pyspark.sql.functions.rank` vs. `pyspark.sql.functions.dense_rank`.
        rank_index_offset: the starting number of the order index; by default it is 1.

    Returns: a dataframe with a new index column starting from `rank_index_col_name`;
        the column saves row order index within each group
        identified by grouping key columns `group_cols`.

    """
    group_cols, order_cols = _solve_group_and_order_cols(group_cols, order_cols, reverse_order)
    if not rank_index_col_name:
        rank_index_col_name = _make_default_col_name(group_cols, order_cols, suffix='rank')
    w = Window().partitionBy(*group_cols).orderBy(*order_cols)
    df = df.withColumn(
        rank_index_col_name,
        (F.dense_rank if dense else F.rank)().over(w) + rank_index_offset - 1
    )
    return df


def _make_default_col_name(*list_of_cols, suffix):
    return (  # noqa: E126
            INTERNAL_USE_COL_NAME_PREFIX.join(
                INTERNAL_USE_COL_NAME_PART_SEP.join(map(lambda x: str(x).replace('.', '_'), cols))
                for cols in list_of_cols
            )
            + INTERNAL_USE_COL_NAME_PREFIX
            + suffix
    )


def top_from_each_group(
        df: DataFrame,
        top: Union[int, float],
        group_cols: Union[NameOrColumn, Iterable[NameOrColumn]],
        order_cols: Union[NameOrColumn, Iterable[NameOrColumn]] = None,
        reverse_order: bool = False,
        rank_colname: str = None,
        verbose: bool = VERBOSE,
):
    """
    Groups the DataFrame by `group_cols`, sorts the rows in each group by `order_cols`,
    and selects the `top` rows from each group.

    Args:
        df: Input DataFrame.
        top: Number of top rows to select (int) or percentage of rows to select (float).
        group_cols: Column names to group by.
        order_cols: Column names to order the rows within each group.
        reverse_order: If True, sorts the rows in the reverse order.
        rank_colname: If specified, the ranking within each group will be preserved in this column
            in the returned dataframe.
        verbose: If True, prints additional information.

    Returns:
        DataFrame: A DataFrame with the top rows from each group.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> data = [("A", 1), ("A", 2), ("B", 3), ("B", 4), ("B", 5)]
        >>> schema = ["group_col", "value"]
        >>> df = spark.createDataFrame(data, schema=schema)
        >>> result = top_from_each_group(df, top=1, group_cols="group_col", order_cols="value")
        >>> result.show()
        +---------+-----+
        |group_col|value|
        +---------+-----+
        |        A|    1|
        |        B|    3|
        +---------+-----+
    """

    group_cols, order_cols = _solve_group_and_order_cols(group_cols, order_cols, reverse_order)
    if not rank_colname:
        rank_colname = _make_default_col_name(group_cols, order_cols, suffix='rank')
        drop_rank_colname = True
    else:
        drop_rank_colname = False

    if isinstance(top, float) and (0 < top < 1.0):
        from alexa_slab_affinity_learning.utils.spark_utils.analysis import get_counts

        group_counts = get_counts(
            df,
            group_cols=group_cols,
            ratio_colname=None,
            verbose=verbose
        ).collect()
        top_cond = F.multi_when_otherwise(
            *(
                (
                    F.and_(
                        *(
                            (F.col_(_group_col) == _row[i])
                            for i, _group_col in enumerate(group_cols)
                        )
                    ),
                    (F.col(rank_colname) <= int(_row[-1] * top))
                ) for _row in group_counts
            )
        )
    else:
        top_cond = (F.col(rank_colname) <= top)

    if verbose:
        hprint_message(
            'group_cols', group_cols,
            'order_cols', order_cols,
            'reverse_order', reverse_order,
            'top_cond', top_cond
        )

    w = Window().partitionBy(*group_cols).orderBy(*order_cols)
    df = df.withColumn(
        rank_colname, F.row_number().over(w)
    ).where(
        top_cond
    )
    if drop_rank_colname:
        df = df.drop(rank_colname)
    return df


def one_from_each_group(df, group_cols, order_cols=None, reverse_order=False):
    """
    Calls `top_from_each_group` to pick the `top-1` row from each group.

    Args:
        df: Input DataFrame.
        group_cols: Column names to group by.
        order_cols: Column names to order the rows within each group.
        reverse_order: If True, sorts the rows in the reverse order.

    Returns:
        DataFrame: A DataFrame with one row from each group.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> data = [("A", 1), ("A", 2), ("B", 3), ("B", 4), ("B", 5)]
        >>> schema = ["group_col", "value"]
        >>> df = spark.createDataFrame(data, schema=schema)
        >>> result = one_from_each_group(df, group_cols="group_col", order_cols="value", reverse_order=True)
        >>> result.show()
        +---------+-----+
        |group_col|value|
        +---------+-----+
        |        A|    2|
        |        B|    5|
        +---------+-----+
    """
    return top_from_each_group(
        df, top=1, group_cols=group_cols, order_cols=order_cols, reverse_order=reverse_order
    )


def add_is_top_label_column(df, is_top_label_col_name, top, group_cols, order_cols=None, reverse_order=False):
    group_cols, order_cols = _solve_group_and_order_cols(group_cols, order_cols, reverse_order)
    w = Window().partitionBy(*group_cols).orderBy(*order_cols)
    return df.withColumn(is_top_label_col_name, F.row_number().over(w) <= top)


def add_is_top_one_label_column(df, is_top_label_col_name, group_cols, order_cols=None, reverse_order=False):
    return add_is_top_label_column(
        df=df,
        is_top_label_col_name=is_top_label_col_name,
        top=1,
        group_cols=group_cols,
        order_cols=order_cols,
        reverse_order=reverse_order
    )


# endregion


# region union
def _sanitize_dataframes(dataframes):
    dataframes = [_df for _df in dataframes if _df is not None]
    if not dataframes:
        return None

    while len(dataframes) == 1:
        dataframes = dataframes[0]
        if isinstance(dataframes, DataFrame):
            # the case when there is a single dataframe in the input
            return dataframes
    return dataframes


def force_union(
        spark: SparkSession,
        tmp_path,
        *dataframes,
        num_files_per_dataframe=100,
        tmp_file_format='json'
):
    from boba_python_utils.spark_utils import solve_input, write_df
    output_paths = []
    for df in dataframes:
        output_path = path.join(tmp_path, str(uuid.uuid4()))
        write_df(
            df, output_path,
            format=tmp_file_format,
            compress=True,
            num_files=num_files_per_dataframe
        )
        output_paths.append(output_path)
    return solve_input(
        [path.join(x, '*') for x in output_paths],
        spark=spark,
        input_format=tmp_file_format
    )


def union(
        *dataframes: Union[DataFrame, List[DataFrame], Mapping[str, DataFrame]],
        allow_missing_columns: bool = False,
        force_union_tmp_path: str = None,
        force_union_tmp_file_format: str = 'json',
        spark: SparkSession = None
) -> Union[DataFrame, None]:
    """
    Union a list of dataframes.
    If the input is a single dataframe, the single dataframe is returned.

    When dataframes have the same columns but columns are of different orders,
        then the DataFrame.union function might return an incorrect dataframe.
        This method handles this issue.

    Args:
        *dataframes: the input dataframes to union.
        allow_missing_columns: allows dataframes to union to have different columns;
            this requires the Spark version supports 'unionByName' function.

    Returns: the union dataframe.

    """

    dataframes = _sanitize_dataframes(dataframes)
    if dataframes is None or isinstance(dataframes, DataFrame):
        return dataframes

    if len(dataframes) == 2:
        # the case when there are only two input objects;
        # in this case we allow the second input be a list of dataframes
        # or a mapping from strings to dataframes
        df, others = dataframes
        if isinstance(others, DataFrame):
            others = [others]
    else:
        # the case when there are multiple input objects;
        # in this case we assume the whole `dataframes` is a sequence of get_spark DataFrame objects
        df = dataframes[0]
        others = dataframes[1:]

    can_union_by_name = hasattr(df, 'unionByName')
    allow_missing_columns = allow_missing_columns and can_union_by_name

    col_name_set = set(df.columns)
    if isinstance(others, Mapping):
        if not allow_missing_columns:
            for name, other in others.items():
                if col_name_set != set(other.columns):
                    raise ValueError(f"dataframe '{name}' has different columns {other.columns}")
        others = others.values()
    else:
        if isinstance(others, DataFrame):
            others = (others,)
        if not allow_missing_columns:
            for i, other in enumerate(others):
                if col_name_set != set(other.columns):
                    raise ValueError(f"dataframe {i + 1} has different columns {other.columns}")

    try:
        if can_union_by_name:
            for other in others:
                df = df.unionByName(other, allowMissingColumns=allow_missing_columns)
        else:
            for other in others:
                df = df.union(other.select(*col_name_set))
    except Exception as ex:
        if not force_union_tmp_path:
            raise ex
        else:
            df = force_union(
                spark, force_union_tmp_path, df, *others,
                tmp_file_format=force_union_tmp_file_format
            )
    return df


def priority_union(
        key_cols,
        *dataframes: Union[DataFrame, List[DataFrame], Mapping[str, DataFrame]],
        priority_colname=None
) -> Union[DataFrame, None]:
    dataframes = _sanitize_dataframes(dataframes)
    if dataframes is None or isinstance(dataframes, DataFrame):
        return dataframes

    if not priority_colname:
        priority_colname = get_internal_colname('priority')
        keep_priority_col = False
    else:
        keep_priority_col = True

    df_union = one_from_each_group(
        union(
            *(
                df.withColumn(priority_colname, F.lit(i))
                for i, df in enumerate(dataframes)
            )
        ),
        group_cols=key_cols,
        order_cols=priority_colname
    )

    if not keep_priority_col:
        df_union = df_union.drop(priority_colname)

    return df_union


# endregion


# region prev and next
def _prev_and_next(
        df: DataFrame,
        group_cols: List[Union[str, Column]],
        order_cols: List[Union[str, Column]],
        prev_next_col_names: Iterable[str],
        suffix_prev: str,
        suffix_next: str,
        null_next_indicator_col_name: str,
        keep_order_cols: bool,
        next_offset: int,
):
    """
    See `prev_and_next` method.
    """

    for col_name in prev_next_col_names:
        df = df.withColumn(
            col_name + suffix_next,
            F.lead(
                col=F.col(col_name),
                offset=next_offset,
            ).over(Window.partitionBy(*group_cols).orderBy(*order_cols)),
        )
        df = df.withColumnRenamed(col_name, col_name + suffix_prev)

    if keep_order_cols:
        for col_name in order_cols:
            if isinstance(col_name, str):
                df = df.withColumn(
                    col_name + suffix_next,
                    F.lead(
                        col=F.col(col_name),
                        offset=next_offset,
                    ).over(
                        Window.partitionBy(*group_cols).orderBy(*order_cols)
                    ),
                )
            else:
                raise ValueError(
                    f"'order_cols' must be column names in order to keep them; got '{col_name}'"
                )
        for col_name in order_cols:
            df = df.withColumnRenamed(col_name, col_name + suffix_prev)

    if null_next_indicator_col_name:
        df = df.where(F.col(null_next_indicator_col_name + suffix_next).isNotNull())
    return df


def prev_and_next(
        df: DataFrame,
        group_cols: List[Union[str, Column]],
        order_cols: List[Union[str, Column]],
        prev_next_col_names: Iterable[str] = None,
        shared_col_names: Iterable[str] = None,
        suffix_prev: str = '_first',
        suffix_next: str = '_second',
        null_next_indicator_col_name: str = None,
        keep_order_cols: bool = False,
        reverse_order: bool = False,
        next_offset: Union[int, Iterable[int], str] = 1,
        offset_colname: str = None
):
    """
    Groups the dataframe and sort within each group. Within each sorted group,
    extracts row pairs an merge them into a single row
    where the first row leads the the second row by the specified offset.

    Args:
        df: the dataframe.
        group_cols: group key columns.
        order_cols: order key columns.
        prev_next_col_names: name of columns where the pair of rows would have different values;
            two columns will be in the merged row for each of `prev_next_col_names`
            with suffixes `suffix_prev` and `suffix_next`.
        shared_col_names: name of columns where the pair of rows would have the same values.
        suffix_prev: the name suffix for columns specified by `prev_next_col_names`
            for values from the first row.
        suffix_next: the name suffix for columns specified by `prev_next_col_names`
            for values from the second row.
        null_next_indicator_col_name: the column used to indicate
            there is no "next" row in the group and the pair is not valid;
            we remove results where the column of this name is null.
        keep_order_cols: True to include `order_cols` in the returned dataframe;
            similar to `prev_next_col_names`,
            two columns will be in the merged row for each of `order_cols`
            with suffixes `suffix_prev` and `suffix_next`.
        reverse_order: True to sort by the reversed order of `order_cols`.
        next_offset: for each extracted row pair, the "prev" row is leading the "next" row
            within each sorted group; may specify a single offset or multiple offsets;
            in case of multiple offsets, the results will be union into a single dataframe.
        offset_colname: specify the name for a column recording the offset
            between the "prev" row and the "next" row;
            when single offset is specified by `next_offset`, this column is optional;
            when multiple offset numbers are specified in `next_offset`, this column is a must-have,
            and will use the default name 'offset' if not specified.

    Returns: a dataframe with merged rows from extracted prev/next row pairs.

    """

    (
        group_cols, order_cols, group_colnames, order_colnames
    ) = _solve_group_and_order_cols(group_cols, order_cols, reverse_order, return_colnames=True)

    if prev_next_col_names is None:
        prev_next_col_names = [
            colname
            for colname in df.columns
            if (
                    (colname not in group_colnames)
                    and (colname not in order_colnames)  # noqa: E126
                    and ((shared_col_names is None) or (colname not in shared_col_names))  # noqa: E126
            )
        ]

    if isinstance(next_offset, int):
        df_out = _prev_and_next(
            df=df,
            group_cols=group_cols,
            order_cols=order_cols,
            prev_next_col_names=prev_next_col_names,
            suffix_prev=suffix_prev,
            suffix_next=suffix_next,
            null_next_indicator_col_name=null_next_indicator_col_name,
            keep_order_cols=keep_order_cols,
            next_offset=next_offset
        )
        if not offset_colname:
            return df_out
        else:
            return df_out.withColumn(offset_colname, F.lit(next_offset))

    else:
        if not offset_colname:
            offset_colname = 'offset'
        return union(
            [
                _prev_and_next(
                    df=df,
                    group_cols=group_cols,
                    order_cols=order_cols,
                    prev_next_col_names=prev_next_col_names,
                    suffix_prev=suffix_prev,
                    suffix_next=suffix_next,
                    null_next_indicator_col_name=null_next_indicator_col_name,
                    keep_order_cols=keep_order_cols,
                    next_offset=_next_offset
                ).withColumn(offset_colname, F.lit(_next_offset))
                for _next_offset in next_offset
            ]
        )


def prev_and_next_by_join(
        df: DataFrame,
        group_cols: List[Union[str, Column]],
        index_col: Union[str, Column],
        prev_next_col_names: Iterable[str] = None,
        shared_col_names: Iterable[str] = None,
        suffix_prev: str = '_first',
        suffix_next: str = '_second',
        keep_index_col: bool = True,
        next_offset: Optional[Union[int, str, Callable[[str, str], Column]]] = 1,
        offset_colname: str = 'offset',
):
    group_cols = make_list_(group_cols)
    group_colnames = [colname for colname in group_cols if isinstance(colname, str)]
    is_index_col_str = isinstance(index_col, str)

    if prev_next_col_names is None:
        prev_next_col_names = [
            col_name
            for col_name in df.columns
            if (
                    (col_name not in group_colnames)
                    and ((not is_index_col_str) or col_name != index_col)  # noqa: E126
                    and ((shared_col_names is None) or (col_name not in shared_col_names))  # noqa: E126
            )
        ]

    if is_index_col_str:
        index_colname_prev = index_col + suffix_prev
        index_colname_next = index_col + suffix_next
        df1 = df.withColumnRenamed(index_col, index_colname_prev)
        df2 = df.withColumnRenamed(index_col, index_colname_next)
    else:
        index_colname_prev = get_internal_colname('index_prev')
        index_colname_next = get_internal_colname('index_next')
        df1 = df.withColumn(index_colname_prev, index_col)
        df2 = df1.withColumnRenamed(index_colname_prev, index_colname_next)

    df1 = rename(
        df1,
        {col_name: (col_name + suffix_prev) for col_name in prev_next_col_names}
    )
    df2 = rename(
        df2.drop(
            *(
                colname for colname in df.columns
                if colname not in prev_next_col_names
                   and colname not in group_cols
            )
        ),
        {col_name: (col_name + suffix_next) for col_name in prev_next_col_names}
    )

    df_joint = df1.join(df2, group_cols)

    if offset_colname is not None:
        df_joint = df_joint.withColumn(
            offset_colname, F.col(index_colname_next) - F.col(index_colname_prev)
        )

    if next_offset is None:
        next_offset = 1

    if next_offset == '*':
        df_joint = df_joint.where(
            F.col(index_colname_next) > F.col(index_colname_prev)
        )
    elif isinstance(next_offset, int) and next_offset > 0:
        df_joint = df_joint.where(
            F.col(index_colname_next) - F.col(index_colname_prev) == next_offset
        )
    elif callable(next_offset):
        df_joint = df_joint.where(
            next_offset(F.col(index_colname_prev), F.col(index_colname_next))
        )
    else:
        raise ValueError(
            "'next_offset' should be a positive integer of the index offset between the row pair, "
            "or the string '*' for all pairs where the first row is of smaller index than the second row, "
            "or a callable function that compares the row indexes; "
            f"got {next_offset}"
        )

    if not keep_index_col:
        df_joint = df_joint.drop(index_colname_prev, index_colname_next)
    return df_joint

# endregion
