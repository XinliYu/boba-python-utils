import pprint
from functools import partial
from itertools import chain, zip_longest
from typing import Callable, Iterable, Dict, Union, Optional, List, Mapping, Tuple, Sequence

from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame
from pyspark.sql.column import Column

import boba_python_utils.general_utils.plot_utils as plotu
import boba_python_utils.spark_utils.spark_functions as F
from boba_python_utils.general_utils.argex import apply_arg_combo
from boba_python_utils.common_utils.array_helper import all_equal
from boba_python_utils.common_utils.iter_helper import zip__, iter__, iter_
from boba_python_utils.common_utils.map_helper import promote_keys
from boba_python_utils.general_utils.console_util import hprint_section_title, hprint_message, hprint_section_separator
from boba_python_utils.general_utils.general import (
    make_list_,
)
from boba_python_utils.general_utils.plot_utils import BucketCoding
from boba_python_utils.spark_utils.spark_functions import (
    solve_column_alias,
)
from boba_python_utils.spark_utils import VERBOSE
from boba_python_utils.spark_utils.aggregation import one_from_each_group, add_group_cols
from boba_python_utils.spark_utils.common import (
    get_col_and_name, get_internal_colname, is_single_row_dataframe, exclude_columns_of_conflict_names, has_col
)
from boba_python_utils.spark_utils.data_transform import rename, with_columns, promote_columns
from boba_python_utils.spark_utils.typing import NameOrColumn, NamesOrColumns
from boba_python_utils.string_utils.prefix_suffix import solve_name_conflict


def show(
        df: DataFrame,
        select: Iterable[NameOrColumn] = None,
        alias: Iterable[str] = None,
        orderby: Iterable[NameOrColumn] = None,
        orderby_reversed: bool = False,
        distinct: bool = True,
        title: str = None,
        show_limit: int = 20,
        remove_header: bool = False,
        show_truncate: bool = False
):
    if title:
        hprint_section_title(title)
    if select is not None:
        df = df.select(
            *(
                F.col_(_select).alias(_alias) if _alias else F.col_(_select)
                for _select, _alias
                in zip__(select, alias)
            )
        )
    if distinct:
        df = df.distinct()
    if orderby:
        df = df.orderBy(
            *iter__(orderby, atom_types=(str, Column)),
            ascending=not orderby_reversed
        )
    if remove_header:
        df = rename(
            df,
            dict(zip(df.columns, [f'c{i}' for i in range(1, len(df.columns) + 1)]))
        )

    if not show_limit:
        _show_limit = df.count()
        hprint_message(f'showing all {_show_limit} results')
    else:
        _show_limit = show_limit
        hprint_message(f'showing top {_show_limit} results')
    df.show(_show_limit, show_truncate)

    if title:
        hprint_section_separator()


def get_delta_cols(
        df: DataFrame,
        metric_col1: Union[str, Column],
        metric_col2: Union[str, Column],
        weight_col: Union[str, Column] = None,
        disp_metric_name1: str = None,
        disp_metric_name2: str = None,
        disp_delta_name: str = None,
        includes_count_col: bool = True,
        round: Union[bool, int] = 4
):
    metric_col1, disp_name1 = get_col_and_name(df, metric_col1)
    metric_col2, disp_name2 = get_col_and_name(df, metric_col2)
    disp_name1 = disp_metric_name1 or disp_name1
    disp_name2 = disp_metric_name2 or disp_name2

    delta_cols = [
        F.count('*').alias('count'),
        F.round_(F.ratio_true(metric_col1.isNotNull()), round).alias(f'coverage_{disp_name1}'),
        F.round_(F.ratio_true(metric_col2.isNotNull()), round).alias(f'coverage_{disp_name2}'),
        F.round_(F.ratio_true(metric_col1.isNotNull() & metric_col2.isNotNull()), round).alias(
            f'coverage_both{"" if not disp_delta_name else "_for_" + disp_delta_name}'
        ),
        F.round_(F.avg(metric_col1), round).alias(disp_name1),
        F.round_(F.avg(metric_col2), round).alias(disp_name2),
        F.round_(F.avg(metric_col1 - metric_col2), round).alias(disp_delta_name or 'delta'),
        F.round_(F.avg(metric_col1) - F.avg(metric_col2), round).alias(
            f'{disp_delta_name or "delta"} (separate average)'
        )
    ]

    if not includes_count_col:
        delta_cols = delta_cols[1:]

    if weight_col is not None:
        weight_col, weight_colname = get_col_and_name(df, weight_col)
        delta_cols.append(
            (
                    F.sum(F.col(weight_col) * (F.col(metric_col1) - F.col(metric_col2))) / F.sum(weight_col)
            ).alias(f'weighted_{disp_delta_name or "delta"}')
        )

    return delta_cols


def _solve_order_by(df, order_by, default_order_by=None):
    if (
            order_by is not None and
            (
                    (isinstance(order_by, Column)) or
                    bool(order_by)
            )
    ):
        if isinstance(order_by, str) and order_by == '#rand':
            df = df.orderBy(F.rand())
        else:
            df = df.orderBy(*iter__(order_by, atom_types=(str, Column)))
    elif default_order_by is not None:
        df = df.orderBy(*iter__(default_order_by, atom_types=(str, Column)))
    return df


def show_delta(
        df: DataFrame,
        col1: Union[str, Column],
        col2: Union[str, Column],
        weight_col: Union[str, Column] = None,
        group_cols=None,
        show_limit=20,
        title=None,
        filter=None,
        order_by=None
):
    if title:
        hprint_section_title(title)

    delta_cols = get_delta_cols(
        df=df,
        metric_col1=col1,
        metric_col2=col2,
        weight_col=weight_col
    )

    if group_cols is None:
        df = df.select(delta_cols)
    else:
        df = df.groupBy(*make_list_(group_cols)).agg(*delta_cols).orderBy(F.col('count').desc())

    if filter is not None:
        from boba_python_utils.spark_utils.join_and_filter import where
        df = where(df, filter)

    df = _solve_order_by(df, order_by, F.col('count') * F.col('delta'))
    df.show(show_limit or df.count(), False)
    return df


def show_with_ids(
        df: DataFrame,
        disp_cols,
        id_field_name,
        distinct=True,
        top=50,
        order_by=None
):
    if distinct:
        df = one_from_each_group(
            df,
            group_cols=[_col for _col in disp_cols if isinstance(_col, str)]
        )
    df = _solve_order_by(df, order_by)
    if top is None:
        top = 50
    df = df.limit(top).cache()
    df.count()

    df.select(disp_cols).show(top, False)
    for row in df.select(id_field_name).collect():
        print(row[id_field_name])
    df.unpersist()


def show_count(df: DataFrame, title: str) -> int:
    """
    Displays the count of a Spark DataFrame.

    Args:
        df: The input Spark DataFrame for which the count needs to be displayed.
        title: A string that serves as the title for the displayed count.

    Returns:
        The count of the input DataFrame.

    Examples:
        >>> from pyspark.sql import Row
        >>> data = [Row(Name="Alice", Age=34), Row(Name="Bob", Age=45), Row(Name="Cathy", Age=29)]
        >>> df = spark.createDataFrame(data)
        >>> show_count(df, "Example DataFrame count")
        Example DataFrame count: 3
    """
    cnt = df.count()
    try:
        from boba_python_utils.general_utils.console_util import (
            hprint_message,
        )

        hprint_message(title, cnt)
    except:  # noqa: E722
        print('{}: {}'.format(title, cnt))
    return cnt


def show_count_ratio(df1, df2, title1, title2, ratio_title='ratio'):
    """
    Displays the counts of two dataframes and their count ratio.
    """
    cnt1, cnt2 = (
        (df1 if isinstance(df1, (int, float)) else df1.count()),
        (df2 if isinstance(df2, (int, float)) else df2.count())
    )
    try:
        from utix.consolex import hprint_pairs

        hprint_pairs((title1, cnt1), (title2, cnt2), (ratio_title, cnt1 / float(cnt2)))
    except:  # noqa: E722
        print(f'{title1}: {cnt1}; {title2}: {cnt2}; {ratio_title}: {cnt1 / float(cnt2)}')
    return cnt1, cnt2


def diff(
        df1: DataFrame,
        df2: DataFrame,
        keys: List[Union[str, Column]],
        distinct_overlap=False,
        df1_extra_cache_name=None,
        df2_extra_cache_name=None,
        df_overlap_cache_name: str = None,
        overlap_suffix1='_1',
        overlap_suffix2='_2',
):
    """
    Finds the difference between two data frames and returns three dataframes for
        1) what is extra in `df1`, as `df1_extra`;
        2) what is extra in `df2` as `df2_extra`; and
        3) what is their overlap, as `df_overlap`;
    In the overlap, the non-key column names might be automatically suffixed
        to avoid column name conflict.

    :param df1: the first dataframe.
    :param df2: the second dataframe.
    :param keys: the columns used to determine the identity of rows; the rows
            from two dataframes with the same values on these columns
            will be considered as the overlap.
    :param distinct_overlap: set this to `True` to randomly choose one
            from those with the same `keys` when calculating the overlap;
        we use the join operation to find the overlap;
                therefore if the `keys` are not unique in the two dataframes,
                the overlap might consider each row multiple times.
    :param df1_extra_cache_name: can specify any meaningful name;
        if specified, the first returned dataframe `df1_extra` will be cached.
    :param df2_extra_cache_name: can specify any meaningful name;
        if specified, the second returned dataframe `df2_extra` will be cached.
    :param df_overlap_cache_name: can specify any meaningful name;
        if specified, the third returned dataframe `df_overlap` will be cached.
    """
    from boba_python_utils.spark_utils.data_transform import rename_by_adding_suffix
    from boba_python_utils.spark_utils.data_loading import cache__
    from boba_python_utils.spark_utils.aggregation import one_from_each_group
    if isinstance(keys, str):
        keys = [keys]
    elif isinstance(keys, tuple):
        keys = list(keys)

    df1_extra = df1.join(df2, keys, how='leftanti')
    df2_extra = df2.join(df1, keys, how='leftanti')

    if distinct_overlap == 'df1':
        df1 = one_from_each_group(df1, group_cols=keys)
    elif distinct_overlap == 'df2':
        df2 = one_from_each_group(df2, group_cols=keys)
    elif distinct_overlap:
        # choose one from those with the same keys if `distinct_overlap` is set `True`,
        #     so then each key is considered only once in the overlap
        df1 = one_from_each_group(df1, group_cols=keys)
        df2 = one_from_each_group(df2, group_cols=keys)

    df1 = rename_by_adding_suffix(df1, suffix=overlap_suffix1, excluded_col_names=keys)
    df2 = rename_by_adding_suffix(df2, suffix=overlap_suffix2, excluded_col_names=keys)
    df_overlap = df1.join(df2, keys, how='inner')

    if df1_extra_cache_name:
        if df1_extra_cache_name is True:
            df1_extra_cache_name = 'df1_extra'
        df1_extra = cache__(df1_extra, df1_extra_cache_name)
    if df2_extra_cache_name:
        if df2_extra_cache_name is True:
            df2_extra_cache_name = 'df2_extra'
        df2_extra = cache__(df2_extra, df2_extra_cache_name)
    if df_overlap_cache_name:
        if df_overlap_cache_name is True:
            df_overlap_cache_name = 'df_overlap'
        df_overlap = cache__(df_overlap, df_overlap_cache_name)

    return df1_extra, df2_extra, df_overlap


def get_counts(
        dfs: Union[DataFrame, Iterable[DataFrame]],
        group_cols: Optional[List[Column]],
        count_colname: str = 'count',
        min_count: int = -1,
        ratio_colname: Optional[str] = 'ratio',
        min_ratio: float = -1.0,
        ratio_digits: int = 4,
        aggregate_existing_counts: bool = False,
        ignore_null: bool = False,
        colname_suffix: bool = True,
        total: Optional[Union[int, float]] = None,
        extra_agg_cols: List[Column] = None,
        extra_cols: Mapping[str, Column] = None,
        promote_cols: Union[str, List[str], Tuple[str]] = None,
        verbose: bool = VERBOSE
) -> DataFrame:
    """
    Group each dataframes by the specified `group_cols`, gets the counts for each group as well the
    ratio of each group against the `total`, and join these statistics for all the dataframes.

    Args:
        dfs: the dataframes on which to perform the grouping and counting.
        group_cols: specify the group columns.
        count_colname: counts of each group will be saved in a new column of this name,
            and a possible suffix will be automatically appended if `colname_suffix` is set True.
        min_count: excludes groups with a count lower than this value; use `-1` to disable this.
        ratio_colname: the ratio of each group against the `total`
            will be saved in a new column of this name,
            and a possible suffix will be automatically appended if `colname_suffix` is set True.
        min_ratio: excludes groups with a ratio lower than this value; use `-1` to disable this.
        ratio_digits: round the ratio to this specified number of digit.
        ignore_null: True to exclude rows whose values of columns `group_cols` contains 'null'
            from the counting and aggregation.
        colname_suffix: enable this to resolve column name conflict; can be
            1) `None` or `False` to disable column name suffix;
            2) `True` to add an automatically increased integer as the column name suffix;
            3) a list of strings to customize a suffix for each dataframe.
        total: the ratio will be calculated against this `total` if specified;
            `None` to indicate using the size of each dataframe.
        extra_agg_cols: additional aggregation columns
            to be included in the returned statistics dataframe.

    Returns: a dataframe consisting of columns of `group_cols`,
        a count column if `count_colname` is not empty,
        a ratio column if `ratio_colname` is not empty,
        and additional aggregation columns from `extra_agg_cols`.

    """
    if group_cols is None:
        is_getting_total_stats = True
        total_disp_colname = 'total'
        total_colname = solve_name_conflict(
            total_disp_colname, sum((df.columns for df in dfs), []), always_with_suffix=False
        )
        dfs = [df.withColumn(total_colname, F.lit('')) for df in dfs]
        group_cols = [total_colname]
    else:
        is_getting_total_stats = False
        group_cols = make_list_(group_cols)

    def _get_cnts(df, total, extra_agg_cols):
        if aggregate_existing_counts:
            if count_colname in df.columns:
                agg_cols = [F.sum(count_colname).alias(count_colname)]
            else:
                raise ValueError(
                    f"'aggregate_existing_counts' is set True, "
                    f"but '{count_colname}' cannot be found "
                    f"in the dataframe's columns {df.columns}"
                )
        else:
            agg_cols = [F.count('*').alias(count_colname)]
        if extra_agg_cols is not None:
            agg_cols = agg_cols + extra_agg_cols

        if verbose:
            hprint_message(
                'group_cols', group_cols,
                'agg_cols', agg_cols,
                'count_colname', count_colname
            )

        df_cnts = df.groupBy(*group_cols).agg(*agg_cols).orderBy(F.col(count_colname).desc())

        # region excluding statistics if their `group_cols` contain 'null'
        if ignore_null:
            is_not_null_cond = df_cnts[0].isNotNull()
            for i in range(1, len(group_cols)):
                is_not_null_cond = is_not_null_cond & df_cnts[i].isNotNull()
            df_cnts = df_cnts.where(is_not_null_cond)
        # endregion

        # region computes `total` before any filtering
        if ratio_colname is not None and total is None:
            # `total` is only needed when we compute the ratio,
            # so `ratio_colname is not None` checks if the ratio is enabled;
            # `total` can also be specified rather than inferred from  the dataframe,
            # so `total is None` checks if it is specified.
            total = (
                df_cnts.select(F.sum(count_colname)).head()[0]
                if ignore_null
                else df.count()
            )
        # endregion

        # region filter statistics by `min_count`
        if min_count >= 0:
            df_cnts = df_cnts.where(F.col(count_colname) > min_count)
        # endregion

        # region computes ratio and filter by `min_ratio`
        if ratio_colname is not None:
            if ratio_digits >= 0:
                df_cnts = df_cnts.withColumn(
                    ratio_colname, F.round(F.col(count_colname) / total, ratio_digits)
                )
            else:
                df_cnts = df_cnts.withColumn(ratio_colname, F.col(count_colname) / total)
            if min_ratio >= 0:
                df_cnts = df_cnts.where(F.col(ratio_colname) > min_ratio)
        # endregion

        return df_cnts

    if not isinstance(dfs, (list, tuple)):
        dfs, group_cols = add_group_cols(dfs, group_cols)
        df_out = _get_cnts(dfs, total, extra_agg_cols)
    else:
        from boba_python_utils.spark_utils.join_and_filter import (
            join_multiple_on_columns
        )
        dfs, _group_cols = zip(*(add_group_cols(df, group_cols) for df in dfs))
        if all_equal(_group_cols):
            group_cols = _group_cols[0]
        else:
            raise ValueError(
                f"group columns '{group_cols}' "
                f"cannot be resolved consistently across all dataframes"
            )
        if isinstance(extra_agg_cols[0], (list, tuple)):
            if isinstance(total, (tuple, list)):
                df_out = join_multiple_on_columns(
                    [
                        _get_cnts(df, _total, _extra_agg_cols)
                        for df, _total, _extra_agg_cols in zip_longest(dfs, total, extra_agg_cols)
                    ],
                    join_colnames=group_cols,
                    colname_suffix=colname_suffix,
                    how='outer',
                    broadcast_join=True
                )
            else:
                df_out = join_multiple_on_columns(
                    [
                        _get_cnts(df, total, _extra_agg_cols)
                        for df, _extra_agg_cols in zip_longest(dfs, extra_agg_cols)
                    ],
                    join_colnames=group_cols,
                    colname_suffix=colname_suffix,
                    how='outer',
                    broadcast_join=True
                )

        else:
            if isinstance(total, (tuple, list)):
                df_out = join_multiple_on_columns(
                    [_get_cnts(df, _total, extra_agg_cols) for df, _total in zip(dfs, total)],
                    join_colnames=group_cols,
                    colname_suffix=colname_suffix,
                    how='outer',
                    broadcast_join=True
                )
            else:
                df_out = join_multiple_on_columns(
                    [_get_cnts(df, total, extra_agg_cols) for df in dfs],
                    join_colnames=group_cols,
                    colname_suffix=colname_suffix,
                    how='outer',
                    broadcast_join=True
                )
    if extra_cols:
        df_out = with_columns(df_out, extra_cols)

    if promote_cols:
        df_out = promote_columns(df_out, df_out.columns[0], *iter__(promote_cols))

    if is_getting_total_stats and total_disp_colname != total_colname:
        total_disp_colname = solve_name_conflict(total_disp_colname, df_out.columns)
        df_out = df_out.withColumnRenamed(total_colname, total_disp_colname)

    return df_out


def show_counts(
        dfs,
        group_cols,
        count_colname: str = 'count',
        min_count: int = -1,
        ratio_colname: Optional[str] = 'ratio',
        min_ratio: float = -1.0,
        ratio_digits: int = 4,
        aggregate_existing_counts: bool = False,
        ignore_null: bool = False,
        show_limit: Optional[int] = 50,
        show_truncate: bool = False,
        show_total: bool = False,
        extra_agg_cols: List[Column] = None,
        extra_cols: Mapping[str, Column] = None,
        orderby: Column = None,
        colname_suffix: Union[bool, Iterable[str]] = None,
        promoted_disp_cols=None,
        table_disp_cols=None,
        dict_disp_cols=None,
        title: str = None,
        return_counts_dataframe: bool = False
) -> DataFrame:
    """
    Displays the results of `get_counts`. See `get_counts` for more details.
    `show_limit` and `truncate` are parameters for the display.
    """

    if table_disp_cols == '*':
        table_disp_cols = None

    df_counts = get_counts(
        dfs,
        group_cols=group_cols,
        count_colname=count_colname,
        min_count=min_count,
        ratio_colname=ratio_colname,
        ratio_digits=ratio_digits,
        aggregate_existing_counts=aggregate_existing_counts,
        ignore_null=ignore_null,
        min_ratio=min_ratio,
        colname_suffix=colname_suffix,
        extra_agg_cols=extra_agg_cols,
        extra_cols=extra_cols,
        promote_cols=promoted_disp_cols
    ).cache()

    def _get_disp_cols(_df, disp_cols):
        return (
            _df.columns[0],
            *(
                filter(
                    lambda colname: colname in _df.columns[1:],
                    iter__(disp_cols)
                )
            )
        )

    if dict_disp_cols:
        if title:
            hprint_section_title(title)

        df_counts_disp = (
            df_counts.select(*_get_disp_cols(df_counts, dict_disp_cols))
            if dict_disp_cols != '*'
            else df_counts
        )

        if orderby is not None:
            df_counts_disp = df_counts_disp.orderBy(orderby)

        for row in df_counts_disp.head(show_limit):
            print(
                promote_keys(
                    row.asDict(),
                    [df_counts.columns[0], *iter__(promoted_disp_cols)]
                )
            )

    df_counts_disp = (
        df_counts.select(*_get_disp_cols(df_counts, table_disp_cols))
        if table_disp_cols
        else df_counts
    )

    if orderby is not None:
        df_counts_disp = df_counts_disp.orderBy(orderby)

    show(
        df_counts_disp,
        show_limit=show_limit,
        show_truncate=show_truncate,
        title=(
            f'{title} Table (main statistics)'
            if table_disp_cols and title
            else title
        ),
        distinct=False
    )

    if show_total and not is_single_row_dataframe(df_counts):
        df_counts_total = get_counts(
            dfs,
            group_cols=None,
            count_colname=count_colname,
            min_count=min_count,
            ratio_colname=ratio_colname,
            ratio_digits=ratio_digits,
            ignore_null=ignore_null,
            min_ratio=min_ratio,
            colname_suffix=colname_suffix,
            extra_agg_cols=extra_agg_cols,
            extra_cols=extra_cols,
            promote_cols=promoted_disp_cols
        ).cache()

        if title:
            hprint_section_title(f'{title} (overall)')

        if dict_disp_cols:
            disp_dict = (
                df_counts_total.select(*_get_disp_cols(df_counts_total, dict_disp_cols))
                if dict_disp_cols != '*'
                else df_counts_total
            ).head().asDict()
            del disp_dict[df_counts_total.columns[0]]
            print(
                promote_keys(
                    disp_dict,
                    [df_counts.columns[0], *iter__(promoted_disp_cols)]
                )
            )

        (
            df_counts_total.select(*_get_disp_cols(df_counts_total, table_disp_cols))
            if table_disp_cols
            else df_counts_total
        ).show(1, truncate=show_truncate)

        df_counts = (df_counts, df_counts_total)

    if return_counts_dataframe:
        return df_counts
    else:
        df_counts.unpersist()


def map_col(df, col, mapping: Union[Mapping, Tuple], output_colname=None):
    if isinstance(mapping, Mapping):
        _map = F.create_map([F.lit(x) for x in chain(*mapping.items())])
    elif isinstance(mapping, tuple):
        _map = F.create_map(*mapping)
    if output_colname:
        colname = output_colname
    else:
        col, colname = get_col_and_name(df, col)
    return df.withColumn(colname, _map.getItem(col))


def bucketize(df, col, buckets, bucket_coding: BucketCoding = BucketCoding.INDEX) -> DataFrame:
    if buckets is not None:
        col, colname = get_col_and_name(df, col)
        if isinstance(buckets, list):
            splits = buckets
        else:
            if isinstance(buckets, int):
                left = df.select(F.min(col)).head()[0]
                right = df.select(F.max(col)).head()[0]
                num_bins = buckets
            elif isinstance(buckets, tuple):
                left, right, num_bins = buckets
            else:
                raise ValueError()
            import numpy as np
            splits = np.linspace(left, right, num_bins).tolist()

        x_tmp_colname = f'{colname}_bucketized'
        bucketizer = Bucketizer(splits=splits, inputCol=colname, outputCol=x_tmp_colname)
        df = bucketizer.transform(df).drop(
            colname
        ).withColumnRenamed(
            x_tmp_colname, colname
        )

        if bucket_coding != BucketCoding.INDEX:
            if bucket_coding == BucketCoding.LeftBoundary:
                index_to_code_map = {i: splits[i] for i in range(len(splits) - 1)}
            elif bucket_coding == BucketCoding.RightBoundary:
                index_to_code_map = {i: splits[i] for i in range(1, len(splits))}
            elif bucket_coding == BucketCoding.Mean:
                index_to_code_map = {
                    i: (splits[i] + splits[i - 1]) / 2
                    for i in range(1, len(splits))
                }
            else:
                raise ValueError()
            df = map_col(df, col, mapping=index_to_code_map, output_colname=colname)
    return df


def _data_proc_method_for_plot2d(df, x_col, y_col, x_colname, y_colname, other_cols):
    x_col = solve_column_alias(x_col, x_colname)
    y_col = solve_column_alias(y_col, y_colname)

    cols = [x_col, y_col, *other_cols] if other_cols else [x_col, y_col]
    df = df.where(F.is_not_null(*cols)).select(*cols)
    x_colname, y_colname = df.columns[:2]
    return df, x_col, y_col, x_colname, y_colname


def plot2d(
        df: DataFrame,
        x_col: Union[str, Column],
        y_col: Union[str, Column],
        plot_method: Union[str, Callable] = 'scatterplot',
        sample_ratio_or_size: Union[float, int] = 200000,
        sample_seed: int = 0,
        x_colname: str = None,
        y_colname: str = None,
        bucketize_x=None,
        bucket_coding_x: BucketCoding = BucketCoding.Mean,
        other_cols=None,
        clear_before_plot=True,
        **kwargs
):
    plotu.plot2d(
        data_obj=df,
        x_col=x_col,
        y_col=y_col,
        data_proc_method=_data_proc_method_for_plot2d,
        plot_method=plot_method,
        sample_ratio_or_size=sample_ratio_or_size,
        sample_seed=sample_seed,
        bucketize_method=bucketize,
        bucketize_x=bucketize_x,
        bucket_coding_x=bucket_coding_x,
        x_colname=x_colname,
        y_colname=y_colname,
        other_cols=other_cols,
        clear_before_plot=clear_before_plot,
        **kwargs
    )


def _eval_cond(
        df: DataFrame,
        cond_gen: Callable,
        compute_cols,
        eval_cols,
        **cond_gen_kwargs
):
    from boba_python_utils.spark_utils.data_transform import with_columns
    cond = cond_gen(**cond_gen_kwargs)
    select_cols = [*(F.lit(v).alias(k) for k, v in cond_gen_kwargs.items()), *compute_cols]
    return (
        with_columns(
            df.where(cond).select(select_cols),
            eval_cols
        ),
        with_columns(
            df.where(~cond).select(select_cols),
            eval_cols
        )
    )


def search_cond(
        df: DataFrame,
        cond_gen: Callable,
        cond_args_to_search: Dict,
        compute_cols: Iterable[Column] = (),
        eval_cols: Dict[str, Column] = None
):
    eval_results, neg_eval_results = zip(*apply_arg_combo(
        method=partial(
            _eval_cond,
            df=df,
            cond_gen=cond_gen,
            compute_cols=compute_cols,
            eval_cols=eval_cols
        ),
        **cond_args_to_search
    ))
    from boba_python_utils.spark_utils.aggregation import union
    df_eval_result = union(*eval_results).cache()
    df_neg_eval_result = union(*neg_eval_results).cache()

    return df_eval_result, df_neg_eval_result


# region Transition Probability
def _agg_trans_prob_with_precomputed_transition_counts(
        df_data: DataFrame,
        transition_source_cols: Iterable[NameOrColumn],
        count_col: NameOrColumn,
        transition_source_count_agg: Column,
        has_transition_count_col: bool,
        transition_count_colname: str,
        output_transition_source_count_colname: str,
        output_transition_prob_colname: str,
        join_with_input: bool
):
    if count_col is None:
        if has_transition_count_col:
            count_col = transition_count_colname
        else:
            raise ValueError(
                "'count_col' or 'transition_count_colname' "
                "must be specified when 'trans_trg_cols' is not specified"
            )

    if not join_with_input:
        raise ValueError(
            "'join_with_input' must be True when 'trans_trg_cols' is not specified"
        )

    count_col = F.col_(count_col)
    return df_data.join(
        df_data.groupBy(*transition_source_cols).agg(
            # summing over `count_col` will get the total counts for the transition source
            (
                F.sum(count_col)
                if transition_source_count_agg is None
                else transition_source_count_agg
            ).alias(output_transition_source_count_colname)
        ),
        list(transition_source_cols),
        how='left'
    ).withColumn(
        output_transition_prob_colname,
        count_col / F.col(output_transition_source_count_colname)
    )


def _agg_trans_prob_without_precomputed_transition_counts(
        df_data: DataFrame,
        transition_source_cols: Iterable[NameOrColumn],
        count_col: NameOrColumn,
        transition_target_cols: Iterable[NameOrColumn],
        transition_source_count_agg: Column,
        transition_count_agg: Column,
        has_transition_count_col: bool,
        transition_count_colname: str,
        output_transition_source_count_colname: str,
        output_transition_prob_colname: str,
        join_with_input: bool
):
    # Two situations,
    # 1 - `transition_count_colname` is specified (`has_transition_count_col` is True)
    # 2 - `transition_count_colname` is not specified (`has_transition_count_col` is False)
    #     then we group by the transition source/target columns and compute the transition counts.

    transition_group_keys = exclude_columns_of_conflict_names(
        # We must deduplicate the group key columns
        # in case `transition_source_cols` have overlap with `transition_target_cols`.
        *transition_source_cols, *transition_target_cols, df=df_data
    )

    if not transition_count_colname:
        raise ValueError(
            "'transition_count_colname' must be specified "
            "when 'transition_target_cols' is specified; "
            "this 'transition_count_colname' can refer to an existing column, "
            "or be the name of a to-be-computed column"
        )

    if has_transition_count_col:
        # `transition_count_colname` is specified
        df_out = df_data.select(*transition_group_keys, transition_count_colname)
    else:
        # `transition_count_colname` is not specified,
        # and we group by the transition source/target columns to compute the transition counts
        if count_col is None:
            raise ValueError(
                "'count_col' must be specified when "
                "'transition_count_colname' is not an existing column"
            )
        df_out = df_data.groupBy(*transition_group_keys).agg(
            F.sum(count_col).alias(transition_count_colname)
            if transition_count_agg is None
            else transition_count_agg.alias(transition_count_colname)
        )

    df_out = df_out.join(
        (
            df_out.groupBy(*transition_source_cols).agg(
                # summing over `transition_count_colname`
                # will get the total counts for the transition source
                F.sum(transition_count_colname).alias(output_transition_source_count_colname)
            )
            if transition_source_count_agg is None
            else df_data.groupBy(*transition_source_cols).agg(
                transition_source_count_agg.alias(output_transition_source_count_colname)
            )
        ),
        list(transition_source_cols),
        how='left'
    ).withColumn(
        output_transition_prob_colname,
        F.col(transition_count_colname) / F.col(output_transition_source_count_colname)
    )

    if join_with_input:
        df_out = df_data.join(df_out, transition_group_keys, how='left')

    return df_out


def agg_trans_prob(
        df_data: DataFrame,
        transition_source_cols: Iterable[NameOrColumn],
        transition_target_cols: Iterable[NameOrColumn] = None,
        count_col: NameOrColumn = None,
        transition_count_agg: Column = None,
        transition_source_count_agg: Column = None,
        transition_count_colname: Optional[str] = 'trans_count',
        output_transition_source_count_colname: str = 'trans_src_count',
        output_transition_prob_colname: str = 'trans_prob',
        keep_transition_count_cols: bool = True,
        join_with_input: bool = False
) -> DataFrame:
    """
    Aggregates transition probability of transitioning from the values of `transition_source_cols`
    to the values of `transition_target_cols`.

    Args:
        df_data: the input dataframe.
        transition_source_cols: specifies the transition source columns.
        transition_target_cols: optionally specifies the transition target columns;
            if this argument is specified, then `count_col` is treated
            as the raw counts, and we first compute the transition count aggregation by
            summing `count_col` or by `transition_count_agg` if the latter is specified;
            if this argument is not specified, then we assume the input dataframe `df_data`
            itself already has transition count aggregation and `count_col` is treated
            as the transition counting column.
        count_col: specifies the counting column;
            1 - treated as the raw counts if `transition_target_cols` is specified;
            2 - treated as the transition counts if `transition_target_cols` is not specified.
        transition_source_count_agg: specifies the transition source counting aggregation;
            the default aggregation is summing over the `count_col` if this parameter
            is not specified; the aggregation group keys are specified by
            `transition_source_cols`, and the result is saved to a column of name
            `trans_source_count_colname`.
        transition_count_agg: specifies the transition counting aggregation; the default
            aggregation is summing over the `count_col` if this parameter is not specified;
            the aggregation group keys consist of both `transition_source_cols` and
            `transition_target_cols`, and the result is saved to a column of name
            `transition_count_colname`, if such a counting column does not already exist;
            this parameter is effective when `transition_target_cols` is specified.
        output_transition_source_count_colname: specifies a column name for the counting column computed from
            summing over the `count_col` grouped by `transition_source_cols`.
        transition_count_colname: specifies the name of an existing column that counts the number of
            transitions; or specifies a column name for the transition counting column computed
            from summing over the `count_col` grouped by both `transition_source_cols`
            and `transition_target_cols`;
            this parameter is effective when `transition_target_cols` is specified.
        output_transition_prob_colname: specifies the name of the column that saves the transition
            probability.
        keep_transition_count_cols: True to keep the transition counting columns corresponding to
            `transition_count_colname` and `trans_source_count_colname` in the returned dataframe;
            otherwise the two columns will be dropped.
        join_with_input: True to join the input dataframe `df_data` with the transition counts
            and probability; must be set True if `transition_target_cols` is not specified.

    Returns: the dataframe with the computed transition probability.

    Example 1:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import DataFrame
        >>> from typing import List
        >>> # Initialize a Spark Session
        >>> spark = SparkSession.builder.getOrCreate()
        >>> # Create a sample dataframe
        >>> data = [
        ...     ('A', 'B', 1),
        ...     ('A', 'C', 2),
        ...     ('A', 'B', 1),
        ...     ('B', 'A', 1),
        ...     ('B', 'C', 2),
        ...     ('C', 'A', 1),
        ...     ('C', 'B', 1),
        ...     ('C', 'B', 1),
        ... ]
        >>> df = spark.createDataFrame(data, ["state_a", "state_b", "count"])

        >>> # Apply `agg_trans_prob` function
        >>> agg_df = agg_trans_prob(
        ...     df_data=df,
        ...     transition_source_cols=['state_a'],
        ...     transition_target_cols=['state_b'],
        ...     count_col='count'
        ... )

        # `agg_df` is now a DataFrame with the following data:
        # (only showing the relevant columns for brevity)

        # +-------+-------+-----------+----------------+
        # |state_a|state_b|trans_count|trans_prob      |
        # +-------+-------+-----------+----------------+
        # |   A   |   B   |     2     |     0.5        |
        # |   A   |   C   |     2     |     0.5        |
        # |   B   |   A   |     1     |     0.333      |
        # |   B   |   C   |     2     |     0.667      |
        # |   C   |   A   |     1     |     0.333      |
        # |   C   |   B   |     2     |     0.667      |
        # +-------+-------+-----------+----------------+

    Example 2:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import DataFrame
        >>> from typing import List
        >>> # Initialize a Spark Session
        >>> spark = SparkSession.builder.getOrCreate()
        >>> # Create a sample dataframe
        >>> data = [
        ...     ('A', 1, 3),
        ...     ('B', 2, 2),
        ...     ('C', 3, 5),
        ...     ('A', 4, 7),
        ...     ('B', 5, 3),
        ...     ('C', 6, 4),
        ... ]
        >>> df = spark.createDataFrame(data, ["state", "count_id", "trans_count"])
        >>> # Apply `agg_trans_prob` function
        >>> agg_df = agg_trans_prob(
        ...     df_data=df,
        ...     transition_source_cols=['state'],
        ...     count_col='trans_count',
        ...     join_with_input=True
        ... )

        In this example, the dataframe `df` has columns `state`, `count_id` and `trans_count`, representing
        the state, count id and the counts of transitions from the state respectively. The function `agg_trans_prob`
        is used to aggregate the transition probability from 'state'. The resulting dataframe `agg_df` contains
        the calculated transition probabilities.

        # `agg_df` is now a DataFrame with the following data:
        # (only showing the relevant columns for brevity)

        # +-------+--------+--------------+----------------+
        # | state |count_id|trans_count   |trans_prob      |
        # +-------+--------+--------------+----------------+
        # |   A   |   1    |     3        |     0.3        |
        # |   A   |   4    |     7        |     0.7        |
        # |   B   |   2    |     2        |     0.4        |
        # |   B   |   5    |     3        |     0.6        |
        # |   C   |   3    |     5        |     0.56       |
        # |   C   |   6    |     4        |     0.44       |
        # +-------+--------+--------------+----------------+
    """
    if transition_count_colname is None:
        has_transition_count_col = False
        transition_count_colname = get_internal_colname('trans_count')
    else:
        has_transition_count_col = has_col(df_data, transition_count_colname)
    if not keep_transition_count_cols:
        output_transition_source_count_colname = get_internal_colname(output_transition_source_count_colname)
        if not has_transition_count_col:
            transition_count_colname = get_internal_colname(transition_count_colname)

    if transition_target_cols is None:
        # `transition_target_cols` is not specified,
        # then we assume `df_data` itself already has computed certain transition counts,
        # stored in either `count_col` or `transition_count_colname`
        df_out = _agg_trans_prob_with_precomputed_transition_counts(
            df_data=df_data,
            transition_source_cols=transition_source_cols,
            count_col=count_col,
            transition_source_count_agg=transition_source_count_agg,
            has_transition_count_col=has_transition_count_col,
            transition_count_colname=transition_count_colname,
            output_transition_source_count_colname=output_transition_source_count_colname,
            output_transition_prob_colname=output_transition_prob_colname,
            join_with_input=join_with_input
        )
    else:
        # `transition_target_cols` is specified
        df_out = _agg_trans_prob_without_precomputed_transition_counts(
            df_data=df_data,
            transition_source_cols=transition_source_cols,
            count_col=count_col,
            transition_target_cols=transition_target_cols,
            transition_source_count_agg=transition_source_count_agg,
            transition_count_agg=transition_count_agg,
            has_transition_count_col=has_transition_count_col,
            transition_count_colname=transition_count_colname,
            output_transition_source_count_colname=output_transition_source_count_colname,
            output_transition_prob_colname=output_transition_prob_colname,
            join_with_input=join_with_input
        )

    if not keep_transition_count_cols:
        df_out = df_out.drop(output_transition_source_count_colname, transition_count_colname)
    elif not has_transition_count_col:
        df_out = df_out.drop(transition_count_colname)

    # ! the `select` operation is to solve a possible bug in Spark
    return df_out.select(*(F.col(_col).alias(_col) for _col in df_out.columns))


# endregion


def get_comparison_ratios(
        df1: DataFrame,
        df2: DataFrame,
        join_colnames: Union[str, Sequence[str]] = None,
        output_ratio_colname: str = 'ratio',
        join_colnames2: Optional[Union[str, Sequence[str]]] = None,
        stats_cols: NamesOrColumns = None,
        stats_cols2: NamesOrColumns = None,
        stats_agg: Union[str, Callable] = 'count',
        ignore_empty_df1_count: bool = False,
        result_group_cols: NamesOrColumns = None,
        result_group_agg: Union[str, Callable] = 'avg'
):
    _TMP_KEY_NUM_TARGET = get_internal_colname('num_target')
    _TMP_KEY_NUM_ALL = get_internal_colname('num_all')
    from boba_python_utils.spark_utils.join_and_filter import join_on_columns

    if join_colnames2 is None:
        join_colnames2 = join_colnames

    if stats_cols is not None:
        if stats_cols2 is None:
            stats_cols2 \
                = stats_cols
        if isinstance(stats_agg, str):
            stats_agg: Callable = getattr(F, stats_agg)
        df1 = df1.groupBy(
            *iter_(join_colnames)
        ).agg(
            stats_agg(*iter_(stats_cols)).alias(_TMP_KEY_NUM_ALL)
        )
        df2 = df2.groupBy(
            *iter_(join_colnames2)
        ).agg(
            stats_agg(*iter_(stats_cols2)).alias(_TMP_KEY_NUM_TARGET)
        )
    df_joint = join_on_columns(
        df1, df2, join_colnames, join_colnames2, how='left'
    )

    if ignore_empty_df1_count:
        df_joint = df_joint.where(F.col(_TMP_KEY_NUM_TARGET).isNotNull())
    else:
        df_joint = df_joint.fillna(0, [_TMP_KEY_NUM_TARGET])

    df_joint = df_joint.withColumn(
        output_ratio_colname, F.col(_TMP_KEY_NUM_TARGET) / F.col(_TMP_KEY_NUM_ALL)
    ).drop(
        _TMP_KEY_NUM_TARGET, _TMP_KEY_NUM_ALL
    )

    if result_group_cols is not None and result_group_agg is not None:
        if isinstance(result_group_agg, str):
            result_group_agg: Callable = getattr(F, result_group_agg)
        if result_group_cols == '*':
            df_joint = df_joint.select(
                result_group_agg(output_ratio_colname).alias(output_ratio_colname)
            )
        else:
            df_joint = df_joint.groupBy(
                *iter_(result_group_cols)
            ).agg(
                result_group_agg(output_ratio_colname).alias(output_ratio_colname)
            )

    return df_joint
