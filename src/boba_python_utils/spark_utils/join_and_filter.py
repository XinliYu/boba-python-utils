from itertools import product, zip_longest
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Union, Sequence, Set

from pyspark.sql import DataFrame, SparkSession, Column

import boba_python_utils.spark_utils.spark_functions as F
from boba_python_utils.common_utils.function_helper import apply_arg
from boba_python_utils.common_utils.iter_helper import iter_
from boba_python_utils.common_utils.misc import is_negligible as _is_negligible
from boba_python_utils.common_utils.typing_helper import is_none_or_empty_str
from boba_python_utils.common_utils.typing_helper import str_eq
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.general_utils.general import (
    make_list_,
    get_relevant_named_args,
)
from boba_python_utils.spark_utils import VERBOSE, NameOrColumn
from boba_python_utils.spark_utils.aggregation import (
    aggregate,
    top_from_each_group,
    priority_union,
    union
)
from boba_python_utils.spark_utils.array_operations import array_self_cartesian_product
from boba_python_utils.spark_utils.common import (
    INTERNAL_USE_COL_NAME_PREFIX,
    CacheOptions,
    num_shuffle_partitions,
    get_colname,
    get_internal_colname,
    _solve_name_for_exploded_column
)
from boba_python_utils.spark_utils.data_loading import cache__, solve_input
from boba_python_utils.spark_utils.data_transform import fold
from boba_python_utils.spark_utils.data_transform import rename_by_adding_suffix, rename, explode_as_flat_columns
from boba_python_utils.spark_utils.spark_functions import CategorizedCondParseOptions
from boba_python_utils.spark_utils.spark_functions import (
    DIRECTIVE_COND_NOT_APPLIED, DIRECTIVE_COND_NEGATION, DIRECTIVE_NON_COLNAME
)
from boba_python_utils.spark_utils.struct_operations import unfold_struct
from boba_python_utils.string_utils.comparison import solve_compare_option


def _solve_columns_for_join(
        df1: DataFrame,
        df2_or_filter_values: Union[DataFrame, Any, Sequence[Any]],
        colnames1: Sequence[str],
        colnames2: Optional[Sequence[str]]
) -> Tuple[
    List[str],
    List[str],
    bool
]:
    """
    In join operations of two dataframes `df1` and `df2`
    (when `df2_or_filter_values` is a DataFrame),
    it is very often we need to specify two sets of columns
    `col_names1` and `col_names2` as the join keys.

    Sometimes, `col_names1` and/or `col_names2` might not be specified in the arguments
    (left as None); this method infers what the two arguments should be in case
    they are not specified.

    This function also handles a special case when `df2_or_filter_values` is a filter value,
    or a set of filter values (see `_filter_on_columns`).

    """
    if colnames1 is None:
        if colnames2 is None:
            # none of `colnames1` and `colnames2` are specified
            if isinstance(df2_or_filter_values, DataFrame):
                colnames1 = colnames2 = df2_or_filter_values.columns
            else:
                # ! a special case when df2 is a value or a set of values (see `_filter_on_columns`)
                if len(df1.columns) == 1:
                    colnames1 = colnames2 = df1.columns
                else:
                    raise ValueError(
                        "Must provide a single column name in 'colnames1' "
                        "when filtering one dataframe column by values."
                    )
            same_cols = True
        else:
            # `colnames1` is not specified, but `colnames2` is specified
            colnames2 = make_list_(colnames2)
            if len(colnames2) == 1 and len(df1.columns) == 1:
                # if `df1` only has one column,
                # and there is also a single column specified in `colnames2`;
                # in this case we infer `colnames1` contains the sole column of `df1`
                colnames1 = df1.columns
                same_cols = colnames2[0] == colnames1[0]
            else:
                # otherwise, we infer `colnames1` is the same as `colnames2`
                colnames1 = colnames2
                same_cols = True
    elif colnames2 is None:
        # `colnames1` is specified, but `colnames2` is not specified;
        # this is the same situation as the other case
        colnames1 = make_list_(colnames1)
        if (
                isinstance(df2_or_filter_values, DataFrame) and
                len(colnames1) == 1 and
                len(df2_or_filter_values.columns) == 1
        ):
            colnames2 = df2_or_filter_values.columns
            same_cols = colnames2[0] == colnames1[0]
        else:
            colnames2 = colnames1
            same_cols = True
    else:
        # both `colnames1` and `colnames2` are specified;
        colnames1 = make_list_(colnames1)
        colnames2 = make_list_(colnames2)
        if len(colnames1) != len(colnames2):
            raise ValueError(
                f"Must provide the same number of fields for the dataframe "
                f"and the filter dataframe; got {colnames1} and {colnames2}."
            )
        same_cols = set(colnames1) == set(colnames2)
    return colnames1, colnames2, same_cols


def _join_on_columns_preventing_missing_attribute_error(
        df1: DataFrame,
        df2: DataFrame,
        join_colnames1: Iterable[str],
        join_colnames2: Iterable[str],
        broadcast_join: bool,
        *join_args,
        **join_kwargs
) -> DataFrame:
    # ! if `df1` and `df2` are derived from the same underlying dataframe,
    # ! the join operation will crash if the join keys are from the same underlying columns;
    # ! the error is "Resolved attribute(s) X missing from ...",
    # ! even if the column X actually exists_path in `df1.columns` or `df2.columns`.
    # The following code solves this error.
    _join_col_names2 = [
        f"{INTERNAL_USE_COL_NAME_PREFIX}{join_col_name}" for join_col_name in join_colnames2
    ]
    df2 = rename(df2, {name2: _name2 for name2, _name2 in zip(join_colnames2, _join_col_names2)})
    return df1.join(
        (F.broadcast(df2) if broadcast_join else df2),
        (
            F.and_(
                [
                    (df1[name1] == df2[name2])
                    for name1, name2 in zip(join_colnames1, _join_col_names2)
                ]
            )
        ),
        *join_args,
        **join_kwargs,
    ).drop(*_join_col_names2)


def drop_conflict_columns_for_join(
        df1: DataFrame,
        df2: DataFrame,
        join_key_colnames1: Iterable[str],
        join_key_colnames2: Iterable[str],
        how: str,
) -> Tuple[DataFrame, DataFrame]:
    """
    Drops conflict columns from one of the dataframes for a non-anti join operation.
    A conflict column is a column whose name exists in both dataframes and the column is not
    among the join key columns.

    If `how` is 'right', the conflict columns from the first dataframe will be dropped;
    otherwise, the conflict columns from the second dataframe will be dropped.

    Args:
        df1: the first dataframe.
        df2: the second dataframe.
        join_key_colnames1: the join key column names for the first dataframe.
        join_key_colnames2: the join key column names for the second dataframe.
        how: the type of the join operation.

    Returns: the two dataframes after dropping conflict columns.

    Note:
        If `how` is 'right', the conflicting columns from the first dataframe will be dropped;
        otherwise, the conflicting columns from the second dataframe will be dropped.
        For 'anti' join operations, no columns are dropped.

    Example:
        Let's say we have two DataFrames as below:

        df1:
        +----+-----+------+
        | id | name|  age |
        +----+-----+------+
        |  1 | Joe |   25 |
        |  2 | Sam |   30 |
        +----+-----+------+

        df2:
        +----+-----+-------+
        | id | name| height|
        +----+-----+-------+
        |  1 | Tim |   170 |
        |  3 | Max |   180 |
        +----+-----+-------+

        Call the function with 'left' join:

        df1, df2 = drop_conflict_columns_for_join(df1, df2, ['id'], ['id'], 'left')

        Then the modified df2 will be:

        df2:
        +----+-------+
        | id | height|
        +----+-------+
        |  1 |   170 |
        |  3 |   180 |
        +----+-------+

    """
    if "anti" not in how:
        if "right" in how:
            df1_drop_colnames = (
                _colname
                for _colname in df1.columns
                if _colname not in join_key_colnames1 and _colname in df2.columns
            )
            df1 = df1.drop(*df1_drop_colnames)
        else:
            df2_drop_colnames = (
                _colname
                for _colname in df2.columns
                if _colname not in join_key_colnames2 and _colname in df1.columns
            )
            df2 = df2.drop(*df2_drop_colnames)
    return df1, df2


def rename_conflict_columns_for_join(
        df1: DataFrame,
        df2: DataFrame,
        join_key_colnames1: List[str],
        join_key_colnames2: List[str],
        how: str,
        suffix1: Any = 1,
        suffix2: Any = 2,
) -> Tuple[DataFrame, DataFrame]:
    """
    Renames conflicting columns from one of the dataframes for a non-anti join operation.

    A conflicting column is one that exists in both dataframes and is not a join key column.
    Depending on the type of join operation, these columns are renamed in either `df1` or `df2`
    by adding a suffix.

    Args:
        df1: The first DataFrame.
        df2: The second DataFrame.
        join_key_colnames1: The join key column names for the first DataFrame.
        join_key_colnames2: The join key column names for the second DataFrame.
        how: The type of join operation.
        suffix1: The suffix to add to the conflicting columns in the first DataFrame.
        suffix2: The suffix to add to the conflicting columns in the second DataFrame.

    Returns:
        Tuple of the two DataFrames after renaming conflicting columns.

    Note:
        If neither 'suffix1' nor 'suffix2' are specified, a ValueError is raised. For 'anti' join operations,
        no columns are renamed.

    Example:
        Let's say we have two DataFrames as below:

        df1:
        +----+-----+------+
        | id | name|  age |
        +----+-----+------+
        |  1 | Joe |   25 |
        |  2 | Sam |   30 |
        +----+-----+------+

        df2:
        +----+-----+-------+
        | id | name| height|
        +----+-----+-------+
        |  1 | Tim |   170 |
        |  3 | Max |   180 |
        +----+-----+-------+

        Call the function with 'left' join:

        df1, df2 = rename_conflict_columns_for_join(df1, df2, ['id'], ['id'], 'left')

        Then the modified df1 and df2 will be:

        df1:
        +----+--------+------+
        | id | name_1 |  age |
        +----+--------+------+
        |  1 | Joe    |   25 |
        |  2 | Sam    |   30 |
        +----+--------+------+

        df2:
        +----+--------+-------+
        | id | name_2 | height|
        +----+--------+-------+
        |  1 | Tim    |   170 |
        |  3 | Max    |   180 |
        +----+--------+-------+
    """
    has_suffix1 = not is_none_or_empty_str(suffix1)
    has_suffix2 = not is_none_or_empty_str(suffix2)

    if not has_suffix1 and not has_suffix2:
        raise ValueError("one of 'suffix1' and 'suffix2' must be specified")

    if "anti" not in how:
        for _colname in df2.columns:
            if _colname in df1.columns:
                if _colname not in join_key_colnames1 and _colname not in join_key_colnames2:
                    if has_suffix1:
                        df1 = df1.withColumnRenamed(_colname, f"{_colname}_{suffix1}")
                    if has_suffix2:
                        df2 = df2.withColumnRenamed(_colname, f"{_colname}_{suffix2}")
    return df1, df2


def join_on_columns(
        df1: DataFrame,
        df2: DataFrame,
        join_key_colnames1: Sequence[str],
        join_key_colnames2: Optional[Sequence[str]] = None,
        non_join_colname_suffix: Optional[str] = None,
        repartition_before_join: Optional[Union[int, bool]] = None,
        prevent_resolved_attribute_missing_error: bool = False,
        broadcast_join: bool = False,
        avoid_column_name_conflict: Union[bool, str, Tuple[Any, Any]] = False,
        spark: SparkSession = None,
        *join_args,
        **join_kwargs,
):
    """
    Join two dataframes using the specified join key column names.

    This function supports adding suffixes to column names to avoid column name conflict, and
    repartitioning before joining to improve performance. If the two dataframes have derived from
    the same underlying dataframe, this function can prevent a "resolved attribute missing" error.

    Args:
        df1: The first dataframe.
        df2: The second dataframe.
        join_key_colnames1: Column names from the first dataframe as the join key columns.
        join_key_colnames2: Column names from the second dataframe as the join key columns.
        non_join_colname_suffix: Suffix to add to columns names of non-join-key columns.
        repartition_before_join: Repartition both dataframes before join;
            useful for very large join operations.
        prevent_resolved_attribute_missing_error: Prevent "resolved attribute missing" error;
            this error can happen when `df1` and `df2` are derived from the same underlying dataframe.
        broadcast_join: Perform a broadcast join.
        avoid_column_name_conflict: A strategy to avoid column name conflicts.
            It can be one of the following:
            - If False (default), no action is taken.
            - If True or "drop", conflicting columns
                (non-join key columns with the same names from both dataframes) are dropped.
            - If "rename", conflicting columns are renamed by adding suffixes.
            - If a tuple of two elements, the elements are used as suffixes for renaming
                conflicting columns from df1 and df2, respectively.
        spark: Spark session.
        *join_args: Extra positional arguments for the join operation.
        **join_kwargs: Extra named arguments for the join operation.

    Returns:
        The resulting joint dataframe.

    Example:
        Assume we have two dataframes df1 and df2:

        df1:
        +----+-----+
        | id | age |
        +----+-----+
        |  1 | 25  |
        |  2 | 30  |
        +----+-----+

        df2:
        +----+-------+
        | id | height|
        +----+-------+
        |  1 | 170   |
        |  3 | 180   |
        +----+-------+

        We can join these dataframes on the "id" column like this:

        df3 = join_on_columns(df1, df2, ["id"])

        The resulting dataframe df3 would be:

        +----+-----+-------+
        | id | age | height|
        +----+-----+-------+
        |  1 | 25  | 170   |
        +----+-----+-------+
    """
    if df1 is df2 or df2 is None:
        df2 = df1
        is_self_join = True
    else:
        is_self_join = False

    join_key_colnames1, join_key_colnames2, same_join_key_cols = _solve_columns_for_join(
        df1, df2, join_key_colnames1, join_key_colnames2
    )

    how = join_kwargs.get("how", "inner")
    if (not is_self_join) and (
            avoid_column_name_conflict is True or avoid_column_name_conflict == "drop"
    ):
        df1, df2 = drop_conflict_columns_for_join(
            df1=df1,
            df2=df2,
            join_key_colnames1=join_key_colnames1,
            join_key_colnames2=join_key_colnames2,
            how=how,
        )
    elif (
            is_self_join
            or avoid_column_name_conflict == "rename"
            or (
                    isinstance(avoid_column_name_conflict, (list, tuple))
                    and len(avoid_column_name_conflict) == 2
            )
    ):
        if (
                isinstance(avoid_column_name_conflict, (list, tuple))
                and len(avoid_column_name_conflict) == 2
        ):
            suffix1, suffix2 = avoid_column_name_conflict
        else:
            suffix1, suffix2 = 1, 2
        df1, df2 = rename_conflict_columns_for_join(
            df1=df1,
            df2=df2,
            join_key_colnames1=join_key_colnames1,
            join_key_colnames2=join_key_colnames2,
            suffix1=suffix1,
            suffix2=suffix2,
            how=how,
        )
    # if the join keys for `df2` is different from the join keys of `df1`, then
    if prevent_resolved_attribute_missing_error:
        return _join_on_columns_preventing_missing_attribute_error(
            df1=df1,
            df2=df2,
            join_colnames1=join_key_colnames1,
            join_colnames2=join_key_colnames2,
            broadcast_join=broadcast_join,
            *join_args,
            **join_kwargs,
        )
    else:
        if not same_join_key_cols:
            df2 = rename(
                df2, {name2: name1 for name1, name2 in zip(join_key_colnames1, join_key_colnames2)}
            )

        if non_join_colname_suffix not in (None, False):
            # adding suffixes to the non-join-key column names
            if non_join_colname_suffix is True:
                suffix1, suffix2 = 1, 2
            elif isinstance(non_join_colname_suffix, (tuple, list)):
                suffix1, suffix2 = non_join_colname_suffix
            else:
                raise ValueError(
                    "'non_join_col_name_suffix' should be a list/tuple of length 2, "
                    f"or None/True/False; got {non_join_colname_suffix}"
                )

            df1 = rename_by_adding_suffix(
                df1, suffix=suffix1, excluded_col_names=join_key_colnames1
            )
            df2 = rename_by_adding_suffix(
                # from above, the columns `join_key_colnames2` in `df2`
                # has been renamed to `join_key_colnames1`,
                # and therefore here 'excluded_col_names=join_key_colnames1'
                df2, suffix=suffix2, excluded_col_names=join_key_colnames1
            )
        if not repartition_before_join:
            if broadcast_join:
                if how == "leftanti":
                    df2 = F.broadcast(df2.select(*join_key_colnames1).distinct())
                else:
                    df2 = F.broadcast(df2)
            return df1.join(df2, join_key_colnames1, *join_args, **join_kwargs)
        else:
            if repartition_before_join is True:
                if spark is None:
                    repartition_before_join = max(
                        df1.rdd.getNumPartitions(), df2.rdd.getNumPartitions()
                    )
                else:
                    repartition_before_join = num_shuffle_partitions(spark)
            if broadcast_join and how == "inner":
                df1 = df1.join(
                    F.broadcast(df2.select(*join_key_colnames1).distinct()), join_key_colnames1
                )
            return df1.repartition(repartition_before_join, *join_key_colnames1).join(
                df2.repartition(repartition_before_join, *join_key_colnames1),
                join_key_colnames1,
                *join_args,
                **join_kwargs,
            )


def join_on_multiple_colname_groups(
        df1: DataFrame,
        df2: DataFrame,
        colname_groups: Iterable[
            Union[
                Sequence[str],
                Tuple[
                    Sequence[str],
                    Optional[Sequence[str]]
                ]
            ]
        ],
        non_join_colname_suffix: Optional[str] = None,
        repartition_before_join: Optional[Union[int, bool]] = None,
        prevent_resolved_attribute_missing_error: bool = False,
        broadcast_join: bool = False,
        avoid_column_name_conflict: Union[bool, str, Tuple[Any, Any]] = False,
        spark: SparkSession = None,
        cache_option: CacheOptions = CacheOptions.IMMEDIATE,
        *join_args,
        **join_kwargs,
) -> Tuple[DataFrame, DataFrame]:
    """
    This function joins two Spark DataFrames (`df1` and `df2`) based on multiple
    groups of column names. These groups are provided through the `colname_groups` parameter.

    The function operates by performing a join operation for each group of column names
    in the `colname_groups`. For example, if `colname_groups` is `[["id", "value"], ["color", "year"]]`,
    it first joins `df1` and `df2` on the columns "id" and "value". After this join operation,
    it performs another join on the remaining part of `df1` and `df2` based on the columns
    "color" and "year".

    The function returns a tuple that consists of a DataFrame which is the union of all the joined
    DataFrames (where each DataFrame corresponds to a join operation on a group of columns), and
    the remaining part of `df1` after all join operations.

    This function is useful when you want to join two DataFrames based on multiple conditions
    and retrieve all rows that satisfy any of these conditions.

    Args:
        df1: First DataFrame to join.
        df2: Second DataFrame to join.
        colname_groups: Groups of column names for the join operation.
        non_join_colname_suffix: Suffix to add to column names not involved in join. Defaults to None.
        repartition_before_join: If set, repartition the DataFrames before joining. Defaults to None.
        prevent_resolved_attribute_missing_error: If True, prevents "resolved attribute(s) missing" error.
                                                  Defaults to False.
        broadcast_join: If True, performs a broadcast join. Defaults to False.
        avoid_column_name_conflict: Option to handle column name conflicts after join. Defaults to False.
        spark: SparkSession instance.
        cache_option: Caching options for intermediate results. Defaults to CacheOptions.IMMEDIATE.
        *join_args: Additional arguments for `join_on_columns`.
        **join_kwargs: Keyword arguments for `join_on_columns`.

    Returns:
        A tuple of a union of all joint DataFrames and the remaining part of df1.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> df1 = spark.createDataFrame([
        ...     (1, "foo", "red", 2000),
        ...     (2, "bar", "blue", 2001),
        ...     (3, "baz", "green", 2002),
        ...     (4, "qux", "yellow", 2003),
        ... ], ["id", "value", "color", "year"])
        >>> df2 = spark.createDataFrame([
        ...     (2, "foo", "red", 2000),
        ...     (3, "baz", "green", 2002),
        ...     (5, "quux", "pink", 2004),
        ...     (6, "corge", "orange", 2005),
        ... ], ["id", "value", "color", "year"])
        >>> colname_groups = [["id", "value"], ["color", "year"]]
        >>> joint, remain = join_on_multiple_colname_groups(df1, df2, colname_groups, spark=spark)
        >>> joint.show()
        +---+-----+-----+----+
        | id|value|color|year|
        +---+-----+-----+----+
        |  3|  baz|green|2002|
        |  1|  foo|  red|2000|
        +---+-----+-----+----+
        >>> remain.show()
        +---+-----+------+----+
        | id|value| color|year|
        +---+-----+------+----+
        |  2|  bar|  blue|2001|
        |  4|  qux|yellow|2003|
        +---+-----+------+----+


    """
    df1_joints = []
    _df_remain = df1
    for i, colnams in enumerate(colname_groups):
        if isinstance(colnams, Tuple):
            df1_joint, count_df1_joint = cache__(
                join_on_columns(
                    _df_remain,
                    df2,
                    *colnams,
                    *join_args,
                    non_join_colname_suffix=non_join_colname_suffix,
                    repartition_before_join=repartition_before_join,
                    prevent_resolved_attribute_missing_error=prevent_resolved_attribute_missing_error,
                    broadcast_join=broadcast_join,
                    avoid_column_name_conflict=avoid_column_name_conflict,
                    spark=spark,
                    **join_kwargs
                ),
                name=f"df1_joint (joint with '{colnams}')",
                return_count=True,
                cache_option=cache_option
            )
            if count_df1_joint != 0:
                _df_remain = cache__(
                    exclude_by_anti_join_on_columns(
                        _df_remain,
                        df2,
                        *colnams,
                        prevent_resolved_attribute_missing_error=prevent_resolved_attribute_missing_error,
                        broadcast_join=broadcast_join
                    ),
                    name=f"df1 (excluding joint with '{colnams}')",
                    unpersist=(None if i == 0 else _df_remain),
                    cache_option=cache_option
                )
        else:
            df1_joint, count_df1_joint = cache__(
                join_on_columns(
                    _df_remain,
                    df2,
                    colnams,
                    *join_args,
                    non_join_colname_suffix=non_join_colname_suffix,
                    repartition_before_join=repartition_before_join,
                    prevent_resolved_attribute_missing_error=prevent_resolved_attribute_missing_error,
                    broadcast_join=broadcast_join,
                    avoid_column_name_conflict=avoid_column_name_conflict,
                    spark=spark,
                    **join_kwargs
                ),
                name=f"df1_joint (joint with '{colnams}')",
                return_count=True,
                cache_option=cache_option
            )
            if count_df1_joint != 0:
                _df_remain = cache__(
                    exclude_by_anti_join_on_columns(
                        _df_remain,
                        df2,
                        colnams,
                        prevent_resolved_attribute_missing_error=prevent_resolved_attribute_missing_error,
                        broadcast_join=broadcast_join
                    ),
                    name=f"df1 (excluding joint with '{colnams}')",
                    unpersist=(None if i == 0 else _df_remain),
                    cache_option=cache_option
                )
        if count_df1_joint:
            df1_joints.append(df1_joint)

    if df1_joints:
        df1_joint_union = cache__(
            union(df1_joints),
            repartition=True,
            name='df1_joint_union',
            unpersist=df1_joints,
            cache_option=cache_option
        )
    else:
        df1_joint_union = None

    return df1_joint_union, _df_remain


def join_multiple_on_columns(
        dfs: Sequence[DataFrame],
        join_colnames: Union[str, Sequence[str]],
        colname_suffix: str = None,
        *join_args,
        **join_kwargs
) -> DataFrame:
    def _add_col_suffix(df, df_idx):
        if colname_suffix in (None, False):
            return df
        elif colname_suffix is True:
            return rename_by_adding_suffix(
                df,
                suffix=f"_{df_idx}",
                excluded_col_names=join_colnames
            )
        elif isinstance(colname_suffix, (tuple, list)):
            return rename_by_adding_suffix(
                df,
                suffix=colname_suffix[df_idx],
                excluded_col_names=join_colnames
            )
        else:
            raise ValueError(
                "'colname_suffix' should be a list, or tuple, "
                "or one of None, True, False"
            )

    if isinstance(join_colnames, str):
        join_colnames = [join_colnames]

    df = _add_col_suffix(dfs[0], 0)

    if isinstance(join_colnames[0], str):
        for df_idx, df2 in enumerate(dfs[1:]):
            df = join_on_columns(
                df, _add_col_suffix(df2, df_idx + 1), join_colnames, *join_args, **join_kwargs
            )
    else:
        colnames1 = join_colnames[0]
        for df_idx, (df2, colnames2) in enumerate(zip(dfs[1:], join_colnames[1:])):
            df = join_on_columns(
                df,
                _add_col_suffix(df2, df_idx + 1),
                colnames1,
                colnames2,
                *join_args,
                **join_kwargs,
            )
    return df


def _filter_on_columns(
        df: DataFrame,
        df_filter_or_filter_values: Union[DataFrame, Iterable[Any]],
        colnames: Sequence[str] = None,
        filter_colnames: Sequence[str] = None,
        how: str = "inner",
        prevent_resolved_attribute_missing_error: bool = False,
        do_not_filter_if_null_for_inner_join: bool = False,
        broadcast_join: bool = False,
):
    """
    Filters rows in a DataFrame based on column values in a second DataFrame or a list.

    Args:
        df: DataFrame to be filtered.
        df_filter_or_filter_values: DataFrame or list used for filtering.
        colnames: Column names in `df` used for join. If None, defaults to all columns.
        filter_colnames: Column names in `df_filter_or_filter_values` (if it is a DataFrame)
            used for join. If None, defaults to all columns.
        how: Type of join operation. Can be "inner" or "leftanti". Default is "inner".
        prevent_resolved_attribute_missing_error: If True, prevents error that occurs when
            DataFrame is missing an attribute resolved during join operation. Default is False.
        do_not_filter_if_null_for_inner_join: If True, for inner join operation, rows with null
            values in the specified columns will not be filtered out. Default is False.
        broadcast_join: If True, a broadcast join is used. This can be faster for small datasets.
            Default is False.

    Returns:
        A DataFrame which is a result of filtering `df` by `df_filter_or_filter_values`.

    Raises:
        ValueError: If `how` is not "inner" or "leftanti", or if the length of `colnames` is
            not 1 when `df_filter` is a list, tuple or set.

    Examples:
        >>> df = spark.createDataFrame([(1, "foo"), (2, "bar"), (3, "baz")], ["id", "value"])
        >>> df_filter = spark.createDataFrame([(1, "foo")], ["id", "value"])
        >>> df_result = _filter_on_columns(df, df_filter, ["id", "value"], ["id", "value"])
        >>> df_result.show()
        +---+-----+
        | id|value|
        +---+-----+
        |  1|  foo|
        +---+-----+

        >>> df = spark.createDataFrame([(1, "foo"), (2, "bar"), (3, "baz")], ["id", "value"])
        >>> filter_values = [("foo")]
        >>> df_result = _filter_on_columns(df, filter_values, ["value"])
        >>> df_result.show()
        +---+-----+
        | id|value|
        +---+-----+
        |  1|  foo|
        +---+-----+
    """
    if df_filter_or_filter_values is None:
        return df

    if not (how == "inner" or how == "leftanti"):
        raise ValueError("'filter_on_columns' method can only use inner join or leftanti join")

    colnames, filter_colnames, same_cols = _solve_columns_for_join(
        df, df_filter_or_filter_values, colnames, filter_colnames
    )
    if isinstance(df_filter_or_filter_values, DataFrame):
        if set(filter_colnames) != set(df_filter_or_filter_values.columns):
            df_filter_or_filter_values = df_filter_or_filter_values.select(*filter_colnames)

        if not same_cols:
            df_filter_or_filter_values = rename(df_filter_or_filter_values, zip(filter_colnames, colnames))

        df_filter_or_filter_values = df_filter_or_filter_values.distinct()  # ! DO NOT forget to make the filter values unique

        if prevent_resolved_attribute_missing_error:
            df_joint = _join_on_columns_preventing_missing_attribute_error(
                df1=df,
                df2=df_filter_or_filter_values,
                broadcast_join=broadcast_join,
                join_colnames1=colnames,
                join_colnames2=colnames,  # `df_filter` has been renamed above
                how=how,
            )
        else:
            df_joint = df.join(
                (F.broadcast(df_filter_or_filter_values) if broadcast_join else df_filter_or_filter_values), colnames, how=how
            )

        if how == "inner" and do_not_filter_if_null_for_inner_join:
            df_joint = df_joint.union(
                df.where(F.or_(*((F.col(_colname).isNull() for _colname in colnames))))
            )
    else:
        if len(colnames) == 1:
            if isinstance(df_filter_or_filter_values, (List, Tuple, Set)):
                df_filter_or_filter_values = set(df_filter_or_filter_values)
                if how == "inner":
                    df_joint = df.where(F.col(colnames[0]).isin(df_filter_or_filter_values))
                elif how == "leftanti":
                    df_joint = df.where(~(F.col(colnames[0]).isin(df_filter_or_filter_values)))
                else:
                    raise ValueError("'how' argument should be either 'inner' or 'leftanti' "
                                     "when 'df_filter_or_filter_values' is a value or "
                                     "or a list of values.")
            else:
                if how == "inner":
                    df_joint = df.where(F.col(colnames[0]) == df_filter_or_filter_values)
                elif how == "leftanti":
                    df_joint = df.where(F.col(colnames[0]) != df_filter_or_filter_values)
                else:
                    raise ValueError("'how' argument should be either 'inner' or 'leftanti' "
                                     "when 'df_filter_or_filter_values' is a value or "
                                     "or a list of values.")
        else:
            raise ValueError(
                "expect a single column name specified by 'colnames' or 'filter_colnames' "
                "when the filter is a sequence of values"
            )

    return df_joint


def filter_by_inner_join_on_columns(
        df: DataFrame,
        df_filter: DataFrame,
        colnames: List[str] = None,
        filter_colnames: List[str] = None,
        prevent_resolved_attribute_missing_error: bool = False,
        broadcast_join: bool = False,
        do_not_filter_if_null: bool = False,
        verbose=VERBOSE,
        filter_name=None,
):
    """
    Filters the DataFrame `df` using columns of another DataFrame `df_filter`.

    This function removes rows of `df` whose combination of values of certain columns
    (specified by `colnames`) do not exist in the corresponding columns of `df_filter`
    (specified by `filter_colnames`; or the same as `colnames` if not specified).

    This method performs the filtering by inner-join `df` with (distinct) `df_filter`
    on the specified columns.

    Args:
        df: The DataFrame we want to remove some of the rows.
        df_filter: The DataFrame serving as the filter.
        colnames: The columns of `df` to apply the filter; we remove rows in `df`
            whose combination of values of these columns do not exist
            in the corresponding columns of `df_filter`.
        filter_colnames: The columns of `df_filter`; the distinct values of these columns
            serve as the filter for corresponding columns specified by `col_names`;
            if this argument is not specified, we use the same columns as `colnames`.
        prevent_resolved_attribute_missing_error: If True, prevents error that occurs when
            DataFrame is missing an attribute resolved during join operation. Default is False.
        broadcast_join: If True, a broadcast join is used. This can be faster for small datasets.
            Default is False.
        do_not_filter_if_null: If True, rows with null values in the specified columns
            will not be filtered out. Default is False.
        verbose: If True, prints additional information during the operation. Default is False.
        filter_name: A name for the filter operation. Default is None.

    Returns:
        A DataFrame that results from filtering `df` with `df_filter`.

    Examples:
        >>> df = spark.createDataFrame([(1, "foo"), (2, "bar"), (3, "baz")], ["id", "value"])
        >>> df_filter = spark.createDataFrame([(1, "foo")], ["id", "value"])
        >>> df_result = filter_by_inner_join_on_columns(df, df_filter, ["id", "value"], ["id", "value"])
        >>> df_result.show()
        +---+-----+
        | id|value|
        +---+-----+
        |  1|  foo|
        +---+-----+
    """

    if verbose:
        hprint_message(
            f"applies dataframe-join based {filter_name or 'filter'}",
            df_filter,
            "filter key columns1",
            colnames,
            "filter key columns2",
            filter_colnames,
        )

    return _filter_on_columns(
        df,
        df_filter,
        colnames=colnames,
        filter_colnames=filter_colnames,
        prevent_resolved_attribute_missing_error=prevent_resolved_attribute_missing_error,
        broadcast_join=broadcast_join,
        do_not_filter_if_null_for_inner_join=do_not_filter_if_null,
        how="inner",
    )


def exclude_by_anti_join_on_columns(
        df: DataFrame,
        df_filter: DataFrame,
        colnames: Sequence[str] = None,
        filter_colnames: Sequence[str] = None,
        prevent_resolved_attribute_missing_error: bool = False,
        broadcast_join: bool = False,
        verbose: bool = VERBOSE,
        filter_name: str = None,
) -> DataFrame:
    """
    Left-anti join on columns to filter out rows from `df`
        whose values of columns specified by `colnames` exist in `df_filter`.
    The names of corresponding columns in `df_filter` can be different from `colnames`,
        and can be specified by `filter_colnames`.

    For example, suppose `df` is,
    # +-------+---------+
    # |   name|      age|
    # +-------+---------+
    # |Michael|     null|
    # |   Andy|       31|
    # | Justin|       20|
    # |   Anna|       20|
    # +-------+---------+

    and `df_filter` is
    # +----------+---------+
    # |first_name|      age|
    # +----------+---------+
    # |      Andy|       31|
    # |      Anna|       20|
    # |       Ted|       27|
    # +----------+---------+

    The following will remove 'Andy' and 'Anna' from `df`
        because they exist in `df_filter`'s `first_name` column,
    >>> exclude_by_anti_join_on_columns(df, df_filter, ['name'], ['first_name'])
    # +-------+---------+
    # |   name|      age|
    # +-------+---------+
    # |Michael|     null|
    # | Justin|       20|
    # +-------+---------+

    and the following will remove 'Andy', 'Justin' and 'Anna' from `df`
        because their age exists_path in `df_filter`.
    >>> exclude_by_anti_join_on_columns(df, df_filter, ['age'])
    # +-------+---------+
    # |   name|      age|
    # +-------+---------+
    # |Michael|     null|
    # +-------+---------+

    Args:
        df: Left-anti join this dataframe with `df_filter` to filter out rows in `df`.
        df_filter: Left-anti join `df` with this dataframe to filter out rows in `df`.
        colnames: Join on these specified columns of `df`;
            leave this parameter empty if all columns of `df` will be used in the join operation.
        filter_colnames: Specify the column names of `df_filter`
            corresponding to `colnames` of `df`;
            leave this parameter empty if the column names of `df_filter` are the same as `df`.
        prevent_resolved_attribute_missing_error:
            True to prevent "resolved attribute missing" error;
            This errors can happen when both `df` and `df_filter`
            are coming from the same source dataframe;
            see https://medium.com/@yhoso/resolving-weird-spark-errors-f34324943e1c.
        broadcast_join: True to broadcast `df_filter`.

    Returns: the resulting dataframe of the left-anti join.

    """

    if verbose:
        hprint_message(
            f"applies dataframe-join based '{filter_name or 'exclusion'}",
            df_filter,
            "filter key columns1",
            colnames,
            "filter key columns2",
            filter_colnames,
        )

    return _filter_on_columns(
        df,
        df_filter,
        colnames=colnames,
        filter_colnames=filter_colnames,
        prevent_resolved_attribute_missing_error=prevent_resolved_attribute_missing_error,
        broadcast_join=broadcast_join,
        how="leftanti",
    )


def having(
        df: DataFrame,
        cond,
        key_cols=None,
        explode_col=None,
        agg_cols=None,
        **kwargs
):
    if explode_col is not None:
        df_filter = explode_as_flat_columns(
            df=df,
            col_to_explode=explode_col,
            explode_colname_or_prefix=get_colname(explode_col),
            overwrite_exist_column=True,
            **get_relevant_named_args(explode_as_flat_columns, **kwargs)
        )
    else:
        df_filter = df

    _KEY_COUNT_AGG_COL = 'count'
    if agg_cols is not None and agg_cols is not False:
        if agg_cols == _KEY_COUNT_AGG_COL:
            agg_cols = None
        if isinstance(agg_cols, (list, tuple)):
            agg_cols = [x for x in agg_cols if x != _KEY_COUNT_AGG_COL]
        if not agg_cols:
            agg_cols = None
        df_filter = aggregate(
            df=df_filter,
            group_cols=key_cols,
            other_agg_cols=agg_cols,
            **get_relevant_named_args(aggregate, **kwargs)
        )

    df_filter = where(
        df_filter,
        cond,
        **get_relevant_named_args(where, **kwargs)
    )

    return filter_by_inner_join_on_columns(
        df=df,
        df_filter=df_filter,
        colnames=key_cols,
        **get_relevant_named_args(filter_by_inner_join_on_columns, **kwargs)
    )


# region where function

_DIRECTIVE_FILTER = 'filter'
_DIRECTIVE_EXCLUDE = 'exclude'
_DIRECTIVE_HAVING = 'having'
_DIRECTIVE_TOP = 'top'
_DIRECTIVE_CATEGORIZED = 'categorized'


def _solve_colnames_for_join_filter(
        df1,
        df2,
        join_args,
        _solve_name_for_possible_exploded_columns=True
):
    if len(join_args) == 1:
        colnames1 = colnames2 = join_args[0]
    elif len(join_args) == 2:
        colnames1, colnames2 = join_args
    else:
        raise ValueError(f"invalid arguments for join-based filter; got {join_args}")

    if isinstance(colnames1, str) and isinstance(colnames2, str):
        _colnames1 = (
            _solve_name_for_exploded_column(df1, colnames1)
            if _solve_name_for_possible_exploded_columns
            else colnames1
        )
        _colnames2 = (
            _solve_name_for_exploded_column(df2, colnames2)
            if _solve_name_for_possible_exploded_columns
            else colnames2
        )
    elif isinstance(colnames1, (list, tuple)) and isinstance(colnames2, (list, tuple)):
        _colnames1 = (
            [_solve_name_for_exploded_column(df1, _colname) for _colname in colnames1]
            if _solve_name_for_possible_exploded_columns
            else list(colnames1)
        )
        _colnames2 = (
            [_solve_name_for_exploded_column(df2, _colname) for _colname in colnames2]
            if _solve_name_for_possible_exploded_columns
            else list(colnames2)
        )
    else:
        raise ValueError(f"invalid arguments for join-based filter; got {join_args}")

    return _colnames1, _colnames2


def _apply_join_filter(
        df, cond,
        is_exclusion=False,
        verbose=VERBOSE
):
    # we expect `cond` be one of the following,
    # 1) a dataframe
    # 2) a 2-tuple, (dataframe, named arguments)
    # 3) a 2-tuple, (dataframe, True/False or 'exclusion'/'filter')
    # 4) a 2-tuple, (dataframe, (True/False or 'exclusion'/'filter', join key columns))
    # 5) a 2-tuple, (dataframe, (True/False or 'exclusion'/'filter', join key columns1, join key columns2))

    if isinstance(cond, DataFrame):
        # if `cond` is a dataframe, then treating it as a filter dataframe
        if is_exclusion:
            hprint_message('applies dataframe-based exclusion', str(cond))
            return exclude_by_anti_join_on_columns(df, cond, verbose=verbose)
        else:
            hprint_message('applies dataframe-based filter', str(cond))
            return filter_by_inner_join_on_columns(df, cond, verbose)
    elif isinstance(cond, tuple) and len(cond) == 2 and isinstance(cond[0], DataFrame):
        # specify more options in a 2-tuple,
        # the first object is a dataframe;
        df_filter, _args = cond
        if isinstance(_args, Mapping):
            # the second object is a dictionary of arguments for
            # `exclude_by_anti_join_on_columns` or `filter_by_inner_join_on_columns`.
            is_exclusion = _args.get('exclusion', False)
            _args = {k: v for k, v in _args.items() if k != 'exclusion'}
            if is_exclusion:
                return exclude_by_anti_join_on_columns(
                    df, df_filter,
                    verbose=verbose,
                    **_args
                )
            else:
                return filter_by_inner_join_on_columns(
                    df, df_filter,
                    verbose=verbose,
                    **_args
                )
        else:
            # the second object is a tuple/list of length 1, 2 or 3 to indicate
            # 1) if it is filter or exclusion (optional for filter, must-have for exclusion);
            # 2) the names of columns used as keys for the filter or exclusion (specify one list if
            # the column names are the same for the dataframe and the filter,
            # otherwise specify two lists of column names)
            _args = make_list_(_args)
            if (
                    _args[0] in DIRECTIVE_COND_NEGATION
                    or _args[0] == _DIRECTIVE_EXCLUDE
                    or _args[0] is True
            ):
                _args = _args[1:]
                is_exclusion = True
            else:
                if _args[0] == _DIRECTIVE_FILTER or _args[0] is False:
                    _args = _args[1:]

            if not _args:
                colnames1 = colnames2 = None
            else:
                colnames1, colnames2 = _solve_colnames_for_join_filter(
                    df, df_filter, _args
                )
            if is_exclusion is False:
                return filter_by_inner_join_on_columns(
                    df, df_filter,
                    colnames=colnames1,
                    filter_colnames=colnames2,
                    verbose=verbose
                )
            elif is_exclusion is True:
                return exclude_by_anti_join_on_columns(
                    df, df_filter,
                    colnames=colnames1,
                    filter_colnames=colnames2,
                    verbose=verbose
                )


def _where_with_null_cond_check(
        df: DataFrame,
        cond: Column,
        null_cond_tolerance=0.0001,
        cache_option=CacheOptions.IMMEDIATE,
        verbose=VERBOSE
):
    if null_cond_tolerance is None or null_cond_tolerance < 0:
        return df.where(cond)
    _KEY_FILTER = 'filter'
    _tmp_filter_field_name = get_internal_colname(_KEY_FILTER)
    df = cache__(
        df.withColumn(_tmp_filter_field_name, cond),
        cache_option=cache_option,
        name='where_with_null_cond_check' if verbose else None,
        unpersist=df
    )
    if verbose:
        from boba_python_utils.spark_utils.analysis import show_counts
        _df = df
        if _KEY_FILTER in _df.columns:
            _df = _df.drop(_KEY_FILTER)
        _df = _df.withColumnRenamed(_tmp_filter_field_name, _KEY_FILTER)
        cnts = show_counts(
            _df, [_KEY_FILTER], return_counts_dataframe=True
        ).collect()
        cnt_total = sum(cnt['count'] for cnt in cnts)
        cnt_null = sum(cnt['count'] for cnt in cnts if cnt[0] is None)
        if not _is_negligible(cnt_null, cnt_total, tolerance=null_cond_tolerance):
            df.unpersist()
            raise ValueError(f"too many null values for condition '{cond}'")
    else:
        if not is_negligible(
                df, F.col(_tmp_filter_field_name).isNull(), tolerance=null_cond_tolerance
        ):
            df.unpersist()
            raise ValueError(f"too many null values for condition '{cond}'")

    df = cache__(
        df.where(_tmp_filter_field_name).drop(_tmp_filter_field_name),
        name='where_with_null_cond_check (after filter)' if verbose else None,
        unpersist=df
    )
    return df


def is_negligible(df: DataFrame, cond, tolerance: float = 0.001, **kwargs):
    return _is_negligible(where(df, cond, **kwargs).count(), df.count(), tolerance=tolerance)


def where(
        df: DataFrame,
        cond: Union[str, Column, Mapping],
        skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
        spark: SparkSession = None,
        id_cols=None,
        negate_cond: bool = False,
        categorized_cond_parse_options: CategorizedCondParseOptions =
        CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside,
        null_cond_tolerance=None,
        cache_option_for_null_cond_check=CacheOptions.IMMEDIATE,
        verbose: bool = VERBOSE,
        **kwargs
):
    """
    Filters a PySpark DataFrame based on the given condition(s). The function supports multiple
    types of conditions, including simple column-based conditions, special filter conditions,
    and mapping-based conditions with multiple filters.

    Args:
        df: The PySpark DataFrame to filter.
        cond: The condition(s) to filter the DataFrame by.
        skip_cond_str: A string indicating that the condition should not be applied.
        spark: The SparkSession object, if required for parsing certain conditions.
        id_cols: The ID columns, required for certain types of conditions.
        negate_cond: If True, negates the specified condition(s).
        categorized_cond_parse_options: Options for parsing categorized conditions.
        null_cond_tolerance: The tolerance for null conditions.
        cache_option_for_null_cond_check: Cache options for null condition checks.
        verbose: If True, display verbose output.


    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.master("local").appName("where_example").getOrCreate()
        >>> data = [(1, "apple"), (2, "orange"), (3, "banana"), (4, "grape")]
        >>> columns = ["id", "fruit"]
        >>> df = spark.createDataFrame(data, columns)

        # Filter by a simple condition on the 'id' column.
        >>> cond = {'id': 3}
        >>> result = where(df, cond)
        >>> result.show()
        +---+------+
        | id| fruit|
        +---+------+
        |  3|banana|
        +---+------+

        # Filter by a mapping-based condition.
        >>> cond = {"id": ">1", "fruit": "!orange"}
        >>> result = where(df, cond)
        >>> result.show()
        +---+------+
        | id| fruit|
        +---+------+
        |  2|banana|
        |  3| grape|
        +---+------+
    """
    if is_none_or_empty_str(cond) or str_eq(cond, skip_cond_str):
        return df

    def _try_parse_cond_as_input(_cond):
        # this allows `_cond` be something parsable by `solve_input`, such as data paths;
        # the loaded dataframe will be used to filter `df`
        if spark is not None:
            if isinstance(_cond, (tuple, list)) and isinstance(_cond[1], Mapping):
                try:
                    _cond = solve_input(
                        _cond[0],
                        spark=spark,
                        **_cond[1]
                    )
                except:
                    pass
            else:
                try:
                    _cond = solve_input(
                        _cond,
                        spark=spark
                    )
                except:
                    pass
        return _cond

    def _solve_single_cond(df, _cond):
        # region STEP1: try to parse `_cond` as a Spark dataframe
        _cond = _try_parse_cond_as_input(_cond)
        # endregion

        # region STEP2: try if the current condition `_cond` represents a filter dataframe

        # if the returned `df_with_special_filter` is not None,
        # then it means the filter is success,
        # and we return `df_with_special_filter`;
        # otherwise `_cond` is not a filter dataframe,
        # and we will continue trying other possibilities
        df_with_special_filter = _apply_join_filter(
            df,
            _cond,
            is_exclusion=False,
            verbose=verbose
        )
        if df_with_special_filter is not None:
            return df_with_special_filter, None
        # endregion

        # region STEP3: `_cond` is a mapping containing multiple filters
        has_special_filter = False
        categorized_conds = []
        if isinstance(_cond, Mapping):
            if not _cond:
                raise ValueError(f"unrecognized condition '{_cond}'")

            # process all dataframe based filters in the mapping,
            # and saves all other conditions in `unparsed_conditions`
            unparsed_conditions = {}
            for k, _cond in _cond.items():
                df_with_special_filter = None
                k_option, _k = solve_compare_option(
                    k,
                    contains_indicator=None,
                    starts_with_indicator=None,
                    ends_with_indicator=None,
                    lower_lexical_order_indicator=None,
                    higher_lexical_order_indicator=None,
                    regular_expression_indicator=None,
                    negation_indicators=DIRECTIVE_COND_NEGATION,
                    case_insensitive_indicator=None,
                    ignore_null_indicator=None,
                    other_option_indicators=DIRECTIVE_NON_COLNAME,
                    option_at_start=True,
                    option_at_end=False,
                    return_none_if_no_option_available=False
                )

                k_is_directive = (
                        k_option.other_options and
                        DIRECTIVE_NON_COLNAME in k_option.other_options
                )

                if k_is_directive:
                    def _disp_filter_info():
                        if verbose:
                            hprint_message(
                                'filter', filter_idx,
                                'directive', _k,
                                'arg', _cond,
                                sep='\t'
                            )

                    if _k.startswith(_DIRECTIVE_CATEGORIZED):
                        _disp_filter_info()
                        categorized_conds.append(
                            F.solve_categorized_cond(
                                category_key_cond_map=_cond,
                                cond_key_colname=_k,
                                df=df
                            )
                        )
                    elif _k.startswith(_DIRECTIVE_HAVING):
                        _disp_filter_info()
                        df_with_special_filter = apply_arg(
                            having,
                            (df, *iter_(_cond)),
                            allows_mixed_positional_and_named_arg=True
                        )

                    elif _k.startswith(_DIRECTIVE_FILTER):
                        _cond = _try_parse_cond_as_input(_cond)
                        _disp_filter_info()
                        df_with_special_filter = _apply_join_filter(
                            df,
                            _cond,
                            is_exclusion=False,
                            verbose=verbose
                        )
                    elif _k.startswith(_DIRECTIVE_EXCLUDE):
                        _cond = _try_parse_cond_as_input(_cond)
                        _disp_filter_info()
                        df_with_special_filter = _apply_join_filter(
                            df,
                            _cond,
                            is_exclusion=True,
                            verbose=verbose
                        )
                    elif _k.startswith(_DIRECTIVE_TOP):
                        _disp_filter_info()
                        df_with_special_filter = apply_arg(
                            top_from_each_group,
                            (df, *iter_(_cond)),
                            allows_mixed_positional_and_named_arg=True
                        )

                # the only use of `k` for dataframe based filtering
                # is that if it includes keyword 'exclude', then we assume it is exclusion
                if df_with_special_filter is not None:
                    df = df_with_special_filter
                    has_special_filter = True
                else:
                    unparsed_conditions[k] = _cond
            _cond = (
                unparsed_conditions
                if unparsed_conditions
                else None
            )

        # endregion

        _cond = (
            F.solve_cond(
                cond=_cond,
                df=df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=categorized_cond_parse_options,
            )
            if _cond is not None
            else None
        )

        if categorized_conds:
            _cond = F.and_(
                *categorized_conds,
                _cond
            )
        return (df if has_special_filter else None), _cond

    df_with_special_filter = []
    non_special_filer_conds = []
    null_cond_check_enabled = not (null_cond_tolerance is None or null_cond_tolerance < 0)
    for filter_idx, _cond in enumerate(iter_(cond)):
        _df, _cond = _solve_single_cond(df, _cond)
        if _cond is None:
            if _df is not None:
                df_with_special_filter.append(_df)
        else:
            if verbose:
                hprint_message(
                    'filter', filter_idx,
                    'condition', _cond,
                    'negate_cond', negate_cond,
                    sep='\t'
                )
            if _df is None:
                non_special_filer_conds.append(_cond)
            else:
                df_with_special_filter.append(
                    _df.where(cond)
                    if not null_cond_check_enabled
                    else _where_with_null_cond_check(
                        _df, _cond,
                        null_cond_tolerance=null_cond_tolerance,
                        cache_option=cache_option_for_null_cond_check,
                        verbose=verbose
                    )
                )

    non_special_filer_conds = F.or_(non_special_filer_conds)

    if df_with_special_filter:
        if negate_cond:
            raise ValueError(
                f"'negate_cond' is currently not supported for condition '{cond}'; "
                f"use 'negate_cond' only if there is no special filter "
                f"in the specified condition"
            )
        if non_special_filer_conds is not None:
            df_with_special_filter.append(
                df.where(non_special_filer_conds)
                if not null_cond_check_enabled
                else _where_with_null_cond_check(
                    df, non_special_filer_conds,
                    null_cond_tolerance=null_cond_tolerance,
                    cache_option=cache_option_for_null_cond_check,
                    verbose=verbose
                )
            )

        if len(df_with_special_filter) == 1:
            df = df_with_special_filter[0]
        else:
            if id_cols is None:
                raise ValueError(
                    f"'id_cols' is required for filter condition '{cond}'"
                )
            df = cache__(
                priority_union(
                    id_cols,
                    *df_with_special_filter
                ),
                cache_option=(
                    cache_option_for_null_cond_check
                    if null_cond_check_enabled
                    else CacheOptions.NO_CACHE
                ),
                unpersist=(
                    filter(lambda x: x is not df, df_with_special_filter)
                    if null_cond_check_enabled
                    else None
                )
            )
    elif non_special_filer_conds is not None:
        if negate_cond:
            non_special_filer_conds = (~non_special_filer_conds)
        df = _where_with_null_cond_check(
            df, non_special_filer_conds,
            null_cond_tolerance=null_cond_tolerance,
            cache_option=cache_option_for_null_cond_check,
            verbose=verbose
        )

    return df


# endregion

def multi_level_filter(df: DataFrame, conds: Iterable, id_cols=None) -> List[DataFrame]:
    out = []

    for cond in conds:
        if cond is None:
            continue
        out.append(where(df, cond))
        try:
            df = where(df, cond, negate_cond=True)
        except:
            if id_cols is not None:
                df = exclude_by_anti_join_on_columns(
                    df, out[-1], id_cols
                )
            else:
                raise ValueError(
                    "must specify 'id_cols' when 'conds' uses special filter"
                )
    out.append(df)
    return out


def iter_subdata(
        df: DataFrame,
        data_name: str,
        subdata_conds: Mapping,
        cache_option: CacheOptions = CacheOptions.IMMEDIATE,
        skip_cond_str=DIRECTIVE_COND_NOT_APPLIED,
        subdata_cond_names=None,
):
    negation_shortcuts = {}
    for k, _conds in subdata_conds.items():
        prev_cond = None
        has_negation_shortcut = False
        for _cond in _conds:
            if isinstance(_cond, str) and len(_cond) == 1 and _cond in DIRECTIVE_COND_NEGATION:
                if prev_cond is None:
                    has_negation_shortcut = True
                else:
                    negation_shortcuts[k] = prev_cond
                    prev_cond = None
            else:
                if has_negation_shortcut:
                    negation_shortcuts[k] = _cond
                    has_negation_shortcut = False
                else:
                    prev_cond = _cond

    if not subdata_cond_names:
        subdata_cond_names = ()

    for subdata_name, _conds in zip_longest(subdata_cond_names, product(*subdata_conds.values())):
        _conds2 = {}
        for k, v in zip(subdata_conds.keys(), _conds):
            if isinstance(v, str) and len(v) == 1 and v in DIRECTIVE_COND_NEGATION:
                _conds2[f'~{k}'] = negation_shortcuts[k]
            else:
                _conds2[k] = v

        if not subdata_name:
            subdata_name = f'{data_name}-{_conds2}'
        else:
            subdata_name = f'{data_name}-{subdata_name}'

        yield subdata_name, cache__(
            where(df, _conds2, skip_cond_str=skip_cond_str),
            name=subdata_name,
            cache_option=cache_option
        )


# region self-join and 3-hop construction

def self_join_by_cartesian_product(
        df: DataFrame,
        join_key_cols: Iterable[NameOrColumn],
        cols_to_join: Iterable[NameOrColumn] = None,
        make_cols_to_join_unique: bool = False,
        sort_func_for_cols_to_join: Callable = None,
        src_colname_or_suffix='_1',
        trg_colname_or_suffix='_2',
        bidirection_product: bool = True,
        include_self_product: bool = True,
        unpack_self_join_tuple: bool = True,
        spark: SparkSession = None,
        task_name: str = 'self_join_by_cartesian_product',
        cache_option: CacheOptions = CacheOptions.IMMEDIATE
):
    """
    Performs a self-join on a DataFrame by computing the Cartesian product of elements in specified
    columns.

    Args:
        df: The input DataFrame.
        join_key_cols: The columns to group the DataFrame by before computing the Cartesian product.
        cols_to_join: The columns to compute the Cartesian product on. Defaults to None.
        make_cols_to_join_unique: If True, only join unique column values. Defaults to False.
        sort_func_for_cols_to_join: A function to sort the input columns before computing the
            Cartesian product. Defaults to None.
        src_colname_or_suffix: The suffix or column name for the first element in the Cartesian
            product. Defaults to '_1'.
        trg_colname_or_suffix: The suffix or column name for the second element in the Cartesian
            product. Defaults to '_2'.
        bidirection_product: If True, include both (a, b) and (b, a) in the Cartesian product.
        include_self_product: If True, include (a, a) in the Cartesian product. Defaults to True.
        unpack_self_join_tuple: If True, unpack the Cartesian product tuple into separate columns.
        spark: SparkSession instance.
        task_name: Name of the task.
        cache_option: Caching option for intermediate results.

    Returns:
        DataFrame: A new DataFrame with the self-join Cartesian product applied.

    Examples:
        >>> data = [("A", 1), ("A", 2), ("B", 3), ("B", 4)]
        >>> schema = StructType([
                StructField("key", StringType(), True),
                StructField("value", IntegerType(), True)
            ])
        >>> df = spark.createDataFrame(data, schema=schema)
        >>> result = self_join_by_cartesian_product(df, join_key_cols=["key"], cols_to_join=["value"])
        >>> result.show()
        +---+------+--------+
        |key|value_1|value_2|
        +---+------+--------+
        |  A|     1|       1|
        |  A|     1|       2|
        |  A|     2|       1|
        |  A|     2|       2|
        |  B|     3|       3|
        |  B|     3|       4|
        |  B|     4|       3|
        |  B|     4|       4|
        +---+------+--------+

        # Exclude self product
        >>> result = self_join_by_cartesian_product(df, join_key_cols=["key"], cols_to_join=["value"],
        ...                                         include_self_product=False)
        >>> result.show()
        +---+------+--------+
        |key|value_1|value_2|
        +---+------+--------+
        |  A|     1|       2|
        |  A|     2|       1|
        |  B|     3|       4|
        |  B|     4|       3|
        +---+------+--------+

        # Unidirectional product
        >>> result = self_join_by_cartesian_product(df, join_key_cols=["key"], cols_to_join=["value"],
        ...                                         bidirection_product=False)
        >>> result.show()
        +---+------+--------+
        |key|value_1|value_2|
        +---+------+--------+
        |  A|     1|       1|
        |  A|     1|       2|
        |  A|     2|       2|
        |  B|     3|       3|
        |  B|     3|       4|
        |  B|     4|       4|
        +---+------+--------+

    """
    if src_colname_or_suffix == trg_colname_or_suffix:
        raise ValueError("'src_colname_or_suffix' and 'trg_colname_or_suffix' cannot be the same")
    group_colnames = [get_colname(df, _col) for _col in iter_(join_key_cols)]
    if cols_to_join is None:
        cols_to_join = list(filter(lambda x: x not in group_colnames, df.columns))

    _fold_colname = get_internal_colname('tmp_fold')
    df_folded = fold(
        df,
        group_cols=join_key_cols,
        fold_colname=_fold_colname,
        cols_to_fold=cols_to_join,
        collect_set=make_cols_to_join_unique
    )

    if not include_self_product:
        df_folded = df_folded.where(
            F.size(_fold_colname) > 1
        )

    df_folded = cache__(
        df_folded,
        name=f'{task_name} (df_folded)',
        cache_option=cache_option
    )

    _cartesian_product_output_colname = get_internal_colname('tmp_cartesian_product')
    _item1_fieldname, _item2_fieldname = src_colname_or_suffix, trg_colname_or_suffix
    if unpack_self_join_tuple:
        _item1_fieldname = get_internal_colname(src_colname_or_suffix)
        _item2_fieldname = get_internal_colname(trg_colname_or_suffix)

    df_cp = cache__(
        array_self_cartesian_product(
            df_folded,
            arr_col=_fold_colname,
            arr_sort_func=sort_func_for_cols_to_join,
            output_arr_colname=_cartesian_product_output_colname,
            item1_fieldname=_item1_fieldname,
            item2_fieldname=_item2_fieldname,
            bidirection_product=bidirection_product,
            include_self_product=include_self_product
        ).drop(_fold_colname),
        repartition=True,
        spark=spark,
        name=f'{task_name} (df_cp)',
        cache_option=cache_option,
        unpersist=df_folded
    )

    df_cp_flat = explode_as_flat_columns(
        df_cp,
        col_to_explode=_cartesian_product_output_colname
    )

    if unpack_self_join_tuple:
        df_cp_flat = unfold_struct(
            df_cp_flat,
            struct_colname=_item1_fieldname,
            col_name_suffix=src_colname_or_suffix,
            prefix_suffix_sep=''
        )

        df_cp_flat = unfold_struct(
            df_cp_flat,
            struct_colname=_item2_fieldname,
            col_name_suffix=trg_colname_or_suffix,
            prefix_suffix_sep=''
        )

    df_cp_flat = cache__(
        df_cp_flat,
        repartition=True,
        spark=spark,
        name=f'{task_name} (df_cp_flat)',
        cache_option=cache_option,
        unpersist=df_cp
    )

    return df_cp_flat


def build_3hop_affinity(
        df_one_hop_affinity: DataFrame,
        src_id_colnames: Union[str, Iterable[str]] = 'user_id',
        trg_id_colnames: Union[str, Iterable[str]] = 'item_id',
        df_two_hop_affinity_id_only: DataFrame = None,
        two_hop_affinity_unique_impression_colname: str = None,
        two_hop_affinity_impression_colname: str = None,
        two_hop_affinity_other_agg_cols: Iterable[NameOrColumn] = None,
        two_hop_affinity_impression_filter: Union[int, Callable, NameOrColumn, Mapping] = None,
        two_hop_affinity_select_top_impression: int = None,
        return_unseen_3hop_only: bool = False,
        spark: SparkSession = None,
        task_name: str = 'build_3hop_affinity',
        cache_option: CacheOptions = CacheOptions.IMMEDIATE
):
    """
    Builds a DataFrame with 3-hop affinity based on the given one-hop affinity DataFrame.

    Args:
        df_one_hop_affinity: The input one-hop affinity DataFrame.
        src_id_colnames: The source ID column names. Defaults to 'user_id'.
        trg_id_colnames: The target ID column names. Defaults to 'item_id'.
        df_two_hop_affinity_id_only: A pre-computed DataFrame for two-hop affinity. If None,
            it will be computed based on `df_one_hop_affinity`.
        two_hop_affinity_unique_impression_colname: Column name for unique impressions.
        two_hop_affinity_impression_colname: Column name for total impressions.
        two_hop_affinity_other_agg_cols: Other aggregation columns to include.
        two_hop_affinity_impression_filter: Filter for selecting impressions.
        two_hop_affinity_select_top_impression: Number of top impressions to select.
        return_unseen_3hop_only: If True, return only unseen 3-hop affinity.
        spark: SparkSession instance.
        task_name: Name of the task.
        cache_option: Caching option for intermediate results.

    Returns:
        DataFrame: A new DataFrame with 3-hop affinity.

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.getOrCreate()
        >>> data = [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Z")]
        >>> schema = ["user_id", "item_id"]
        >>> df_one_hop_affinity = spark.createDataFrame(data, schema=schema)
        >>> df_3hop = build_3hop_affinity(df_one_hop_affinity)
        >>> df_3hop.show()
        +-------+---------+-------+
        |user_id|user_id_2|item_id|
        +-------+---------+-------+
        |      A|        B|      X|
        |      A|        B|      Z|
        |      B|        A|      X|
        |      B|        A|      Y|
        +-------+---------+-------+

        >>> df_3hop = build_3hop_affinity(df_one_hop_affinity, return_unseen_3hop_only=True)
        >>> df_3hop.show()
        +-------+---------+-------+
        |user_id|user_id_2|item_id|
        +-------+---------+-------+
        |      A|        B|      Z|
        |      B|        A|      Y|
        +-------+---------+-------+
    """
    src_id_colnames = make_list_(src_id_colnames)
    trg_id_colnames = make_list_(trg_id_colnames)
    src_trg_id_colnames = [*src_id_colnames, *trg_id_colnames]

    src_id_suffix1 = '_1'
    src_id_suffix2 = '_2'
    src_id_colnames1_for_2hop = list(x + src_id_suffix1 for x in src_id_colnames)
    src_id_colnames2_for_2hop = list(x + src_id_suffix2 for x in src_id_colnames)
    src_id_colnames_for_2hop = src_id_colnames1_for_2hop + src_id_colnames2_for_2hop

    # region STEP1: compute `df_two_hop_affinity_id_only` if not provided
    _df_two_hop_affinity_id_only_for_unpersist = None
    if df_two_hop_affinity_id_only is None:
        df_one_hop_affinity_id_only = df_one_hop_affinity.select(
            *src_id_colnames,
            *trg_id_colnames
        )
        if not two_hop_affinity_impression_colname:
            df_one_hop_affinity_id_only = df_one_hop_affinity_id_only.distinct()

        df_two_hop_affinity_id_only = self_join_by_cartesian_product(
            df_one_hop_affinity_id_only,
            join_key_cols=trg_id_colnames,
            bidirection_product=True,
            include_self_product=False,
            unpack_self_join_tuple=True,
            src_colname_or_suffix=src_id_suffix1,
            trg_colname_or_suffix=src_id_suffix2,
            spark=spark,
            task_name=task_name,
            cache_option=cache_option
        )
        _df_two_hop_affinity_id_only_for_unpersist = df_two_hop_affinity_id_only

    if two_hop_affinity_impression_colname:
        if two_hop_affinity_impression_colname in df_two_hop_affinity_id_only.columns:
            agg_cols = [
                F.sum(
                    two_hop_affinity_impression_colname
                ).alias(two_hop_affinity_impression_colname)
            ]
        else:
            agg_cols = [
                F.count('*').alias(two_hop_affinity_impression_colname)
            ]

        if two_hop_affinity_unique_impression_colname:
            agg_cols.append(
                F.countDistinct(
                    *src_id_colnames_for_2hop
                ).alias(two_hop_affinity_unique_impression_colname)
            )

        if not is_none_or_empty_str(two_hop_affinity_other_agg_cols):
            agg_cols.extend(iter_(two_hop_affinity_other_agg_cols))

        df_two_hop_affinity_id_only = df_two_hop_affinity_id_only.groupBy(
            *src_id_colnames_for_2hop
        ).agg(*agg_cols)
    elif two_hop_affinity_unique_impression_colname:
        agg_cols = [F.count('*').alias(two_hop_affinity_unique_impression_colname)]
        if is_none_or_empty_str(two_hop_affinity_other_agg_cols):
            df_two_hop_affinity_id_only = df_two_hop_affinity_id_only.dropDuplicates(
                [*src_id_colnames_for_2hop, *trg_id_colnames]
            )
        else:
            agg_cols.extend(iter_(two_hop_affinity_other_agg_cols))

        df_two_hop_affinity_id_only = df_two_hop_affinity_id_only.groupBy(
            *src_id_colnames_for_2hop
        ).agg(*agg_cols)
    # endregion

    # region STEP2: filter by `two_hop_affinity_impression_filter`
    if isinstance(two_hop_affinity_impression_filter, int):
        if two_hop_affinity_impression_filter > 1 and two_hop_affinity_unique_impression_colname:
            df_two_hop_affinity_id_only = df_two_hop_affinity_id_only.where(
                F.col(two_hop_affinity_unique_impression_colname) >= two_hop_affinity_impression_filter
            )
        elif two_hop_affinity_impression_colname:
            df_two_hop_affinity_id_only = df_two_hop_affinity_id_only.where(
                F.col(two_hop_affinity_impression_colname) >= two_hop_affinity_impression_filter
            )
    elif callable(two_hop_affinity_impression_filter):
        df_two_hop_affinity_id_only = two_hop_affinity_impression_filter(
            df_two_hop_affinity_id_only
        )
    else:
        df_two_hop_affinity_id_only = where(
            df_two_hop_affinity_id_only,
            two_hop_affinity_impression_filter
        )
    # endregion

    # region STEP3: top selection by `two_hop_affinity_select_top_impression`
    if two_hop_affinity_select_top_impression:
        order_cols = None
        if two_hop_affinity_unique_impression_colname:
            order_cols = [
                F.col(two_hop_affinity_unique_impression_colname).desc()
            ]
            if two_hop_affinity_impression_colname:
                order_cols.append(F.col(two_hop_affinity_impression_colname).desc())
        elif two_hop_affinity_impression_colname:
            order_cols = [F.col(two_hop_affinity_impression_colname).desc()]

        df_two_hop_affinity_id_only = top_from_each_group(
            df_two_hop_affinity_id_only,
            group_cols=src_id_colnames1_for_2hop,
            order_cols=order_cols,
            top=two_hop_affinity_select_top_impression
        )
    # endregion

    # region STEP4: compute 3-hop affinity `df_3hop`
    df_3hop = join_on_columns(
        df_two_hop_affinity_id_only,
        df_one_hop_affinity,
        src_id_colnames2_for_2hop,
        src_id_colnames
    )

    df_3hop = rename(
        df_3hop,
        {
            k: v
            for k, v in zip(src_id_colnames_for_2hop, src_id_colnames)
        }
    )

    from boba_python_utils.spark_utils import one_from_each_group
    df_3hop = one_from_each_group(
        df_3hop,
        group_cols=src_trg_id_colnames
    )

    df_3hop = cache__(
        df_3hop,
        repartition=True,
        name=f'{task_name} (df_3hop)',
        unpersist=_df_two_hop_affinity_id_only_for_unpersist
    )

    if return_unseen_3hop_only:
        df_3hop = cache__(
            exclude_by_anti_join_on_columns(
                df_3hop,
                df_one_hop_affinity,
                src_trg_id_colnames
            ),
            repartition=True,
            name=f'{task_name} (df_3hop-unseen)',
            unpersist=df_3hop
        )
    # endregion

    return df_3hop

# endregion
