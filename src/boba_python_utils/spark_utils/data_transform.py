from boba_python_utils.common_utils.typing_helper import make_list_if_not_none_
from boba_python_utils.general_utils.general import make_list_

from boba_python_utils.spark_utils.common import *
from boba_python_utils.spark_utils.spark_functions.common import or_, col_
from boba_python_utils.general_utils.strex import (
    add_suffix, add_prefix_suffix, replace_suffix, replace_prefix, solve_name_conflict,
)
from boba_python_utils.string_utils.prefix_suffix import remove_prefix_suffix


# region null-value operations
def fillna(df: DataFrame, *value_and_cols: Union[Tuple[Any, Union[str, List[str]]]]) -> DataFrame:
    """
    Fills null values in a Spark DataFrame columns with specified values.

    Args:
        df: Input Spark DataFrame with null values.
        value_and_cols: One or more value and column name pairs;
            for each column name, the null values should be filled with the value.

    Returns:
        A Spark DataFrame with null values filled according to the provided value-column pairs.

    Examples:
        # Assuming 'spark' is a SparkSession object and 'data' is a list of dictionaries
        >>> data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": None}]
        >>> df = spark.createDataFrame(data)

        # Fill missing age values with -1 and missing name values with 'Jane Doe'
        >>> filled_df = fillna(df, (-1, "age"), ("Jane Doe", "name"))
    """
    for value, colnames in solve_key_value_pairs(*value_and_cols):
        df = df.fillna(value, make_list_(colnames))
    return df


def set_null_if_empty_str(df: DataFrame, *cols):
    for _col in cols:
        if isinstance(_col, tuple):
            _main_col = col_(_col[0])
        else:
            _main_col = col_(_col)
            _col = (_col,)

        for _col_to_set_null in _col:
            _col_to_set_null_name = get_colname(df, _col_to_set_null)
            df = df.withColumn(
                _col_to_set_null_name,
                F.when(
                    (_main_col == ''),
                    F.lit(None)
                ).otherwise(
                    col_(_col_to_set_null)
                )
            )
    return df


# endregion

# region misc
def promote_columns(df: DataFrame, *colnames: str) -> DataFrame:
    colnames = solve_nested_singleton_tuple_list(colnames)
    return df.select(
        *(colname for colname in colnames if colname in df.columns),
        *(colname for colname in df.columns if colname not in colnames)
    )


# endregion

# region naming


def rename(df, *name_map, ignore_non_existing_column=False, select_renamed_columns: bool = False):
    """
    Renames columns of the dataframe according to the `name_map`, which can be a mapping,
        or a sequence of tuples.

    The function is equivalent to multiple `withColumnRenamed`;
        for example, the following are equivalent
    >>> df.withColumnRenamed('a', 'a1').withColumnRenamed('b', 'b1')
    >>> rename(df, [('a', 'a1'), ('b', 'b1')])
    >>> rename(df, { 'a': 'a1', 'b': 'b1' })

    """
    names = tuple(zip(
        *(
            (src_name, trg_name) for src_name, trg_name in solve_key_value_pairs(name_map)
            if src_name != trg_name and ((not ignore_non_existing_column) or (src_name in df.columns))
        )
    ))

    if not names:
        return df

    src_names, trg_names = names

    # two iterations in case we are doing renaming like
    # { 'a': 'b',  'b': 'c'}
    # first we should first rename 'a' to a temporary column name like '___b', and 'b' to '___c',
    _trg_names = []
    for src_name, trg_name in zip(src_names, trg_names):
        _trg_name = solve_name_conflict(
            name=get_internal_colname(trg_name),
            existing_names=df.columns,
        )
        df = df.withColumnRenamed(src_name, _trg_name)
        _trg_names.append(_trg_name)
    # and then rename '___b' to 'b' and '___c' to 'c'
    for _trg_name, trg_name in zip(_trg_names, trg_names):
        df = df.withColumnRenamed(_trg_name, trg_name)

    if select_renamed_columns:
        df = df.select(*trg_names)

    return df


def rename_by_replacing_prefix(df: DataFrame, prefix: str, replacement: str, sep='_'):
    rename_dict = {}
    if prefix != replacement:
        for src_name in df.columns:
            if src_name.startswith(prefix):
                trg_name = replace_prefix(src_name, prefix, replacement, sep)
                if src_name != trg_name:
                    rename_dict[src_name] = trg_name
        if rename_dict:
            return rename(df, rename_dict)
    return df


def rename_by_replacing_suffix(df: DataFrame, suffix: str, replacement: str, sep='_'):
    rename_dict = {}
    if suffix != replacement:
        for src_name in df.columns:
            if src_name.endswith(suffix):
                trg_name = replace_suffix(src_name, suffix, replacement, sep)
                if src_name != trg_name:
                    rename_dict[src_name] = trg_name
        if rename_dict:
            return rename(df, rename_dict)
    return df


def rename_by_swapping_suffix(df: DataFrame, suffix1: str, suffix2: str, sep='_'):
    rename_dict = {}
    if suffix1 != suffix2:
        for src_name in df.columns:
            if src_name.endswith(suffix1):
                trg_name = replace_suffix(src_name, suffix1, suffix2, sep)
                if src_name != trg_name:
                    rename_dict[src_name] = trg_name
            elif src_name.endswith(suffix2):
                trg_name = replace_suffix(src_name, suffix2, suffix1, sep)
                if src_name != trg_name:
                    rename_dict[src_name] = trg_name
        if rename_dict:
            return rename(df, rename_dict)
    return df


def rename_by_adding_suffix(
        df,
        suffix,
        included_cols_names: List[str] = None,
        excluded_col_names: List[str] = None
):
    """
    Appending a `suffix` to every column name specified in `included_cols_names`
        except for those in `excluded_col_names`.
    If `included_cols_names` is not specified,
        then every field in the dataframe will get the suffix.

    For example, for dataframe { "a": "x", "b": "y" },
    >>> rename_by_adding_suffix(df, suffix=2)
    will result in { "a_2": "x", "b_2": "y" }.

    """
    if included_cols_names is None:
        included_cols_names = df.columns

    return rename(
        df,
        *(
            (col_name, add_suffix(col_name, suffix))
            for col_name in included_cols_names
            if (excluded_col_names is None or col_name not in excluded_col_names)
        )
    )


def rename_by_adding_prefix(df, prefix, included_cols_names=None, excluded_col_names=None):
    """
    Appending a `prefix` to every column name specified in `included_cols_names`
        except for those in `excluded_col_names`.
    If `included_cols_names` is not specified,
        then every field in the dataframe will get the prefix.
    """
    if included_cols_names is None:
        included_cols_names = df.columns

    return rename(
        df,
        (
            (col_name, str(prefix) + col_name)
            for col_name in included_cols_names
            if (excluded_col_names is None or col_name not in excluded_col_names)
        ),
    )


# endregion

# region flatten

def explode_as_flat_columns(
        df: DataFrame,
        col_to_explode: Union[str, Column],
        explode_colname_or_prefix: Optional[str] = None,
        explode_colname_suffix: Optional[str] = None,
        select_struct_field_names: Optional[List[str]] = None,
        prefix_suffix_exempted_struct_field_names: Optional[List[str]] = None,
        index_colname: Optional[str] = None,
        new_cols: Mapping[str, Column] = None,
        overwrite_exist_column: bool = False
) -> DataFrame:
    """
    Explode a specified array column in the dataframe to flattened column(s).
    If the array column is an array of structs,
        fields in the struct will be extracted as flattened columns.

    Args:
        df: the dataframe.
        col_to_explode: the column to explode.
        explode_colname_or_prefix:
            1) if the `col_to_explode` is a value array, then this will be the column name of
                the exploded values;
            2) if the `col_to_explode` is a struct array,
                then we extract struct fields from the structs as top-level columns
                in the returned dataframe, and specify this parameter to add prefix to
                the field names as the column names.
        explode_colname_suffix: specifies a name suffix to all new columns
            generated by the explosion.
        select_struct_field_names: when `col_to_explode` is an array of structs,
            then flatten the fields of these names in the struct as top-level columns
            in the returned dataframe.
        new_cols: a convenience parameter if we would like to create new columns
            ('withColumn' transformation) right after the explosion;
            this function will perform some name conflict checking
            between the columns created through explosion and
            through this `new_cols` parameter.
        index_colname:
            adds an index column to track the position of the record in the original array column.
        overwrite_exist_column:
            True to overwrite existing top-level column if there is name conflict when extracting
            fields from the array column during the explosion.

    Returns: a new dataframe with the specified column `col_to_explode` exploded.

    """

    # TODO: split function into two sub functions for `is_struct_array_to_explode` True/False

    # region STEP1: determine if the explosion target is a struct array
    # and determine `select_struct_fieldnames` if it is not specified
    col_to_explode_name = get_colname(df, col_to_explode)
    col_to_explode_type = get_coltype(df, col_to_explode)
    if select_struct_field_names is None:
        select_struct_field_names = getattr(
            col_to_explode_type.elementType,
            'names', None
        )
    is_struct_array_to_explode = (select_struct_field_names is not None)
    # endregion

    # region STEP2: perform the explosion
    # we also consider column naming and name conflict or column overwrite in this step

    # if `explode_colname_or_prefix` is not provided,
    # then we synthetic one
    has_explode_prefix = bool(explode_colname_or_prefix)
    if not has_explode_prefix:
        explode_colname_or_prefix = solve_name_conflict(
            name=f'{col_to_explode_name}_item',
            existing_names=df.columns
        )

    output_explode_colname = (
        explode_colname_or_prefix
        if is_struct_array_to_explode
        # when `is_struct_array_to_explode` is False,
        # the column name for the exploded items is final
        # as explode_colname_or_prefix + explode_colname_suffix
        else add_suffix(explode_colname_or_prefix, suffix=explode_colname_suffix)
    )

    if is_struct_array_to_explode:
        if output_explode_colname in df.columns:
            raise ValueError(f"the output temporary column for the explosion "
                             f"is named {output_explode_colname}; "
                             f"however, this column already exists in the dataframe; "
                             f"try giving different arguments for 'explode_colname_or_prefix' "
                             f"or 'explode_colname_suffix'.")
    elif not overwrite_exist_column:
        if output_explode_colname in df.columns:
            raise ValueError(f"the output column for the explosion "
                             f"is named {output_explode_colname}; "
                             f"however, this column already exists in the dataframe; "
                             f"try giving different arguments for 'explode_colname_or_prefix' "
                             f"or 'explode_colname_suffix' or "
                             f"set 'overwrite_exist_column' as True.")

    if index_colname is None:
        df = df.withColumn(
            output_explode_colname,
            F.explode(col_to_explode)
        )
    else:
        if index_colname is True:
            index_colname = 'index'
        explode_index_colname = add_prefix_suffix(
            index_colname,
            prefix=(explode_colname_or_prefix if has_explode_prefix else None),
            suffix=explode_colname_suffix
        )
        if overwrite_exist_column:
            _existing_columns = [
                colname for colname in df.columns
                if colname not in (explode_index_colname, output_explode_colname)
            ]
        else:
            if explode_index_colname in df.columns:
                raise ValueError(f"the output column for the explosion index"
                                 f"is named {explode_index_colname}; "
                                 f"however, this column already exists in the dataframe; "
                                 f"try giving different arguments for 'explode_colname_or_prefix' "
                                 f"or 'explode_colname_suffix' or "
                                 f"set 'overwrite_exist_column' as True.")
            _existing_columns = df.columns

        df = df.select(
            *_existing_columns,
            F.posexplode(col_to_explode).alias(
                explode_index_colname,
                output_explode_colname
            )
        )
    # endregion

    # region STEP3: extract fields from the exploded structs as top-level columns,
    # only for when `col_to_explode` is a struct array
    if is_struct_array_to_explode:
        if output_explode_colname != explode_colname_or_prefix:
            raise ValueError("logic error; "
                             "'explode_colname' must be the same as 'explode_colname_or_prefix' "
                             "at this step when 'is_struct_array_to_explode' is True")
        for sub_colname in select_struct_field_names:
            if new_cols is None or sub_colname not in new_cols:
                if (
                        prefix_suffix_exempted_struct_field_names and
                        sub_colname in prefix_suffix_exempted_struct_field_names
                ):
                    _sub_col_name = sub_colname
                else:
                    _sub_col_name = add_prefix_suffix(
                        sub_colname,
                        prefix=(output_explode_colname if has_explode_prefix else None),
                        suffix=explode_colname_suffix
                    )

                if (not overwrite_exist_column) and (_sub_col_name in df.columns):
                    raise ValueError(
                        f"column '{_sub_col_name}' already exists; consider using "
                        f"try giving different arguments for 'explode_colname_or_prefix' "
                        f"or 'explode_colname_suffix' or "
                        f"set 'overwrite_exist_column' as True."
                    )
                df = df.withColumn(
                    _sub_col_name,
                    F.col('{}.{}'.format(output_explode_colname, sub_colname))
                )

    # endregion

    # region STEP4: a convenience operation to add extra columns right after the explosion;
    # note we performed name conflict checking for these `new_cols` in STEP3.
    if new_cols is not None:
        for key, col in new_cols.items():
            if isinstance(col, str):
                df = df.withColumn(key, F.col(col))
            else:
                df = df.withColumn(key, col)
    # endregion

    # region STEP5: clean up
    if is_struct_array_to_explode:
        # when we explode a struct array,
        # a temporary column of name `explode_prefix` was created to save the structs,
        # and now we drop it
        df = df.drop(output_explode_colname)
    if is_struct_array_to_explode or (col_to_explode_name != output_explode_colname):
        # drops the original `col_to_explode`,
        # except for the case when `is_struct_array_to_explode` is False
        # and `col_to_explode` is the same as `explode_prefix`
        df = df.drop(col_to_explode)
    # endregion

    return df


# endregion

# region folding

def fold(
        df,
        group_cols,
        fold_colname,
        cols_to_fold=None,
        other_agg_cols=None,
        fold_cols_prefix_to_remove=None,
        fold_cols_suffix_to_remove=None,
        keep_top_level_cols: bool = False,
        flatten_for_single_cols_to_fold: bool = False,
        collect_set: bool = False
):
    if group_cols is None:
        raise ValueError("'group_cols' must be specified")

    group_cols = make_list_(group_cols)
    cols_to_fold = make_list_if_not_none_(cols_to_fold)

    _cols_to_fold = tuple(
        F.col(col_name).alias(
            remove_prefix_suffix(
                col_name, fold_cols_prefix_to_remove, fold_cols_suffix_to_remove
            )
        )
        for col_name in df.columns
        if ((col_name not in group_cols)
            and (cols_to_fold is None or col_name in cols_to_fold))
    )
    fold_col = (F.collect_set if collect_set else F.collect_list)(
        (
            _cols_to_fold[0]
            if (flatten_for_single_cols_to_fold and len(_cols_to_fold) == 1)
            else F.struct(*_cols_to_fold)
        )
    ).alias(fold_colname)

    if other_agg_cols is None:
        other_agg_cols = [fold_col]
    else:
        other_agg_cols = make_list_(other_agg_cols) + [fold_col]

    df_folded = df.groupBy(*group_cols).agg(*other_agg_cols)
    if cols_to_fold is not None and keep_top_level_cols:
        return df.drop(*cols_to_fold).join(
            df_folded,
            group_cols
        )
    else:
        return df_folded


def fold_as_struct_by_prefix(
        df, prefix, prefix_connector='_', excluded_col_names=None, suffix=None
):
    """
    Folding each set of fields with each of the specified `prefixes` into a struct.

    For example, if the dataframe has three fields
            'customer_field1', 'customer_field2', 'customer_field3',
            'global_field1', 'global_field2', 'global_field3',
        then calling `fold_as_struct_by_prefix(df, [customer, global], '_', None)`
        will create two fields 'customer' and 'global' like
        `{ "customer": { "field1": ..., "field2": ..., "field3": ... }, "global": { "field1": ..., "field2": ..., "field3": ... }}`. # noqa: E501

    You can make hte folding only effective for fields ending with the specified `suffix`.
    For example, `fold_as_struct_by_prefix(df, [customer, global], '_', 'first')`
        will only fold columns whose names are of pattern "customer_xxx_first", "global_xxx_first".

    """
    if isinstance(excluded_col_names, str):
        excluded_col_names = [excluded_col_names]

    if suffix is not None and not suffix.startswith(prefix_connector):
        suffix = prefix_connector + suffix

    for _prefix in make_list_(prefix):
        __prefix = _prefix
        if _prefix.endswith(prefix_connector):
            _prefix = _prefix[:-1]
        else:
            __prefix = _prefix + prefix_connector

        if suffix is None:
            cols = [
                col_name
                for col_name in df.columns
                if col_name.startswith(__prefix)
                   and (
                           excluded_col_names is None or col_name not in excluded_col_names
                   )  # noqa: E126,E501
            ]
            df = df.withColumn(
                _prefix,
                F.struct(*(F.col(col_name).alias(col_name[len(__prefix):]) for col_name in cols)),
            ).drop(*cols)
        else:
            cols = [
                col_name
                for col_name in df.columns
                if (
                        col_name.startswith(__prefix)
                        and (  # noqa: E126
                                excluded_col_names is None
                                or col_name not in excluded_col_names  # noqa: E126
                        )
                        and col_name.endswith(suffix)
                )
            ]
            df = df.withColumn(
                _prefix + suffix,
                F.struct(
                    *(
                        F.col(col_name).alias(col_name[len(__prefix): -len(suffix)])
                        for col_name in cols
                    )
                ),
            ).drop(*cols)
    return df


# endregion

def _solve_selection(
        df: DataFrame,
        *selection: Union[str, Column, Mapping, list, tuple]
) -> List[Column]:
    """
    Transforms a selection of columns into a list of Column objects.

    This function handles multiple types of input for selection, including column names (strings),
    Column objects, lists or tuples of these types, and dictionaries mapping from old column names to new names.

    This function is used internally by the 'select' function and not directly.

    Args:
        df: The DataFrame from which to select the columns.
        selection: Columns to select.

    Returns:
        A list of Column objects ready for selection.

    Raises:
        ValueError: If a dictionary is provided and the key is not a string.

    """
    out_selection = []

    for _selection in selection:
        if isinstance(_selection, str):
            out_selection.append(F.col(_selection))
        elif isinstance(_selection, Column):
            out_selection.append(_selection)
        elif isinstance(_selection, (list, tuple)):
            out_selection.extend(_solve_selection(df, *_selection))
        elif isinstance(_selection, Mapping):
            for src, trg in _selection.items():
                if not isinstance(src, str):
                    raise ValueError(f"column name must be a string; got {src}")

                if trg is None:
                    trg = F.lit(None).alias(src)
                elif isinstance(trg, str) and has_col(df, trg):
                    if src == trg:
                        trg = F.col(trg)
                    else:
                        trg = F.col(trg).alias(src)
                elif isinstance(trg, Column):
                    trg = trg.alias(src)
                else:
                    trg = F.lit(trg).alias(src)

                out_selection.append(trg)
    return out_selection


def select(
        df: DataFrame,
        *selection: Union[str, Column, Mapping, list, tuple],
) -> DataFrame:
    """
    Selects specified columns from the DataFrame, providing flexibility in renaming or aliasing columns.

    Args:
        df: The DataFrame from which to select the columns.
        selection: Columns to select. It accepts multiple arguments and each one can be a string (column name),
            a Column object, a list or tuple of these types, or a dictionary mapping from old column names to new names.

    Returns:
        The DataFrame with the selected columns.

    Examples:
    >>> from pyspark.sql import SparkSession, functions as F
    >>> spark = SparkSession.builder.getOrCreate()
    >>> data = [("John", "Doe", 30), ("Jane", "Doe", 25)]
    >>> df = spark.createDataFrame(data, ["FirstName", "LastName", "Age"])
    >>> select(df, "FirstName", "Age").show()
    +---------+---+
    |FirstName|Age|
    +---------+---+
    |     John| 30|
    |     Jane| 25|
    +---------+---+

    >>> select(df, {"FirstName": "Name", "Age": None}).show()
    +----+----+
    |Name|Age |
    +----+----+
    |John|null|
    |Jane|null|
    +----+----+
    """
    return df.select(*_solve_selection(df, *selection))


# endregion

# region new data

def with_columns(df: DataFrame, *name_val_map, overwrite_existing_cols=True):
    _rename = {}
    for src, trg in solve_key_value_pairs(name_val_map):
        if not isinstance(src, str):
            raise ValueError(f"column name must be a string; got {src}")

        is_existing_col = (src in df.columns)

        if (
                (not (isinstance(trg, str) and src == trg)) and
                (overwrite_existing_cols or (not is_existing_col))
        ):
            if trg is None:
                trg = F.lit(None)
            elif isinstance(trg, str) and has_col(df, trg):
                trg = F.col(trg)
            elif not isinstance(trg, Column):
                trg = F.lit(trg)

            if is_existing_col:
                # ! we need this to prevent a bug in Spark
                # ! that silently corrupts the data
                # ! when the column name already exists
                _src = solve_name_conflict(
                    name=get_internal_colname(src),
                    existing_names=df.columns,
                )
                _rename[_src] = src
            else:
                _src = src

            df = df.withColumn(_src, trg)

    if _rename:
        df = rename(
            df.drop(*_rename.values()),
            _rename
        )
    return df


# endregion

# region normalization

def uniquefy_columns(
        df: DataFrame,
        id_colnames: List[str],
        to_uniquefy_colnames: Iterable[str],
        auxiliary_cols: Mapping[str, Column],
        group_colnames: List[str] = None,
        order_metric_cols: Mapping[str, Column] = None
):
    """
    Suppose the combination of the values of multiple columns should uniquely identify each record,
    but for some reason their values are not unique, then this function chooses one combination to
    make the id columns unique.

    Args:
        df: the dataframe.
        id_colnames: the names of the id columns.
        to_uniquefy_colnames: a subset of `id_colnames`, which are names of the columns
            to make distinctive.
        auxiliary_cols: adding additional columns before grouping.
        group_colnames: grouping the columns of `id_colnames` by these specified columns; can use
            columns from `auxiliary_cols`; if this argument is not specified, then we use columns
            in `id_colnames` but not in `to_uniquefy_colnames` plus the columns specified by
            `auxiliary_cols` as the group columns.
        order_metric_cols: ordering by these columns within each group and select the top values
            as the chosen values to make columns of `id_colnames` unique; if this argument is
            not specified, then we randomly choose one of the values in each group.

    Returns: the input dataframe with the values of columns of names `id_colnames` made unique.

    """
    from boba_python_utils.spark_utils.aggregation import one_from_each_group
    from boba_python_utils.spark_utils.join_and_filter import join_on_columns

    if not group_colnames:
        group_colnames = [*(set(id_colnames) - set(to_uniquefy_colnames)), *auxiliary_cols]

    if not order_metric_cols:
        order_metric_cols = [F.rand()]

    # region STEP1: selects normalization dependent columns, make distinct and adds auxiliary columns
    df_entity_normalization = with_columns(
        df.select(
            *id_colnames
        ).distinct(),
        auxiliary_cols
    )
    # endregion

    # region STEP2: group by normalization keys, and finds need normalization, and then ungroup
    _KEY_TMP = get_internal_colname('tmp')
    df_entity_normalization = explode_as_flat_columns(
        fold(
            df_entity_normalization,
            group_cols=group_colnames,
            fold_colname=_KEY_TMP
        ).where(
            F.size(_KEY_TMP) > 1
        ),
        col_to_explode=_KEY_TMP
    )
    # endregion

    # region STEP3: choose the normalized form a list of forms
    _KEY_TMP = [get_internal_colname(x) for x in to_uniquefy_colnames]
    df_entity_normalization = join_on_columns(
        df_entity_normalization,
        rename(
            one_from_each_group(
                with_columns(df_entity_normalization, order_metric_cols),
                group_cols=group_colnames,
                order_cols=[F.col(x).desc() for x in order_metric_cols]
            ).drop(*order_metric_cols),
            dict(zip(to_uniquefy_colnames, _KEY_TMP))
        ),
        group_colnames
    ).where(
        or_(
            (F.col(x) != F.col(y))
            for x, y in zip(to_uniquefy_colnames, _KEY_TMP)
        )
    ).drop(*auxiliary_cols)
    # endregion

    # region STEP4: join back and normalize the input dataframe
    return with_columns(
        df.join(
            df_entity_normalization,
            id_colnames,
            how='left'
        ),
        {
            x:
                F.when(
                    F.col(y).isNotNull(), F.col(y)
                ).otherwise(
                    F.col(x)
                )
            for x, y in zip(to_uniquefy_colnames, _KEY_TMP)
        }
    )
    # endregion

# endregion
