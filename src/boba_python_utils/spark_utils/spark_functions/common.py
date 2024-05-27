from typing import Union

from pyspark.sql import Column, functions as F
from pyspark.sql.types import StringType

from boba_python_utils.common_utils.typing_helper import solve_nested_singleton_tuple_list
from boba_python_utils.string_utils import add_prefix_suffix


def col_(x: Union[str, Column]) -> Column:
    """
    Returns a Column object given a string as the column name,
    or returns the Column object itself if the input is itself a Column object.
    """
    return F.col(x) if isinstance(x, str) else (x if isinstance(x, Column) else F.lit(x))


def col__(x: Union[str, Column]) -> Column:
    """
    Returns a Column object given a string as the column name,
    or a string that can be parsed as a column.
    """
    if isinstance(x, str):
        try:
            import boba_python_utils.spark_utils.spark_functions as F
            return col_(eval(x, {'F': F}))
        except:
            import pyspark.sql.functions as F
            return F.col(x)
    else:
        import pyspark.sql.functions as F
        return (x if isinstance(x, Column) else F.lit(x))


def solve_column(_col, col_name_prefix=None, col_name_suffix=None, col_name_sep='_') -> Column:
    if isinstance(_col, Column):
        return _col
    elif isinstance(_col, str):
        return F.col(add_prefix_suffix(
            s=_col,
            prefix=col_name_prefix,
            suffix=col_name_suffix,
            sep=col_name_sep
        ))
    else:
        return F.col(_col)


def solve_column_alias(col: Column, alias: str) -> Column:
    if isinstance(col, str):
        if not alias:
            col = F.col(col)
        else:
            col = F.col(col).alias(alias)
    elif alias:
        col = col.alias(alias)
    return col


def to_str(
        *cols: Union[Column, str],
        concat: str = '|'
) -> Column:
    """
    Coverts one or more columns to a string column
    Args:
        *cols: the columns to convert to a string column.
        concat: when `cols` have multiple columns,
            and if this argument is not specified, or is specified as 'json',
            then the columns will be represented by a string of the json format.
            Otherwise, the columns will be concatenated by this argument.

    Returns: one or more columns converted to a single string column.

    """
    cols = solve_nested_singleton_tuple_list(cols, atom_types=(str, Column))
    if len(cols) == 1:
        return col_(cols[0]).cast(StringType())
    elif (concat is None) or concat == 'json':
        return F.to_json(F.struct(*cols))
    else:
        return F.concat_ws(concat, *(col_(_col).cast(StringType()) for _col in cols))


def and_(*conds) -> Column:
    """
    Returns True if all `conds` hold True.

    None is returned if `conds` is empty.
    """
    conds = solve_nested_singleton_tuple_list(conds, atom_types=(str, Column))
    out_cond = None
    for cond in conds:
        if cond is not None:
            if isinstance(cond, str):
                cond = F.col(cond)
            if out_cond is None:
                out_cond = cond
            else:
                out_cond = (out_cond & cond)

    return out_cond


def or_(*conds) -> Column:
    """
    Returns True if any of `conds` holds True.

    None is returned if `conds` is empty.
    """
    conds = solve_nested_singleton_tuple_list(conds, atom_types=(str, Column))
    out_cond = None
    for cond in conds:
        if cond is not None:
            if isinstance(cond, str):
                cond = F.col(cond)
            if out_cond is None:
                out_cond = cond
            else:
                out_cond = (out_cond | cond)

    return out_cond


def is_not_null_or_false(*cols) -> Column:
    """
    Checks if all the specified columns are neither null nor False.
    """
    cols = solve_nested_singleton_tuple_list(cols, atom_types=(str, Column))
    return and_(
        *(
            (
                    col_(_col).isNotNull() &
                    (col_(_col) != False)
            )
            for _col in cols
            if (
                _col is not None and
                not (isinstance(_col, str) and not _col)
        )
        )
    )


def is_not_null(*cols) -> Column:
    """
    Checks if all the specified columns are not null.
    """
    cols = solve_nested_singleton_tuple_list(cols, atom_types=(str, Column))
    return and_(
        *(
            (col_(_col).isNotNull())
            for _col in cols
            if (
                _col is not None and
                not (isinstance(_col, str) and not _col)
        )
        )
    )


def is_null(*cols) -> Column:
    """
    Checks if all the specified columns are null.
    """
    cols = solve_nested_singleton_tuple_list(cols, atom_types=(str, Column))
    return and_(
        *(
            (col_(_col).isNull())
            for _col in cols
            if (
                _col is not None and
                not (isinstance(_col, str) and not _col)
        )
        )
    )


def first_non_null(*cols) -> Column:
    """
    Returns the first non-null column from a list of given columns.
    """
    if len(cols) == 1:
        if isinstance(cols[0], (list, tuple)):
            return first_non_null(*cols[0])
        else:
            return col_(cols[0])
    else:
        return F.when(col_(cols[0]).isNotNull(), col_(cols[0])).otherwise(
            first_non_null(*cols[1:])
        )


def set_value_for_null(col, value) -> Column:
    col = col_(col)
    return F.when(col.isNull(), F.lit(value)).otherwise(col)


def set_null_if(col, cond):
    col = col_(col)
    if isinstance(cond, Column):
        return F.when(cond, F.lit(None)).otherwise(col)
    elif isinstance(cond, set):
        return F.when((col.isin(cond)), F.lit(None)).otherwise(col)
    else:
        return F.when((col == cond), F.lit(None)).otherwise(col)
