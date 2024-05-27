# mypy: ignore-errors

from enum import Enum
from typing import Any, Iterable, Mapping, Union

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql.types import BooleanType, DoubleType, FloatType, IntegerType, LongType, StringType

from boba_python_utils.common_utils import iter_, solve_key_value_pairs
from boba_python_utils.spark_utils.typing import NameOrColumn
from boba_python_utils.spark_utils.spark_functions.common import and_, col__, or_

from boba_python_utils.string_utils import solve_compare_option

DIRECTIVE_COND_NOT_APPLIED = "n/a"
DIRECTIVE_NULL = "null"
DIRECTIVE_COND_NEGATION = "~!"
DIRECTIVE_COND_ALLOWS_NULL = "?"
DIRECTIVE_NON_COLNAME = "#"

COND_EXPRESSION_OR_VALUE = Union[str, Any]
COND_TYPE = Union[
    NameOrColumn,  # a column itself as condition
    Mapping[
        str,
        Union[
            COND_EXPRESSION_OR_VALUE,  # can be as single expression/value
            Iterable[
                # can be multiple expressions/values,
                # with the outside itertable representing 'and' logic,
                # and the inside iterable representing 'or' logic
                Union[COND_EXPRESSION_OR_VALUE, Iterable[COND_EXPRESSION_OR_VALUE]]
            ],
            NameOrColumn,
        ],
    ],  # column name and its condition object
]

CATEGORIZED_COND_TYPE = Mapping[str, COND_TYPE]


class CategorizedCondParseOptions(int, Enum):
    CategoryKeysOutside = 0
    ConditionKeysOutside = 1
    MixedPrioritizeCategoryKeysOutside = 2
    MixedPrioritizeConditionKeysOutside = 3


def _parse_categorized_cond(
    k: str, cond: COND_TYPE, df: DataFrame, skip_cond_str: str, options: CategorizedCondParseOptions
):
    """
    Parse a categorized condition based on the provided options.

    Args:
        k: Column name.
        cond: Condition object.
        df: DataFrame to which the condition will be applied.
        skip_cond_str: String representing a condition to be skipped.
        options: Parsing options for categorized conditions.

    Example:

        >>> from pyspark.sql import SparkSession
        >>> # Create a Spark session
        ... spark = SparkSession.builder \
        ...     .appName("Categorized Conditions Example") \
        ...     .getOrCreate()
        >>> # Create a sample DataFrame
        ... data = [
        ...     ("apple", "red", 8, 3),
        ...     ("orange", "orange", 6, 5),
        ...     ("apple", "green", 12, 2),
        ...     ("banana", "yellow", 3, 1),
        ... ]
        >>> columns = ["fruit", "color", "price", "size"]
        >>> df = spark.createDataFrame(data, columns)
        >>> # Define the categorized condition
        >>> cond =  {
        ...         "apple": {"price": '<10'},
        ...         "orange": {"price": '>5'},
        ... }
        >>> # Use the _parse_categorized_cond function
        ... combined_condition = _parse_categorized_cond(
        ...     k="fruit",
        ...     cond=cond,
        ...     df=df,
        ...     skip_cond_str='n/a',
        ...     options=CategorizedCondParseOptions.CategoryKeysOutside
        ... )
    """
    if options == CategorizedCondParseOptions.CategoryKeysOutside:
        return solve_single_categorized_cond(
            category_key_colname=k,
            category_cond_map=cond,
            df=df,
            skip_cond_str=skip_cond_str,
            categorized_cond_parse_options=options,
        )
    elif options == CategorizedCondParseOptions.ConditionKeysOutside:
        return solve_categorized_cond(
            category_key_cond_map=cond,
            cond_key_colname=k,
            df=df,
            skip_cond_str=skip_cond_str,
            categorized_cond_parse_options=options,
        )
    if options == CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside:
        try:
            return solve_single_categorized_cond(
                category_key_colname=k,
                category_cond_map=cond,
                df=df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=options,
            )
        except:
            return solve_categorized_cond(
                category_key_cond_map=cond,
                cond_key_colname=k,
                df=df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=options,
            )
    elif options == CategorizedCondParseOptions.MixedPrioritizeConditionKeysOutside:
        try:
            return solve_categorized_cond(
                category_key_cond_map=cond,
                cond_key_colname=k,
                df=df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=options,
            )
        except:
            return solve_single_categorized_cond(
                category_key_colname=k,
                category_cond_map=cond,
                df=df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=options,
            )
    else:
        raise ValueError(f"invalid argument 'options'; got {options}")


def _kv_to_cond(
    k: str,
    v: Any,
    df: DataFrame = None,
    categorized_cond_parse_options: CategorizedCondParseOptions = CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside,
    skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
    _solve_name_for_possibly_exploded_column: bool = True,
) -> Column:
    """
    Converts a key/value pair to a Boolean Spark column, which can be used as a condition in the `where`
    function. The key represents a column name and the value represents the condition to be applied.

    Args:
        k: The key, which represents a column name.
        v: The value, which represents the condition to be applied.
            For string based condition, :func:`string_compare` is used for quick and flexible
            condition specification.
        df: An optional DataFrame for schema-dependent parsing.
        categorized_cond_parse_options: An enumeration specifying
            how to parse categorized conditions. Defaults to MixedPrioritizeCategoryKeysOutside.
        skip_cond_str: A string used to represent a condition to be skipped. Defaults to
            DIRECTIVE_COND_NOT_APPLIED.
        _solve_name_for_possibly_exploded_column: A flag to indicate if column names
            should be resolved for exploded columns.

    Returns: A boolean Spark column representing the condition.

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.master("local[1]").appName("example").getOrCreate()
        >>> data = [(1, "apple"), (2, "orange"), (3, "banana"), (4, "apple")]
        >>> df = spark.createDataFrame(data, ["id", "fruit"])
        >>> cond = _kv_to_cond("fruit", "apple", df=df)
        >>> df.where(cond).show()
        +---+-----+
        | id|fruit|
        +---+-----+
        |  1|apple|
        +---+-----+
        |  4|apple|
        +---+-----+

        >>> cond = _kv_to_cond("fruit", "n/a", df=df)
        >>> df.where(cond).show()
        +---+------+
        | id| fruit|
        +---+------+
        |  1| apple|
        |  2|orange|
        |  3|banana|
        |  4| apple|
        +---+------+

        # "*an" means the string must contain a substring "an"
        >>> cond = _kv_to_cond("fruit", "*an", df=df)
        >>> df.where(cond).show()
        +---+------+
        | id| fruit|
        +---+------+
        |  2|orange|
        |  3|banana|
        +---+------+

        # The 'id' column is an integer type, and '<3' will parsed as "F.col('id')<3"
        >>> cond = _kv_to_cond("id", "<3", df=df)
        >>> df.where(cond).show()
        +---+------+
        | id| fruit|
        +---+------+
        |  1| apple|
        |  2|orange|
        +---+------+

        # Can specify multiple conditions for 'id' column.
        >>> cond = _kv_to_cond("id", [">1", "<3"], df=df)
        >>> df.where(cond).show()
        +---+------+
        | id| fruit|
        +---+------+
        |  2|orange|
        +---+------+
    """

    k_option, k = solve_compare_option(
        k,
        negation_indicators=DIRECTIVE_COND_NEGATION,
        ignore_null_indicator=DIRECTIVE_COND_ALLOWS_NULL,
        other_option_indicators=DIRECTIVE_NON_COLNAME,
        option_at_start=True,
        option_at_end=True,
        return_none_if_no_option_available=True,
    )

    is_k_non_colname = (
        k_option and k_option.other_options and DIRECTIVE_NON_COLNAME in k_option.other_options
    )
    if not is_k_non_colname and _solve_name_for_possibly_exploded_column and df is not None:
        from boba_python_utils.spark_utils.common import _solve_name_for_exploded_column
        k = _solve_name_for_exploded_column(df, k)

    colk = col__(k)

    if df is not None:
        # if `df` is supplied, and `v` is str, then we supports schema-dependent parsing;
        from boba_python_utils.spark_utils.common import get_coltype
        colk_type = get_coltype(df, colk)
    else:
        colk_type = None

    def solve_single_cond(_cond):
        _cond_option = None
        if is_k_non_colname:
            # this is the case when the condition key `k` starts with '#';
            # in this case the condition key is like a comment and has no practical effect,
            # and the value `v` represents the condition
            _cond_col = col__(_cond)
        else:
            if _cond is None:
                _cond_col = colk.isNull()
            elif isinstance(_cond, Mapping):
                # categorized conditions
                _cond_col = _parse_categorized_cond(
                    k=k,
                    cond=_cond,
                    df=df,
                    skip_cond_str=skip_cond_str,
                    options=categorized_cond_parse_options,
                )
            elif isinstance(_cond, set):
                # NOTE when the condition tuple/list/set contains None,
                # then it means we allow the column value be null,
                # and so we add `F.col(k).isNull()` before negation
                if None in _cond:
                    allows_null = True
                    _cond = set(_v for _v in _cond if _v is not None)
                else:
                    allows_null = False
                    _cond = set(_cond)
                if len(_cond) == 1:
                    _cond_col = colk == tuple(_cond)[0]
                else:
                    _cond_col = colk.isin(_cond)
                if allows_null:
                    _cond_col = colk.isNull() | _cond_col
            elif callable(_cond):
                _cond_col = F.udf(_cond, returnType=BooleanType())(colk)
            else:
                if isinstance(_cond, str):
                    _cond_option, _cond = solve_compare_option(
                        _cond,
                        negation_indicators=DIRECTIVE_COND_NEGATION,
                        ignore_null_indicator=DIRECTIVE_COND_ALLOWS_NULL,
                        other_option_indicators=DIRECTIVE_NON_COLNAME,
                        lower_lexical_order_indicator=None,
                        higher_lexical_order_indicator=None,
                        option_at_start=True,
                        option_at_end=True,
                        return_none_if_no_option_available=True,
                    )
                else:
                    _cond_option = None

                if _cond == DIRECTIVE_NULL:
                    if _cond_option is not None and _cond_option.negation:
                        _cond_col = colk.isNotNull()
                    else:
                        _cond_col = colk.isNull()
                elif _cond_option is None:
                    if (
                        colk_type is not None
                        and isinstance(_cond, str)
                        and not isinstance(colk_type, StringType)
                    ):
                        try:
                            _cond_col = eval("colk" + _cond, {"F": F, "colk": colk})
                        except:
                            if isinstance(colk_type, (FloatType, DoubleType)):
                                _cond_col = colk == float(_cond)
                            elif isinstance(colk_type, (IntegerType, LongType)):
                                _cond_col = colk == int(_cond)
                            elif isinstance(colk_type, BooleanType):
                                _cond = _cond.lower()
                                if _cond == "true":
                                    _cond_col = colk
                                elif _cond == "false":
                                    _cond_col = ~colk
                                else:
                                    raise ValueError(
                                        f"Column {k} is of boolean type; "
                                        f"its condition can only be 'true' or 'false'"
                                    )
                            else:
                                raise ValueError(
                                    f"unable to parse condition '{_cond}' " f"for column '{colk}'"
                                )
                    else:
                        _cond_col = colk == _cond
                else:
                    from boba_python_utils.spark_utils.spark_functions.string_functions import string_compare
                    _cond_col = string_compare(col=colk, pattern=_cond, option=_cond_option)

        return _cond_col

    cond_col = and_(
        F.lit(True)
        if _v == skip_cond_str
        else or_(
            (F.lit(False) if __v == skip_cond_str else solve_single_cond(__v))
            for __v in iter_(_v, iter_none=True)
        )
        for _v in iter_(v, iter_none=True)
    )
    if k_option is not None:
        if k_option.negation:
            cond_col = ~cond_col
        if k_option.ignore_null:
            cond_col = cond_col.isNull() | cond_col
    return cond_col


def mapping_to_cond_all(
    d: Mapping,
    skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
    df: DataFrame = None,
    categorized_cond_parse_options: CategorizedCondParseOptions = CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside,
) -> Column:
    """
    Creates a Spark Column representing a condition that is True when all of the conditions
    in the input dictionary `d` is True. The conditions are specified as key/value pairs,
    where the key represents a column name and the value is the condition that the column
    must satisfy. Skips conditions with values equal to the `skip_cond_str`.

    Args:
        d: A mapping of column names to condition values.
        skip_cond_str: A string indicating that a condition should be skipped.
        df: An optional DataFrame for schema-dependent parsing.
        categorized_cond_parse_options: Options for parsing categorized conditions.

    Returns:
        A Spark Column representing the combined conditions.

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.master("local").appName("mapping_to_cond_all").getOrCreate()
        >>> from pyspark.sql import Row
        >>> df = spark.createDataFrame([Row(id=1, fruit="apple"), Row(id=2, fruit="orange"), Row(id=3, fruit="banana")])

        >>> cond = mapping_to_cond_all({"fruit": "*an", 'id': '<3'}, df=df)
        >>> df.where(cond).show()
        +---+------+
        | id| fruit|
        +---+------+
        |  2|orange|
        +---+------+

        >>> cond = mapping_to_cond_all({"fruit": "*an", 'id': 'n/a'}, df=df)
        >>> df.where(cond).show()
        +---+------+
        | id| fruit|
        +---+------+
        |  2|orange|
        |  3|banana|
        +---+------+
    """
    cond = None
    for k, v in d.items():
        if isinstance(v, str) and v == skip_cond_str:
            continue
        if cond is None:
            cond = _kv_to_cond(
                k,
                v,
                df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=categorized_cond_parse_options,
            )
        else:
            cond = cond & _kv_to_cond(
                k,
                v,
                df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=categorized_cond_parse_options,
            )
    return cond


def mapping_to_cond_any(
    d: Mapping,
    skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
    df: DataFrame = None,
    categorized_cond_parse_options: CategorizedCondParseOptions = CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside,
) -> Column:
    """
    Creates a Spark Column representing a condition that is True when any of the conditions
    in the input dictionary `d` is True. The conditions are specified as key/value pairs,
    where the key represents a column name and the value is the condition that the column
    must satisfy. Skips conditions with values equal to the `skip_cond_str`.

    Args:
        d: A dictionary containing the key/value pairs representing the conditions.
        skip_cond_str: A string value indicating that the condition should be skipped.
        df: An optional DataFrame used to infer column types when necessary.
        categorized_cond_parse_options: An optional parameter to control how categorized
            conditions are parsed.

    Returns:
        A Spark Column representing the combined condition.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import Row

        >>> spark = SparkSession.builder.master("local").appName("mapping_to_cond_any_example").getOrCreate()
        >>> df = spark.createDataFrame([Row(id=1, fruit="apple"), Row(id=2, fruit="orange"), Row(id=3, fruit="banana")])

        >>> cond = mapping_to_cond_any({"id": "<3", "fruit": "banana"}, df=df)
        >>> result_df = df.where(cond)
        >>> result_df.show()
        # +---+------+
        # | id| fruit|
        # +---+------+
        # |  1| apple|
        # |  2|orange|
        # |  3|banana|
        # +---+------+

        >>> cond = mapping_to_cond_any({"id": "<3", "fruit": "n/a"}, df=df)
        >>> result_df = df.where(cond)
        >>> result_df.show()
        # +---+------+
        # | id| fruit|
        # +---+------+
        # |  1| apple|
        # |  2|orange|
        # +---+------+
    """
    cond = None
    for k, v in d.items():
        if isinstance(v, str) and v == skip_cond_str:
            continue
        if cond is None:
            cond = _kv_to_cond(
                k,
                v,
                df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=categorized_cond_parse_options,
            )
        else:
            cond = cond | _kv_to_cond(
                k,
                v,
                df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=categorized_cond_parse_options,
            )
    return cond


def solve_cond(
    cond: COND_TYPE,
    skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
    df: DataFrame = None,
    categorized_cond_parse_options: CategorizedCondParseOptions = CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside,
) -> Column:
    """
    Solves a condition for filtering a DataFrame.

    Args:
        cond: The condition to be solved, either as a single Column, or a dictionary of conditions.
        skip_cond_str: A string value indicating that the condition should be skipped.
        df: An optional DataFrame used to infer column types when necessary.
        categorized_cond_parse_options: An optional parameter to control how categorized
            conditions are parsed when cond is a Mapping.

    Returns:
        A Spark Column representing the condition.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import Row

        >>> spark = SparkSession.builder.master("local").appName("solve_cond_example").getOrCreate()
        >>> df = spark.createDataFrame([Row(id=1, fruit="apple"),
        ...                              Row(id=2, fruit="orange"),
        ...                              Row(id=3, fruit="banana")])

        >>> cond = solve_cond({"id": "<3", "fruit": "*an"}, df=df)
        >>> result_df = df.where(cond)
        >>> result_df.show()
        # +---+------+
        # | id| fruit|
        # +---+------+
        # |  2|orange|
        # +---+------+
    """
    if isinstance(cond, Mapping):
        _cond = mapping_to_cond_all(
            cond,
            df=df,
            skip_cond_str=skip_cond_str,
            categorized_cond_parse_options=categorized_cond_parse_options,
        )
    else:
        _cond = cond

    return _cond


def solve_single_categorized_cond(
    category_key_colname: str,
    category_cond_map: CATEGORIZED_COND_TYPE,
    default_cond: Union[NameOrColumn, Mapping] = None,
    cond_key_colname: str = None,
    df: DataFrame = None,
    skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
    categorized_cond_parse_options: CategorizedCondParseOptions = CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside,
) -> Column:
    """
    Solves a single categorized condition for a given category_key_col, applying the
    appropriate condition based on the category value.

    Args:
        category_key_colname: The column containing the category values.
        category_cond_map: A dictionary mapping category values to their respective conditions.
        default_cond: The default condition to apply when no category-specific condition exists.
        cond_key_colname: The column to which the conditions will be applied.
        df: An optional DataFrame used to infer column types when necessary.
        skip_cond_str: A string value indicating that the condition should be skipped.
        categorized_cond_parse_options: An optional parameter to control how categorized
            conditions are parsed.

    Returns:
        A Spark Column representing the combined condition.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql import Row

        >>> spark = SparkSession.builder.master("local").appName("solve_single_categorized_cond_example").getOrCreate()
        >>> df = spark.createDataFrame([Row(id=1, fruit="apple", category="A"),
        ...                              Row(id=2, fruit="orange", category="B"),
        ...                              Row(id=3, fruit="banana", category="A")])

        >>> cond = solve_single_categorized_cond("category", {"A": {"id": "<3"}, "B": {"id": ">1"}}, df=df)
        >>> result_df = df.where(cond)
        >>> result_df.show()
        # +---+------+--------+
        # | id| fruit|category|
        # +---+------+--------+
        # |  1| apple|       A|
        # |  2|orange|       B|
        # +---+------+--------+

        >>> cond = solve_single_categorized_cond("category", {"A": "<3", "B": ">1"}, cond_key_colname="id", df=df)
        >>> result_df = df.where(cond)
        >>> result_df.show()
        # +---+------+--------+
        # | id| fruit|category|
        # +---+------+--------+
        # |  1| apple|       A|
        # |  2|orange|       B|
        # +---+------+--------+
    """
    _category_cond_map = []
    for category, category_cond in solve_key_value_pairs(category_cond_map):
        if category == "default":
            default_cond = category_cond
        elif not (isinstance(category_cond, str) and category_cond == skip_cond_str):
            _category_cond_map.append((category, category_cond))

    out_cond = F.lit(False)
    if cond_key_colname is None:
        if default_cond is not None:
            out_cond = solve_cond(default_cond, df=df)
        for category, category_cond in _category_cond_map:
            out_cond = F.when(
                solve_cond({category_key_colname: category}, df=df),
                solve_cond(
                    category_cond,
                    df=df,
                    categorized_cond_parse_options=categorized_cond_parse_options,
                ),
            ).otherwise(out_cond)

        return out_cond
    else:
        if default_cond is not None:
            out_cond = _kv_to_cond(cond_key_colname, default_cond, df=df)
        for category, category_cond in _category_cond_map:
            out_cond = F.when(
                solve_cond({category_key_colname: category}, df=df),
                _kv_to_cond(
                    cond_key_colname,
                    category_cond,
                    df=df,
                    categorized_cond_parse_options=categorized_cond_parse_options,
                ),
            ).otherwise(out_cond)

        return out_cond


def solve_categorized_cond(
    category_key_cond_map: Mapping[str, CATEGORIZED_COND_TYPE],
    default_cond: Union[NameOrColumn, Mapping] = None,
    cond_key_colname: str = None,
    df: DataFrame = None,
    skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
    categorized_cond_parse_options: CategorizedCondParseOptions = CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside,
) -> Column:
    """
    Create a combined condition based on a category key condition map and apply it to filter a DataFrame.
    It processes the input through the :func:`solve_single_categorized_cond`
    function for each category key column name and category condition map.

    Example:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql.functions import col

        # Create a Spark session
        >>> spark = SparkSession.builder \
        ...    .appName("Categorized Conditions Example") \
        ...    .getOrCreate()

        # Create a sample DataFrame
        >>> data = [
        ...    ("apple", "red", 8, 3),
        ...    ("orange", "green", 6, 5),
        ...    ("apple", "red", 12, 2),
        ...    ("orange", "yellow", 3, 1),
        ... ]

        >>> columns = ["fruit", "color", "price", "size"]

        >>> df = spark.createDataFrame(data, columns)

        # Define the categorized conditions
        >>> category_key_cond_map = {
        ...     "fruit": {
        ...         "apple": {"price": '<10'},
        ...         "orange": {"price": '>5'},
        ...     },
        ...     "color": {
        ...         "red": {"size": '>2'},
        ...         "green": {"size": '<4'},
        ...     },
        ... }

        # Use the `solve_categorized_cond` function to create the combined condition
        >>> combined_condition = solve_categorized_cond(category_key_cond_map, df=df)

        # Apply the combined condition to filter the DataFrame
        >>> result_df = df.filter(combined_condition)

        # Show the original and filtered DataFrames
        >>> print("Original DataFrame:")
        >>> df.show()
        +------+-------+-----+----+
        | fruit|  color|price|size|
        +------+-------+-----+----+
        | apple|    red|    8|   3|
        |orange|  green|    6|   5|
        | apple|    red|   12|   2|
        |orange| yellow|    3|   1|
        +------+-------+-----+----+

        >>> print("Filtered DataFrame:")
        >>> result_df.show()
        +------+-------+-----+----+
        | fruit|  color|price|size|
        +------+-------+-----+----+
        | apple|    red|    8|   3|
        |orange|  green|    6|   5|
        +------+-------+-----+----+
    """
    return and_(
        solve_single_categorized_cond(
            category_key_colname=category_key_colname,
            category_cond_map=category_cond_map,
            default_cond=default_cond,
            cond_key_colname=cond_key_colname,
            df=df,
            skip_cond_str=skip_cond_str,
            categorized_cond_parse_options=categorized_cond_parse_options,
        )
        for category_key_colname, category_cond_map in solve_key_value_pairs(category_key_cond_map)
    )
