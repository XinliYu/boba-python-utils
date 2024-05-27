from boba_python_utils.spark_utils.spark_functions.algorithms import *
from boba_python_utils.spark_utils.spark_functions.array_functions import *
from boba_python_utils.spark_utils.spark_functions.common import *  # noqa: E401
from boba_python_utils.spark_utils.spark_functions.cond_solution import *  # noqa: E401
from boba_python_utils.spark_utils.spark_functions.misc import *  # noqa: E401
from boba_python_utils.spark_utils.spark_functions.nlp_functions import *  # noqa: E401
from boba_python_utils.spark_utils.spark_functions.string_functions import *  # noqa: E401
from boba_python_utils.spark_utils.spark_functions.selection import *  # noqa: E401
from boba_python_utils.spark_utils.typing import NameOrColumn
from boba_python_utils.string_utils.tokenization import tokenize
from functools import reduce as _reduce
from typing import List

from pyspark.sql.functions import *  # noqa: 401
from pyspark.sql.functions import *  # noqa: F401,F403

import boba_python_utils.general_utils.strex as strex
from boba_python_utils.spark_utils.common import (
    _solve_name_for_exploded_column, solve_names_and_columns
)

DIRECTIVE_COND_NOT_APPLIED = 'n/a'
DIRECTIVE_NULL = 'null'
DIRECTIVE_COND_NEGATION = '~!'
DIRECTIVE_COND_ALLOWS_NULL = '?'
DIRECTIVE_NON_COLNAME = '#'


def solve_column(_col, col_name_prefix=None, col_name_suffix=None, col_name_sep='_') -> Column:
    if isinstance(_col, Column):
        return _col
    elif isinstance(_col, str):
        return F.col(strex.add_prefix_suffix(
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


def get_all_leaf_column_names(df: DataFrame, *col_names, sort_nested_column_names=True):
    """
    Retrieves names of all leaf columns at the bottom level of the dataframe schema.
    A leaf column name consists names of all columns along the path
        from the top-level column to this leaf column.

    For example, suppose the data is
    {
        "c": 1,
        "a": {
            "a2": 1
            "a1": 2
        },
        "b": {
            "b1": {
                "b11": 1,
                "b12": 2
            },
            "b2": 3
        }
    }
    Then this method returns ["c", "a.a1", "a.a2", "b.b1.b11", "b.b1.b12", "b.b2"],
        if we pass ["c", "a", "b"] to parameter `col_names`.

    If `sort_nested_column_names` set False,
        then this method returns ["c", "a.a2", "a.a1", "b.b1.b11", "b.b1.b12", "b.b2"],
        where the order of nested columns ["a.a2", "a.a1"] are sorted alphabetically;

    The order of the input `col_names` will always be preserved in the returned columns.
        For example, if we pass ["b", "a"] to parameter `col_names`,
        then this method returns  ["b.b1.b11", "b.b1.b12", "b.b2", "a.a1", "a.a2"]
        (suppose `sort_nested_column_names` is True).

    We can also specify a non-top level ["b.b1"] as the ``col_names`,
        and this method will start searching leaf node starting from "b.b1",
        and return ["b.b1.b11", "b.b1.b12"].

    Args:
        df: the dataframe.
        *col_names: names of the columns where the search for the leaf columns begins.
        sort_nested_column_names: True to let the nested columns be sorted alphabetically.


    Returns: search for leaf columns whose roots are specified by `col_names`,
        and returns names of all the leaf columns.

    """
    schema = df.select(*col_names).schema
    out = []

    def _get_field_names(prefix, fields: List[StructField]):
        _out = []
        for field in fields:
            field_name = f'{prefix}.{field.name}'
            if isinstance(field.dataType, StructType):
                _out.extend(_get_field_names(field_name, field.dataType.fields))
            else:
                _out.append(field_name)
        return _out

    for field in schema.fields:
        if isinstance(field.dataType, StructType):
            nested_fields = _get_field_names(field.name, field.dataType.fields)
            if sort_nested_column_names:
                nested_fields = sorted(nested_fields)
            out.extend(nested_fields)
        else:
            out.append(field.name)
    return out


def null_on_cond(cond: Column, otherwise_col: Union[str, Column]) -> Column:
    """
    Returns a None literal if the condition is True;
    otherwise uses values from the 'otherwise_col' column.

    Args:
        cond: the condition to test.
        otherwise_col: uses values from this column when the condition `cond` is False.

    Returns: a new Column, whose value is None when `cond` is True, otherwise it uses values
        from the `otherwise_col` column.

    """
    return F.when(cond, F.lit(None)).otherwise(
        # ! 'otherwise' function only takes a string as a literal value rather than a column name
        col_(otherwise_col)
    )


def multi_when_otherwise(
        *conds: Union[Column, Tuple[Column, Column]],
        default=None
) -> Column:
    if isinstance(conds[0], tuple):
        out_cond = F.lit(False) if default is None else col_(default)
        for _when, _cond in reversed(conds):
            out_cond = F.when(_when, col_(_cond)).otherwise(out_cond)
    else:
        index = 0
        out_cond = F.lit(index) if default is None else col_(default)
        for _when in reversed(conds):
            index += 1
            out_cond = F.when(_when, F.lit(index)).otherwise(out_cond)

    return out_cond


def categorize(*conds: Column, default=None) -> Column:
    return multi_when_otherwise(*conds, default=default)


def equal(
        eq_cols: Union[Iterable[str], Iterable[Tuple[Union[str, Column], Union[str, Column]]]],
        col_suffix1: str = None,
        col_suffix2: str = None,
):
    return _reduce(
        lambda x, y: x & y,
        (
            solve_column(col_name1, col_name_suffix=col_suffix1) ==
            solve_column(col_name2, col_name_suffix=col_suffix2)
            for col_name1, col_name2 in solve_names_and_columns(eq_cols)
        ),
    )


def ratio(count_col: NameOrColumn) -> Column:
    return F.col(count_col) / F.sum(count_col)


def count_true(col: Union[str, Column]):
    """
    Counts the number of True values in a BooleanType column.
    Args:
        col: the column, must be of BooleanType.

    Returns: the number of True values in the specified column.

    """
    if isinstance(col, str):
        col = F.col(col)
    return F.sum(col.astype(IntegerType()))


def ratio_true(col: Union[str, Column]):
    """
    Counts the ratio of True values in a BooleanType column.
    Args:
        col: the column, must be of BooleanType.

    Returns: the ratio of True values in the specified column.

    """
    if isinstance(col, str):
        col = F.col(col)
    return F.avg(col.astype(IntegerType()))


def round_(col, scale):
    if scale is True:
        return F.round(col, 4)
    elif scale and scale > 0:
        return F.round(col, scale)
    else:
        return col


def is_close(
        close_cols: Union[Iterable[str], Iterable[Tuple[Union[str, Column], Union[str, Column]]]],
        close_tolerance: Union[float, Iterable[float]] = 0.0001,
        col_suffix1: str = None,
        col_suffix2: str = None,
):
    close_diff_cols = abs_diff(cols=close_cols, col_suffix1=col_suffix1, col_suffix2=col_suffix2)
    if isinstance(close_tolerance, float):
        return _reduce(
            lambda x, y: x & y,
            (close_diff_col < close_tolerance for close_diff_col in close_diff_cols),
        )
    else:
        return _reduce(
            lambda x, y: x & y,
            (
                close_diff_col < _close_tolerance
                for close_diff_col, _close_tolerance in zip(close_diff_cols, close_tolerance)
            ),
        )


def abs_diff(
        cols: Union[Iterable[str], Iterable[Tuple[Union[str, Column], Union[str, Column]]]],
        col_name_suffix1: str = None,
        col_name_suffix2: str = None,
        col_name_suffix_sep='_',
        df_for_nested_columns=None,
):
    """
    Finds the absolute difference between corresponding pairs of columns.
    The column data types in the column pairs need to support the subtraction operator ('-') .

    Args:
        cols: the columns to compare, or a list of column pairs.
        col_name_suffix1: adds this suffix to
                the end of columns names of the first column in the specified column pairs.
        col_name_suffix2: adds this suffix to
                the end of columns names of the second column in the specified column pairs.
        df_for_nested_columns: need to provide a dataframe if
            any of the column pairs to compare includes a struct type
            (i.e. if some nested columns are considered in the comparison)

    Returns: the absolute difference between the specified pairs of columns;
        this will include nested columns (if any)
        if the data frame `df_for_nested_columns` is provided.

    """
    col_names1, col_names2 = zip(*solve_names_and_columns(cols))
    if df_for_nested_columns is not None:
        col_names1 = get_all_leaf_column_names(
            df_for_nested_columns,
            *(
                strex.add_suffix(s=col_name, suffix=col_name_suffix1, sep=col_name_suffix_sep)
                for col_name in col_names1
            ),
            sort_nested_column_names=True
        )
        col_names2 = get_all_leaf_column_names(
            df_for_nested_columns,
            *(
                strex.add_suffix(s=col_name, suffix=col_name_suffix2, sep=col_name_suffix_sep)
                for col_name in col_names2
            ),
            sort_nested_column_names=True
        )

        col_names1_nested = [col_name.split('.', 1)[1] for col_name in col_names1 if '.' in col_name]
        col_names2_nested = [col_name.split('.', 1)[1] for col_name in col_names2 if '.' in col_name]

        if col_names1_nested != col_names2_nested:
            raise ValueError(
                f"nested columns must have the same names; "
                f"got {col_names1_nested} and {col_names2_nested}"
            )
    else:
        col_names1 = [
            strex.add_suffix(s=col_name, suffix=col_name_suffix1)
            for col_name in col_names1
        ]
        col_names2 = [
            strex.add_suffix(s=col_name, suffix=col_name_suffix2)
            for col_name in col_names2
        ]

    return [
        F.abs(
            solve_column(col_name1) - solve_column(col_name2)
        ).alias(
            col_name2.replace('.', '-') + '_diff'
        )
        for col_name1, col_name2 in zip(col_names1, col_names2)
    ]


def _lower(x):
    return F.lower(x)


def _remove_spaces(x):
    return F.regexp_replace(x, r'\s', '')


def _replace_method(x, replacement: Mapping):
    for k, v in replacement.items():
        x = F.regexp_replace(x, k, v)

    return x


def _sort_tokens(x, tokenizer, reverse):
    if tokenizer is None:
        return F.concat_ws(' ', F.sort_array(F.split(x, r'\s'), asc=not reverse))
    else:
        return F.concat_ws(
            ' ', F.sort_array(F.udf(partial(tokenize, tokenizer=tokenizer))(x), asc=not reverse)
        )


def multi_col_udf(
        *cols,
        func,
        return_type=None,
        results_as_separated_cols=True,
        **kwargs
):
    result_col = F.udf(
        partial(
            func,
            **kwargs
        ),
        returnType=ArrayType(return_type or StringType())
    )(*cols)

    if results_as_separated_cols:
        return [
            result_col.getItem(i)
            for i in range(
                results_as_separated_cols
                if isinstance(results_as_separated_cols, int) else len(cols)
            )
        ]
    else:
        return result_col


def num_overlap_tokens(
        col_first,
        col_second,
        sanitization_actions: Iterable[StringSanitizationOptions] = None,
        tokenizer=None,
        **kwargs
):
    if sanitization_actions:
        col_first, col_second = string_sanitize_udf(
            col_first,
            col_second,
            config=sanitization_actions,
            tokenizer=tokenizer,
            **kwargs
        )

    return F.udf(
        partial(strex.num_overlap_tokens, tokenizer=tokenizer), returnType=IntegerType()
    )(col_first, col_second)


@F.udf
def uuid4():
    return str(uuid.uuid4())


def contains_whole_word(col1, col2, col1_has_punctuations=True, white_space=' '):
    """
    Returns a boolean value indicating if 'col1' contains 'col2' as a whole word.

    Args:
        col1: will check if this column contains 'col2' as a whole word.
        col2: will check if this column is contained in 'col1' as a whole word.
        col1_has_punctuations: True to indicate 'col1' might contain punctuations;
            set this to False if we are sure 'col1' does not contain punctuations
            and save computation time.
        white_space: the white space character used in 'col1' and 'col2' to separate words.

    Returns: True if 'col1' contains 'col2' as a whole word.

    """
    if isinstance(col1, str):
        col1 = F.col(col1)
    if isinstance(col2, str):
        col2 = F.col(col2)

    if col1_has_punctuations:
        col1 = F.regexp_replace(col1, r'(\p{Punct})\s', ' $1 ')

    return (
            col1.contains(F.concat(F.lit(white_space), col2, F.lit(white_space))) |
            col1.startswith(F.concat(col2, F.lit(white_space))) |
            col1.endswith(F.concat(F.lit(white_space), col2))
    )


def replace_punctuations_and_whitespaces_by_space(col):
    """
    Replaces punctuations and whitespaces from the a string column 'col' by single spaces.

    Consecutive punctuations and whitespaces will be replaced by a single space.
    For example, "hello, world" will become "hello world".
    """
    return F.concat_ws(' ', F.split(col, r'[\p{Punct}\s]+'))


@F.udf(returnType=ArrayType(StringType()))
def remove_substrings_from_list(string_list):
    out = []
    for s1 in string_list:
        should_skip = False
        for s2 in string_list:
            if len(s1) < len(s2) and (
                    s2.startswith(s1 + ' ')
                    or s2.endswith(' ' + s1)
                    or (' ' + s1 + ' ') in s2
            ):
                should_skip = True
                break
        if not should_skip:
            out.append(s1)
    return out


# region condition resolution
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
                Union[
                    COND_EXPRESSION_OR_VALUE,
                    Iterable[COND_EXPRESSION_OR_VALUE]
                ]
            ],
            NameOrColumn
        ]
    ]  # column name and its condition object
]

CATEGORIZED_COND_TYPE = Mapping[str, COND_TYPE]


class CategorizedCondParseOptions(int, Enum):
    CategoryKeysOutside = 0
    ConditionKeysOutside = 1
    MixedPrioritizeCategoryKeysOutside = 2
    MixedPrioritizeConditionKeysOutside = 3


def _parse_categorized_cond(
        k: str,
        cond: COND_TYPE,
        df: DataFrame,
        skip_cond_str: str,
        options: CategorizedCondParseOptions
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
            categorized_cond_parse_options=options
        )
    elif options == CategorizedCondParseOptions.ConditionKeysOutside:
        return solve_categorized_cond(
            category_key_cond_map=cond,
            cond_key_colname=k,
            df=df,
            skip_cond_str=skip_cond_str,
            categorized_cond_parse_options=options
        )
    if options == CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside:
        try:
            return solve_single_categorized_cond(
                category_key_colname=k,
                category_cond_map=cond,
                df=df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=options
            )
        except:
            return solve_categorized_cond(
                category_key_cond_map=cond,
                cond_key_colname=k,
                df=df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=options
            )
    elif options == CategorizedCondParseOptions.MixedPrioritizeConditionKeysOutside:
        try:
            return solve_categorized_cond(
                category_key_cond_map=cond,
                cond_key_colname=k,
                df=df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=options
            )
        except:
            return solve_single_categorized_cond(
                category_key_colname=k,
                category_cond_map=cond,
                df=df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=options
            )
    else:
        raise ValueError(f"invalid argument 'options'; got {options}")


def _kv_to_cond(
        k: str,
        v: Any,
        df: DataFrame = None,
        categorized_cond_parse_options: CategorizedCondParseOptions =
        CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside,
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
        return_none_if_no_option_available=True
    )

    is_k_non_colname = (
            k_option and
            k_option.other_options and
            DIRECTIVE_NON_COLNAME in k_option.other_options
    )
    if not is_k_non_colname and _solve_name_for_possibly_exploded_column and df is not None:
        k = _solve_name_for_exploded_column(df, k)

    colk = col__(k)

    if df is not None:
        # if `df` is supplied, and `v` is str, then we supports schema-dependent parsing;
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
                    options=categorized_cond_parse_options
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
                    _cond_col = (colk == tuple(_cond)[0])
                else:
                    _cond_col = colk.isin(_cond)
                if allows_null:
                    _cond_col = (colk.isNull() | _cond_col)
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
                        return_none_if_no_option_available=True
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
                            and isinstance(_cond, str) and
                            not isinstance(colk_type, StringType)
                    ):
                        try:
                            _cond_col = eval("colk" + _cond, {'F': F, 'colk': colk})
                        except:
                            if isinstance(colk_type, (FloatType, DoubleType)):
                                _cond_col = (colk == float(_cond))
                            elif isinstance(colk_type, (IntegerType, LongType)):
                                _cond_col = (colk == int(_cond))
                            elif isinstance(colk_type, BooleanType):
                                _cond = _cond.lower()
                                if _cond == 'true':
                                    _cond_col = colk
                                elif _cond == 'false':
                                    _cond_col = (~colk)
                                else:
                                    raise ValueError(f"Column {k} is of boolean type; "
                                                     f"its condition can only be 'true' or 'false'")
                            else:
                                raise ValueError(f"unable to parse condition '{_cond}' "
                                                 f"for column '{colk}'")
                    else:
                        _cond_col = (colk == _cond)
                else:
                    _cond_col = string_compare(
                        col=colk,
                        pattern=_cond,
                        option=_cond_option
                    )

        return _cond_col

    cond_col = and_(
        F.lit(True) if _v == skip_cond_str
        else or_(
            (F.lit(False) if __v == skip_cond_str else solve_single_cond(__v))
            for __v in iter_(_v, iter_none=True)
        )
        for _v in iter_(v, iter_none=True)
    )
    if k_option is not None:
        if k_option.negation:
            cond_col = (~cond_col)
        if k_option.ignore_null:
            cond_col = (cond_col.isNull() | cond_col)
    return cond_col


def mapping_to_cond_all(
        d: Mapping,
        skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
        df: DataFrame = None,
        categorized_cond_parse_options: CategorizedCondParseOptions =
        CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside
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
                k, v, df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=categorized_cond_parse_options
            )
        else:
            cond = cond & _kv_to_cond(
                k, v, df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=categorized_cond_parse_options
            )
    return cond


def mapping_to_cond_any(
        d: Mapping,
        skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
        df: DataFrame = None,
        categorized_cond_parse_options: CategorizedCondParseOptions =
        CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside
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
                k, v, df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=categorized_cond_parse_options
            )
        else:
            cond = cond | _kv_to_cond(
                k, v, df,
                skip_cond_str=skip_cond_str,
                categorized_cond_parse_options=categorized_cond_parse_options
            )
    return cond


def solve_cond(
        cond: COND_TYPE,
        skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
        df: DataFrame = None,
        categorized_cond_parse_options: CategorizedCondParseOptions =
        CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside
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
            categorized_cond_parse_options=categorized_cond_parse_options
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
        categorized_cond_parse_options: CategorizedCondParseOptions =
        CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside
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
        if category == 'default':
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
                    categorized_cond_parse_options=categorized_cond_parse_options
                )
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
                    categorized_cond_parse_options=categorized_cond_parse_options
                )
            ).otherwise(out_cond)

        return out_cond


def solve_categorized_cond(
        category_key_cond_map: Mapping[str, CATEGORIZED_COND_TYPE],
        default_cond: Union[NameOrColumn, Mapping] = None,
        cond_key_colname: str = None,
        df: DataFrame = None,
        skip_cond_str: str = DIRECTIVE_COND_NOT_APPLIED,
        categorized_cond_parse_options: CategorizedCondParseOptions =
        CategorizedCondParseOptions.MixedPrioritizeCategoryKeysOutside
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
            categorized_cond_parse_options=categorized_cond_parse_options
        )
        for category_key_colname, category_cond_map
        in solve_key_value_pairs(category_key_cond_map)
    )


# endregion

# region numeric array functions


def dot_product(arr_colname1: str, arr_colname2: str):
    if isinstance(arr_colname1, Column) or isinstance(arr_colname2, Column):
        raise ValueError("must pass in string names of existing columns")
    return array_sum(
        F.transform(
            F.arrays_zip(arr_colname1, arr_colname2),
            # x is a StructType column,
            # we have to use colmn names as index,
            # and it is not possible to get the field by integer index
            lambda x: x[arr_colname1] * x[arr_colname2]
        )
    )


def euclidean_dist(arr_colname1: str, arr_colname2: str):
    if isinstance(arr_colname1, Column) or isinstance(arr_colname2, Column):
        raise ValueError("must pass in string names of existing columns")
    return array_sum(
        F.transform(
            F.arrays_zip(arr_colname1, arr_colname2),
            # x is a StructType column,
            # we have to use colmn names as index,
            # and it is not possible to get the field by integer index
            lambda x: (x[arr_colname1] - x[arr_colname2]) ** 2
        ),
        final=lambda x: x ** 0.5
    )


def euclidean_norm(arr_col):
    return F.aggregate(
        arr_col,
        F.lit(0.0),
        lambda acc, x: acc + x ** 2,
        lambda x: x ** 0.5
    )


def cosine(arr_colname1, arr_colname2):
    return dot_product(arr_colname1, arr_colname2) / (euclidean_norm(arr_colname1) * euclidean_norm(arr_colname2))


# endregion


def weighted_average(avg_col: NameOrColumn, weight_col: NameOrColumn) -> Column:
    return F.sum(col_(avg_col) * col_(weight_col)) / F.sum(weight_col)
