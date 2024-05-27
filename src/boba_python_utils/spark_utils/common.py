from enum import Enum
from functools import partial
from typing import Union, Mapping, List, Tuple, Any, Iterator, Optional, Iterable

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.functions import col
from pyspark.sql.types import (
    DataType, StructField, BooleanType, StringType, IntegerType, FloatType, StructType, ArrayType
)

from boba_python_utils.common_utils.function_helper import apply_func
from boba_python_utils.common_utils.typing_helper import solve_nested_singleton_tuple_list, make_list_
from boba_python_utils.general_utils.general import (
    solve_key_value_pairs,
)
from boba_python_utils.spark_utils.typing import NameOrColumn, AliasAndColumn, ColumnsOrAliasedColumns
from boba_python_utils.string_utils.prefix_suffix import solve_name_conflict
from boba_python_utils.string_utils.split import split_with_escape_and_quotes

"""
This script should not import any other spark utility module to avoid circular import
"""


# region common Enums
class CacheOptions(str, Enum):
    NO_CACHE = 'no_cache'
    IMMEDIATE = 'immediate'
    AUTO = 'auto'


# endregion

# region misc
def get_spark(
        log_level='ERROR',
        app_name: str = 'data',
        configs: Mapping[str, Any] = None,
        master_port=7077,
        variable_dict: Mapping = None,
) -> SparkSession:
    """
    Gets a cluster spark session object with master being "spark://0.0.0.0".

    Args:
        log_level: sets the logging level, ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE, WARN.
        app_name: specifies the application name.
        configs: spark configuration key-value pairs.
        master_port: specifies the master port for the spark session.
        variable_dict: pass in a variable dictionary that contains a SparkSession object named
        'spark';
            this method will configure this SparkSession object and returns it instead of
            creating a new one.

    Examples:
        The `variable_dict` parameter is intended for convenience in the Pyspark notebook
        environment where a SparkSession object is already created, but we just want to change
        some of its configurations.
        >>> spark = get_spark() # returns a new SparkSession object
        >>> spark = get_spark(variable_dict=globals()) # returns the SparkSession object
        already exists in the global variable scope.

    Returns: a spark session.

    """
    ss = None
    if variable_dict is not None and 'spark' in variable_dict:
        ss = variable_dict['spark']
        if isinstance(ss, SparkSession):
            ss.sparkContext.setLogLevel(log_level)
        else:
            ss = None

    if ss is None:
        app = SparkSession.builder.appName(app_name).master(f"spark://0.0.0.0:{master_port}")
        if configs:
            for k, v in solve_key_value_pairs(configs):
                app.config(k, v)
        ss = app.getOrCreate()
        ss.sparkContext.setLogLevel(log_level)
    return ss


def num_workers(spark: SparkSession) -> int:
    """
    Gets the number of available get_spark workers.
    """
    return spark._jsc.sc().getExecutorMemoryStatus().size()


def num_shuffle_partitions(spark: SparkSession) -> int:
    """
    Gets the number of available get_spark workers.
    """
    return int(spark.conf.get('spark.sql.shuffle.partitions'))


INTERNAL_USE_COL_NAME_PREFIX = '___'
INTERNAL_USE_COL_NAME_PART_SEP = '--'


def get_internal_colname(colname: str, existing_colnames=None) -> str:
    internal_colname = INTERNAL_USE_COL_NAME_PREFIX + colname
    if existing_colnames:
        internal_colname = solve_name_conflict(
            internal_colname,
            existing_names=existing_colnames
        )
    return internal_colname


def _repartition(
        df: DataFrame,
        num_partitions: int = None,
        cols=(),
        spark: SparkSession = None,
        force_repartition: bool = True
):
    rdd_num_partitions = df.rdd.getNumPartitions()
    if not num_partitions:
        num_partitions = (
            rdd_num_partitions
            if spark is None
            else num_shuffle_partitions(spark)
        )

    if len(cols) == 0 and num_partitions == rdd_num_partitions and not force_repartition:
        return df

    return df.repartition(num_partitions, *cols)


def repartition(
        df: Union[DataFrame, List[DataFrame], Tuple[DataFrame], Mapping[str, DataFrame]],
        num_partitions: int = None,
        *cols,
        spark: SparkSession = None,
        force_repartition: bool = True
):
    """
    Repartition input the dataframe(s).

    Different from `df.repartition`, the number of partitions is optional for this function.
    1) if the Spark session object is provided, we use the number of shuffle partitions;
    2) otherwise we use the current number of partitions of the dataframe.

    Examples:
        >>> repartition(df, 4800, 'col1')
        >>> repartition(df, 'col1')
        >>> repartition(df, 4800, 'col1', 'col2')
        >>> repartition(df, 'col1', 'col2')
        >>> repartition((df1, df2, df3))
        >>> repartition({'name1': df1, 'name2': df2})
    """
    return apply_func(
        func=partial(
            _repartition,
            num_partitions=num_partitions,
            cols=cols,
            spark=spark,
            force_repartition=force_repartition
        ),
        input=df
    )


def is_single_row_dataframe(df: DataFrame) -> bool:
    return len(df.head(2)) == 1


def create_empty_dataframe(
        spark: SparkSession,
        *colnames: Iterable[str]
):
    if spark is None:
        raise ValueError("Spark session object must be provided for creating a dataframe")
    return spark.createDataFrame(
        [[''] * len(colnames)],
        colnames
    ).where(F.lit(False))


# endregion

# region column information and solution

def get_colname(
        df: DataFrame, *cols: NameOrColumn
) -> Union[str, List[str]]:
    """
    Gets the column name(s) of the specified column(s).
    Needs a dataframe object to resolve the colum names.

    Args:
        df: the dataframe object to help resolve the colum names.
        *cols: the columns.

    Returns: a column name if a single column is specified,
        or a list of column names if multiple columns are specified.

    """
    cols = solve_nested_singleton_tuple_list(cols, atom_types=(str, Column))
    if len(cols) == 1:
        col = cols[0]
        if isinstance(col, str):
            return col
        else:
            return df.select(col).columns[0]
    else:
        return df.select(*cols).columns


def has_conflict_colnames(df: DataFrame, *cols: NameOrColumn) -> bool:
    """
    Checks if the columns have conflict column names.
    """
    colnames = get_colname(df, *cols)
    return len(colnames) > len(set(colnames))


def exclude_columns_of_conflict_names(
        *cols: NameOrColumn, df: Optional[DataFrame] = None
) -> List[NameOrColumn]:
    """
    Excludes columns of conflict names. If two columns have the same name,
    then the first column specified in `cols` will be kept.

    Args:
        *cols: the columns.
        df: optionally provides a dataframe to help
            resolve column names for all columns specified in `cols`;
            if the dataframe is not specified, then we are only able to identify duplicate strings
            or duplicate column objects in `cols`.

    Returns: a list of columns with the name conflict removed.

    Examples:
        >>> # in this example, we are unable to identify the two `F.col('a')` are conflict columns
        >>> exclude_columns_of_conflict_names(F.col('a'), F.col('a'), 'b', 'b', 'c')
        [Column<'a'>, Column<'a'>, 'b', 'c']

        >>> # in this example, by providing a dataframe, we are able to deduplicate the column 'a'
        >>> df = create_empty_dataframe(spark, 'a', 'b', 'c')
        >>> exclude_columns_of_conflict_names(F.col('a'), F.col('a'), 'b', 'b', 'c', df=df)
        [Column<'a'>, 'b', 'c']
    """
    _colnames = set()
    out = []
    if df is None:
        _cols = []
        cols = solve_nested_singleton_tuple_list(cols, atom_types=(str, Column))
        for _col in cols:
            if isinstance(_col, str):
                if _col not in _colnames:
                    out.append(_col)
                    _colnames.add(_col)
            elif isinstance(_col, Column):
                if not any(_col is __col for __col in _cols):
                    out.append(_col)
                    _cols.append(_col)
            else:
                raise ValueError(f"{_col} is neither a column name or a column")
    else:
        _colnames = set()
        colnames = get_colname(df, *cols)
        for colname, _col in zip(colnames, cols):
            if colname not in _colnames:
                out.append(_col)
                _colnames.add(colname)
    return out


def solve_name_and_column(
        col_spec: Union[NameOrColumn, AliasAndColumn],
        df: DataFrame = None
) -> Tuple[str, Optional[Union[Column, List[Column]]]]:
    """
    Allows a column specification being a string, or a tuple,
    or a mapping of a single key-value pair.
    Returns the column name and its corresponding column object(s).

    When the column specification is a tuple, or a mapping of one key-valeu pair,
    and the first element of the tuple/pair is a string,
    then this string is the alias for the column object(s).

    This method supports a name/alias being associated with multiple columns,
    for the case when we want to reduce multiple columns into a single column.

    Examples:
        >>> solve_name_and_column('col1')
        ('col1', Column<'col1'>)
        >>> solve_name_and_column(('col1_alias', 'col1'))
        ('col1_alias', Column<'col1'>)
        >>> solve_name_and_column({'col1_alias': 'col1'})
        ('col1_alias', Column<'col1'>)
        >>> from pyspark.sql.functions import max
        >>> solve_name_and_column({'col1_alias': max('col1')})
        ('col1_alias', Column<'max(col1)'>)
        >>> solve_name_and_column({'reduced_col': ['col1', 'col2']})
        ('reduced_col', [Column<'col1'>, Column<'col2'>])
        >>> solve_name_and_column({'reduced_col': ['col1', max('col2')]})
        ('reduced_col', [Column<'col1'>, Column<'max(col2)'>])
    """
    from boba_python_utils.spark_utils.spark_functions.common import col_
    if col_spec is None:
        return '', None
    if isinstance(col_spec, str):
        return col_spec, col(col_spec)

    if isinstance(col_spec, Column):
        if df is None:
            raise ValueError("needs a dataframe to solve the name of a column object")
        return get_colname(df, col_spec), col_spec

    if isinstance(col_spec, Mapping):
        col_spec = next(iter(col_spec.items()))

    if isinstance(col_spec, (list, tuple)) and len(col_spec) == 2:
        if isinstance(col_spec[0], str):
            if isinstance(col_spec[1], (str, Column)):
                return col_spec[0], col_(col_spec[1])
            elif (
                    isinstance(col_spec[1], (list, tuple)) and
                    all(isinstance(x, (str, Column)) for x in col_spec[1])
            ):
                return col_spec[0], list(map(col_, col_spec[1]))
    raise ValueError(
        f"the column name and column specification cannot be solved; "
        f"got '{col_spec}'"
    )


def solve_names_and_columns(
        col_specs: ColumnsOrAliasedColumns
) -> Iterator[Tuple[str, Optional[Union[Column, List[Column]]]]]:
    """
    Solves column names and column objects for one or more column specifications.
    See :func:`solve_name_and_column` for a single column specification.

    Examples:
        >>> list(solve_names_and_columns('col1'))
        [('col1', Column<'col1'>)]
        >>> list(solve_names_and_columns(['col1', 'col2']))
        [('col1', Column<'col1'>), ('col2', Column<'col2'>)]
        >>> list(solve_names_and_columns([('col1_alias', 'col1'), ('col2_alias', 'col2')]))
        [('col1_alias', Column<'col1'>), ('col2_alias', Column<'col2'>)]
        >>> list(solve_names_and_columns({'col1_alias': 'col1', 'col2_alias': 'col2'}))
        [('col1_alias', Column<'col1'>), ('col2_alias', Column<'col2'>)]
        >>> import pyspark.sql.functions as F
        >>> list(solve_names_and_columns({'col1_alias': F.max('col1'), 'col2_alias': F.min('col2')}))
        [('col1_alias', Column<'max(col1)'>), ('col2_alias', Column<'min(col2)'>)]
        >>> list(
        ...    solve_names_and_columns(
        ...       {'reduced_col1': [F.max('col1'), 'col2'], 'reduced_col12': [F.min('col2'), 'col3']}
        ...    )
        ... )
        [('reduced_col1', [Column<'max(col1)'>, Column<'col2'>]), ('reduced_col12', [Column<'min(col2)'>, Column<'col3'>])]
    """
    if col_specs is None:
        yield '', None
    if isinstance(col_specs, str):
        # a single column name is specified
        yield col_specs, col(col_specs)
    else:
        if isinstance(col_specs, (Mapping, list, tuple)):
            yield from (
                solve_name_and_column(_col)
                for _col in solve_key_value_pairs(
                col_specs, parse_seq_as_alternating_key_value=False
            )
            )
        else:
            raise ValueError(
                f"the column name and column specifications cannot be solved; got '{col_specs}'"
            )


def is_nested_colname(df: DataFrame, colname: str):
    colname_splits = colname.split('.', maxsplit=1)
    if len(colname_splits) == 1:
        return False
    return colname_splits[0] in df.columns


def nested_colname_bisplit(
        df: DataFrame,
        colname: str,
        check_existence=True
) -> Tuple[str, Optional[str]]:
    """
    A convenience function to split the name of a nested column into two parts,
    the top-level column name, and the remaining name.

    For example, the field name 'a.b.c.d' will be split into 'a' and 'b.c.d'.

    This function will perform name existence checking if `check_existence` is set True,
        1) check if the whole name 'a.b.c.d' itself exists_path in the dataframe `df`,
            if so the whole name 'a.b.c.d' will be returned as the top-level column name
            without split.
        2) check if the possible top-level column name 'a' exists_path in the dataframe `df`,
            if not, the whole name 'a.b.c.d' will be returned as the top-level column name
            without split.

    """
    if check_existence and (colname in df.columns):
        return colname, None
    colname_splits = colname.split('.', maxsplit=1)
    if len(colname_splits) == 1:
        return colname, None
    if (not check_existence) or colname_splits[0] in df.columns:
        return colname_splits[0], colname_splits[1]
    else:
        return colname, None


def _get_col_type_from_col(df: DataFrame, col: Union[str, Column]) -> DataType:
    if isinstance(col, str) and col in df.columns:
        return df.schema[col].dataType
    coltype = df.select(col).schema[0]
    return coltype.dataType if isinstance(coltype, StructField) else coltype


def get_coltype(df: DataFrame, *cols: Union[str, Column]) -> Union[List[DataType], DataType]:
    """
    Gets the data type of one or more columns.

    For example,
    >>> get_coltype(df, 'col1')
    >>> get_coltype(df, 'col1', 'col2')
    >>> get_coltype(df, ['col1', 'col2'])

    """
    if len(cols) == 1:
        col = cols[0]
        if isinstance(col, (list, tuple)):
            # this is needed to support calling this function
            # like `get_coltype(df, ['col1', 'col2'])`
            return get_coltype(df, *col)
        else:
            return _get_col_type_from_col(df, col)
    else:
        return [_get_col_type_from_col(df, col) for col in cols]


def has_col(df: DataFrame, _col):
    """
    Tests if a column of the specified name exists_path; can test the existence of a nested column.

    Examples:
        >>> has_col(df, 'col1')
        >>> has_col(df, 'a.b.c.e')

    """
    if _col is None or (isinstance(_col, str) and _col == ''):
        return False
    if isinstance(_col, str):
        if _col in df.columns:
            return True
        else:
            try:
                df.select(F.col(_col))
                return True
            except:
                return False
    else:
        try:
            df.select(_col)
            return True
        except:
            return False


def iter_col_and_name(
        df: DataFrame,
        *cols: Union[str, Column]
) -> Iterator[Tuple[Column, str]]:
    if len(cols) == 1:
        col = cols[0]
        if isinstance(col, str):
            yield F.col(col), col
        elif isinstance(col, (list, tuple)):
            yield from iter_col_and_name(df, *col)
        else:
            yield col, df.select(col).columns[0]
    else:
        yield from zip(cols, df.select(*cols).columns)


def get_col_and_name(
        df: DataFrame,
        *cols: Union[str, Column]
) -> Union[Tuple[Column, str], List[Tuple[Column, str]]]:
    if len(cols) == 1:
        col = cols[0]
        if isinstance(col, str):
            return F.col(col), col
        elif isinstance(col, (list, tuple)):
            return get_col_and_name(df, *col)
        else:
            return col, df.select(col).columns[0]
    else:
        return list(zip(cols, df.select(*cols).columns))


def is_cols_distinct(df: DataFrame, *cols: Union[str, Column]) -> bool:
    """
    Checks if the specified columns are distinct in the dataframe.
    """
    if not cols:
        return df.distinct().count() == df.count()
    cols = solve_nested_singleton_tuple_list(cols, atom_types=(str, Column))
    return df.select(*cols).distinct().count() == df.count()


def _solve_name_for_exploded_column(
        df: DataFrame,
        colname: str,
        colname_connector='_'
) -> str:
    """
    Checks if the dataframe `df` contains a column resulted from one explosion transformation,
    given the `colname` in dot-separated format,
    and returns the corresponding column name after the explosion.

    Sometimes we use dot-separated name 'a.b' to conveniently refer to
    field 'b' in an array 'a' of structs, for example,
    {
        'a': [
            {
                'b': 1,
                'c': 2
            },
            {
                'b': 3,
                'c': 4
            }
        ]
    }
    We assume after explosion it becomes the following if `colname_connector` is '_'.
    [
        {
            'a_b': 1,
            'a_c': 2
        },
        {
            'a_b: 3,
            'a_c':4
        }
    ]

    The `df` is the dataframe after the explosion transformation.
    Then given the dot-separated `colname` 'a.b' and the dataframe `df`,
    this function infers the column name after the explosion transformation is 'a_b'.

    If `colname` itself exists in `df` as the top-level column,
    or the solution fails,
    then the original `colname` is returned.

    """
    if colname in df.columns:
        return colname
    colname_primary, colname_secondary = nested_colname_bisplit(
        df, colname, check_existence=False
    )
    if not colname_secondary:
        return colname
    possible_colname = colname_primary + colname_connector + colname_secondary
    if has_col(df, possible_colname):
        return possible_colname
    else:
        return colname


# endregion

# region typing
def get_spark_type(python_obj_or_type, integer_type=IntegerType, float_number_type=FloatType):
    if not isinstance(python_obj_or_type, type):
        python_obj_or_type = type(python_obj_or_type)
    if python_obj_or_type is int:
        return integer_type()
    elif python_obj_or_type is float:
        return float_number_type()
    elif python_obj_or_type is bool:
        return BooleanType()
    else:
        return StringType()


def get_spark_type_recursive(
        obj, field_name_prefix='', nullable=True, integer_type=IntegerType, float_number_type=FloatType
):
    if isinstance(obj, (tuple, list)):
        if not obj:
            raise ValueError("unable to determine array element type")
        return ArrayType(
            get_spark_type_recursive(
                obj[0], integer_type=integer_type, float_number_type=float_number_type
            ),
            containsNull=nullable,
        )
    elif isinstance(obj, Mapping):
        return StructType(
            [
                StructField(
                    field_name_prefix + k,
                    get_spark_type_recursive(
                        v, integer_type=integer_type, float_number_type=float_number_type
                    ),
                    nullable=nullable,
                )
                for k, v in obj.items()
            ]
        )
    else:
        return get_spark_type(
            type(obj), integer_type=integer_type, float_number_type=float_number_type
        )


def cast_col_types(df, col_types):
    for src, trg in solve_key_value_pairs(col_types):
        df = df.withColumn(src, F.col(src).cast(trg() if callable(trg) else trg))
    return df


def _is_array_selection(selection: str):
    return selection[0] == '[' and selection[-1] == ']'


def select_fields(root_schema: StructType, selection: str) -> List[StructField]:
    """
    Retrieves a list of StructFields from a given schema. It handles nested fields and array types.

    Args:
        root_schema (StructType): The schema of the DataFrame to be searched.
        selection (str): The field to be searched. It can be a simple string representing a column name,
                         or a nested field specified as "field1.field2...".
                         Array fields can be represented using square brackets, such as "[field1|field2|field3]".

    Returns:
        List[StructField]: The StructFields of the specified fields.

    Raises:
        ValueError: If 'root_schema' is not an instance of ArrayType or a StructType.
                    If a part of the 'selection' string doesn't exist in the schema.

    Examples:
        >>> from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType
        >>> schema = StructType([
        ...     StructField("field1", IntegerType(), True),
        ...     StructField("field2", StructType([
        ...         StructField("subfield1", IntegerType(), True),
        ...         StructField("subfield2", ArrayType(IntegerType(), True), True)
        ...     ]), True)
        ... ])
        >>> select_fields(schema, 'field1')
        [StructField('field1', IntegerType(), True)]
        >>> select_fields(schema, 'field2.subfield1')
        [StructField('subfield1', IntegerType(), True)]
        >>> select_fields(schema, 'field2.subfield2')
        [StructField('subfield2', ArrayType(IntegerType(), True), True)]
        >>> select_fields(schema, '[field1|field2.subfield1]')
        [StructField('field1', IntegerType(), True), StructField('subfield1', IntegerType(), True)]

    Note:
        The function assumes that '.' is not part of a field name.
        The function assumes that array fields are enclosed in '[' and ']' and the elements are separated by '|'.
    """
    if not selection:
        return root_schema

    selection_parts = split_with_escape_and_quotes(
        selection,
        delimiter='.',
        escape=None,
        quotes=('[', ']'),
        max_split=1
    )
    current = selection_parts[0]
    remaining = selection_parts[1] if len(selection_parts) > 1 else ""

    if isinstance(root_schema, StructType):
        if _is_array_selection(current):
            current_splits = split_with_escape_and_quotes(
                current[1:-1],
                delimiter='|',
                escape=None,
                quotes=('[', ']')
            )

            return [
                select_fields(
                    root_schema=select_fields(
                        root_schema=root_schema,
                        selection=current_field,
                    ),
                    selection=remaining
                ) for current_field in current_splits
            ]
        else:
            while True:
                # the while-loop is in case 'A.B.C' etc. is itself a column name
                if current in root_schema.names:
                    if remaining:
                        _root_schema = root_schema[current].dataType
                        if isinstance(_root_schema, ArrayType):
                            _root_schema = _root_schema.elementType
                            return StructField(
                                name=current,
                                dataType=ArrayType(
                                    StructType(
                                        fields=make_list_(
                                            select_fields(
                                                root_schema=_root_schema,
                                                selection=remaining
                                            )
                                        )
                                    )
                                )
                            )
                        else:
                            return StructField(
                                name=current,
                                dataType=StructType(
                                    fields=make_list_(
                                        select_fields(
                                            root_schema=_root_schema,
                                            selection=remaining
                                        )
                                    )
                                )
                            )
                    else:
                        return root_schema[current]  # returns StructField
                else:
                    if remaining:
                        remaining_splits = split_with_escape_and_quotes(
                            selection,
                            delimiter='.',
                            escape=None,
                            quotes=('[', ']'),
                            max_split=1
                        )
                        _current = remaining_splits[0]
                        if not _is_array_selection(_current):
                            current += _current
                            remaining = remaining_splits[1]
                        else:
                            raise ValueError(f"{current} is not found in {root_schema.names}")
                    else:
                        raise ValueError(f"{current} is not found in {root_schema.names}")
    else:
        raise ValueError(f"'root_schema' must be a StructType")

# endregion
