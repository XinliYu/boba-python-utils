from pyspark.sql import DataFrame

from boba_python_utils.string_utils.prefix_suffix import add_prefix_suffix
import pyspark.sql.functions as F


def unfold_struct(
        df: DataFrame,
        struct_colname: str,
        col_name_prefix: str = None,
        col_name_suffix: str = None,
        prefix_suffix_sep: str = '_',
        drop_original_col: bool = True,
):
    """
    Unfolds the fields of a top-level struct in the DataFrame as top-level columns.

    Args:
        df: The input DataFrame.
        struct_colname: The name of the top-level struct column whose fields to unfold.
        col_name_prefix: The prefix to add before the column name of each column
            from the struct.
        col_name_suffix: The suffix to add before the column name of each column
            from the struct.
        prefix_suffix_sep: The separator for the prefix and suffix. Defaults to '_'.
        drop_original_col: If True, drop the struct column from the DataFrame.

    Returns:
        DataFrame: A DataFrame with the specified top-level struct unfolded.

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql.types import StructType, StructField, IntegerType
        >>> spark = SparkSession.builder.master("local").appName("unfold_struct").getOrCreate()

        >>> schema = StructType([
        ...     StructField("a", IntegerType()),
        ...     StructField("b", StructType([
        ...         StructField("c", IntegerType()),
        ...         StructField("d", IntegerType())
        ...     ]))
        ... ])

        >>> data = [
        ...     (0, {"c": 1, "d": 2})
        ... ]
        >>> input_df = spark.createDataFrame(data, schema=schema)

        >>> output_df = unfold_struct(input_df, struct_colname='b')
        >>> output_df.show()
        +---+---+---+
        |  a|  c|  d|
        +---+---+---+
        |  0|  1|  2|
        +---+---+---+

        >>> output_df = unfold_struct(input_df, struct_colname='b', col_name_prefix='0',
        ...                           col_name_suffix='2')
        >>> output_df.show()
        +---+---+-------+
        |  a|0_c_2|0_d_2|
        +---+----+------+
        |  0|   1|     2|
        +---+----+------+
    """
    fields = df.schema[struct_colname].dataType.fields
    for field in fields:
        df = df.withColumn(
            add_prefix_suffix(
                s=field.name,
                prefix=col_name_prefix,
                suffix=col_name_suffix,
                sep=prefix_suffix_sep
            ),
            F.col(f'{struct_colname}.{field.name}'),
        )
    if drop_original_col:
        df = df.drop(struct_colname)
    return df
