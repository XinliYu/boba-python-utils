from functools import partial
from typing import List

from pyspark.sql import DataFrame, Column
from pyspark.sql.functions import udf
from pyspark.sql.types import StructField, ArrayType, StructType

from boba_python_utils.common_utils import make_list_
from boba_python_utils.spark_utils.common import get_coltype, select_fields, NameOrColumn


def _select_from_stru_col_by_fields(struct_col: Column, fields: List[StructField]):
    """
    Selects fields from a StructType column of a DataFrame.

    Args:
        struct_col (Column): The column to be selected from.
        fields (List[StructField]): The fields to be selected.

    Returns:
        dict: Dictionary where keys are field names and values are selected fields.
    """
    def _select(
            _name,
            _data_type
    ):
        if isinstance(_data_type, StructType):
            return _select_from_stru_col_by_fields(struct_col[_name], _data_type.fields)
        elif isinstance(_data_type, ArrayType):
            _data_type = _data_type.elementType
            if isinstance(_data_type, StructType):
                return [
                    _select_from_stru_col_by_fields(_row, _data_type.fields)
                    for _row in struct_col[_name]
                ]
            else:
                return struct_col[_name]
        else:
            return struct_col[_name]

    return {
        field.name: _select(field.name, field.dataType)
        for field in fields
    }


def _select_from_arr_col_by_fields(arr_col, fields: List[StructField]):
    """
    Selects fields from an ArrayType column of a DataFrame.

    Args:
        arr_col: The array column to be selected from.
        fields (List[StructField]): The fields to be selected.

    Returns:
        list: List of selected fields.
    """
    return [
        _select_from_stru_col_by_fields(struct_col=row, fields=fields)
        for row in arr_col
    ]


def select_from_nested_struct_or_array(
        df: DataFrame,
        stru_or_arr_col: NameOrColumn,
        selection: str
):
    """
    Selects fields from a nested StructType or ArrayType column of a DataFrame.

    Args:
        df: The DataFrame to be selected from.
        stru_or_arr_col: The StructType or ArrayType column to be selected from.
        selection: The field to be selected.

    Returns:
        Column: Column object containing selected fields.

    Raises:
        ValueError: If 'stru_or_arr_col' is not a StructType or an ArrayType.

    Example:
        >>> from pyspark.sql import SparkSession, Row
        >>> from pyspark.sql.types import *
        >>> spark = SparkSession.builder.getOrCreate()
        >>> data = [
        ...     Row(
        ...         artist_credit=[
        ...             Row(
        ...                 artist=Row(
        ...                     name="Garbage",
        ...                     sort_name="Garbage",
        ...                     disambiguation="US rock band",
        ...                     id="f9ef7a22-4262-4596-a2a8-1d19345b8e50",
        ...                     genres=[
        ...                         Row(name="alternative rock", id="ceeaa283-5d7b-4202-8d1d-e25d116b2a18", count=9),
        ...                         Row(name="dance-rock", id="2c473395-9cd5-465d-8d85-c308de067c04", count=1),
        ...                         Row(name="electronic", id="89255676-1f14-4dd8-bbad-fca839d6aff4", count=1),
        ...                         Row(name="electronic rock", id="c4a69842-f891-4569-9506-1882aa5db433", count=2),
        ...                         Row(name="electronica", id="53a3cea3-17af-4421-a07a-5824b540aeb5", count=1),
        ...                         Row(name="indie rock", id="ccd19ebf-052c-4afe-8ad9-fbb0a73f94a7", count=1),
        ...                         Row(name="rock", id="0e3fc579-2d24-4f20-9dae-736e1ec78798", count=3)
        ...                     ],
        ...                     aliases=[
        ...                         Row(type="Artist name", name="ガービッジ", sort_name="ガービッジ", locale="ja")
        ...                     ]
        ...                 )
        ...             )
        ...         ]
        ...     )
        ... ]
        >>> schema = StructType([
        ...     StructField("artist_credit", ArrayType(
        ...         StructType([
        ...             StructField("artist", StructType([
        ...                 StructField("name", StringType()),
        ...                 StructField("sort_name", StringType()),
        ...                 StructField("disambiguation", StringType()),
        ...                 StructField("id", StringType()),
        ...                 StructField("genres", ArrayType(
        ...                     StructType([
        ...                         StructField("name", StringType()),
        ...                         StructField("id", StringType()),
        ...                         StructField("count", IntegerType())
        ...                     ])
        ...                 )),
        ...                 StructField("aliases", ArrayType(
        ...                     StructType([
        ...                         StructField("type", StringType()),
        ...                         StructField("name", StringType()),
        ...                         StructField("sort_name", StringType()),
        ...                         StructField("locale", StringType())
        ...                     ])
        ...                 ))
        ...             ]))
        ...         ])
        ...     ))
        ... ])
        >>> df = spark.createDataFrame(data, schema)
        >>> struct_col = 'artist_credit'
        >>> selection = 'artist.[name|disambiguation|id|genres.[name|id|count]]'
        >>> output_col = select_from_nested_struct_or_array(df=df, stru_or_arr_col=struct_col, selection=selection)
        >>> df.select(output_col.alias('artist_credit')).head().asDict()
        {'artist-credit': [Row(artist=Row(name='Garbage', disambiguation='US rock band', id='f9ef7a22-4262-4596-a2a8-1d19345b8e50', genres=[Row(name='alternative rock', id='ceeaa283-5d7b-4202-8d1d-e25d116b2a18', count=9), Row(name='dance-rock', id='2c473395-9cd5-465d-8d85-c308de067c04', count=1), Row(name='electronic', id='89255676-1f14-4dd8-bbad-fca839d6aff4', count=1), Row(name='electronic rock', id='c4a69842-f891-4569-9506-1882aa5db433', count=2), Row(name='electronica', id='53a3cea3-17af-4421-a07a-5824b540aeb5', count=1), Row(name='indie rock', id='ccd19ebf-052c-4afe-8ad9-fbb0a73f94a7', count=1), Row(name='rock', id='0e3fc579-2d24-4f20-9dae-736e1ec78798', count=3)]))]}
    """
    schema = get_coltype(df, stru_or_arr_col)
    if isinstance(schema, StructType):
        fields = select_fields(schema, selection)
        return udf(
            partial(_select_from_stru_col_by_fields, fields=fields),
            returnType=StructType(fields)
        )(stru_or_arr_col)
    elif isinstance(schema, ArrayType):
        schema = schema.elementType
        if isinstance(schema, StructType):
            fields = make_list_(select_fields(schema, selection))
            return udf(
                partial(_select_from_arr_col_by_fields, fields=fields),
                returnType=ArrayType(StructType(fields))
            )(stru_or_arr_col)

    raise ValueError(f"'stru_or_arr_col' must be of StructType or ArrayType; got {schema}")
