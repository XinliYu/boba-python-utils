from typing import Union, Callable

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, ArrayType, StructField

from boba_python_utils.common_utils.typing_helper import is_none_or_empty_str
from boba_python_utils.io_utils.pickle_io import pickle_save


def get_field_schema_from_struct(schema: StructType, field_name):
    if field_name in schema.names:
        return schema.fields[schema.names.index(field_name)].dataType


def get_child_schema_by_path(schema_path: str, schema: StructType, return_element_schema: bool = True):
    if not is_none_or_empty_str(schema_path):
        parts = schema_path.split('.')
        for part in parts:
            if isinstance(schema, ArrayType):
                schema = schema.elementType
            if isinstance(schema, StructType):
                try:
                    schema = schema[part].dataType
                except Exception:
                    raise ValueError(
                        f"the field '{part}' of schema path '{schema_path}' "
                        f"cannot be found in {schema}"
                    )
            else:
                raise ValueError(
                    f"schema fetch failed at '{part}' of schema path '{schema_path}'; "
                    f"current schema {schema}"
                )
    if return_element_schema and isinstance(schema, ArrayType):
        schema = schema.elementType
    return schema


def insert_schema(
        schema_path: str,
        schema: StructType,
        insertion_field_name_and_schema,
        overwrite: bool = False
):
    top_schema = schema
    schema = get_child_schema_by_path(
        schema_path=schema_path,
        schema=schema,
        return_element_schema=True
    )
    if isinstance(schema, StructType):
        for field_name, field_schema in insertion_field_name_and_schema.items():
            if field_name in schema.names:
                if overwrite:
                    schema.fields[schema.names.index(field_name)] = \
                        StructField(field_name, field_schema, True)
            else:
                schema.fields.append(StructField(field_name, field_schema, True))
                schema.names.append(field_name)
    else:
        raise ValueError(f"cannot insert fields into data type {type(schema)}")

    return top_schema


def has_field_in_schema(*field_names: str, schema_path: str, schema: StructType):
    schema = get_child_schema_by_path(
        schema_path=schema_path,
        schema=schema,
        return_element_schema=True
    )
    if isinstance(schema, StructType):
        for field_name in field_names:
            if field_name not in schema.names:
                return False
        return True
    else:
        return False


def save_schema(
        dataframe_or_schema: Union[DataFrame, StructType],
        output_path: str,
        write_method: Callable = pickle_save,
        **write_method_kwargs
):
    schema = (
        dataframe_or_schema.schema
        if isinstance(dataframe_or_schema, DataFrame)
        else dataframe_or_schema
    )

    write_method(schema, output_path, **write_method_kwargs)
