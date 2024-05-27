from functools import partial
from typing import Mapping, Callable, List, Iterable
from typing import Union
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from pyspark.sql.types import StructField, ArrayType, StructType, Row

from boba_python_utils.common_utils.array_helper import get_cartesian_product
from boba_python_utils.common_utils.sorting_helper import iter_sorted_
from boba_python_utils.common_utils.typing_helper import solve_atom
from boba_python_utils.spark_utils.common import get_coltype
from boba_python_utils.spark_utils.spark_functions.common import col_
from boba_python_utils.spark_utils.data_transform import with_columns
from boba_python_utils.spark_utils.typing import NameOrColumn


def _sort_array_by_fields(array_col, key_field_names, reverse):
    if isinstance(key_field_names, str):
        return sorted(
            array_col,
            key=lambda item: item[key_field_names],
            reverse=reverse
        )
    else:
        return sorted(
            array_col,
            key=lambda item: tuple(item[field_name] for field_name in key_field_names),
            reverse=reverse
        )


def sort_array(df: DataFrame, arr_colname: str, sort_key: Iterable[str], reverse: bool = False):
    sort_key = solve_atom(sort_key)
    arr_col_type = get_coltype(df, arr_colname)
    return df.withColumn(
        arr_colname,
        F.udf(
            partial(
                _sort_array_by_fields,
                key_field_names=sort_key,
                reverse=reverse
            ), returnType=arr_col_type
        )(arr_colname)
    )



def columns_from_fixed_length_list(
        df,
        list_col,
        new_cols_name_list_or_pattern,
        start_index_for_new_cols_name_pattern=0
):
    len_list = df.select(F.size(list_col)).head()[0]
    if isinstance(new_cols_name_list_or_pattern, str):
        return [
            col_(list_col).getItem(i).alias(
                new_cols_name_list_or_pattern.format(i + start_index_for_new_cols_name_pattern)
            )
            for i in range(len_list)
        ]
    else:
        return [
            col_(list_col).getItem(i).alias(new_cols_name_list_or_pattern[i])
            for i in range(len_list)
        ]


def unfold_columns_from_fixed_length_list(
        df,
        list_col,
        new_cols_name_list_or_pattern,
        start_index_for_new_cols_name_pattern=0,
        drop_list_col=True
):
    len_list = df.select(F.size(list_col)).head()[0]
    if isinstance(new_cols_name_list_or_pattern, str):
        df = with_columns(
            df,
            {
                (new_cols_name_list_or_pattern.format(i + start_index_for_new_cols_name_pattern)):
                    col_(list_col).getItem(i)
                for i in range(len_list)
            }
        )
    else:
        df = with_columns(
            df,
            {
                new_cols_name_list_or_pattern[i]: col_(list_col).getItem(i)
                for i in range(len_list)
            }
        )
    if drop_list_col:
        df = df.drop(list_col)
    return df


def _element_transform_wrap(item, field_names, element_transform):
    out = element_transform(item)
    for field_name in field_names:
        if field_name not in out:
            out[field_name] = item[field_name]
    return out


def _sorted_arr_element_transform(
        arr_col,
        field_names,
        element_transform: Callable[[Mapping], dict],
        key,
        reverse: bool,
        flatten_if_single_field: bool,
        sort_before_transform: bool
):
    if arr_col is not None:
        if element_transform is not None:
            element_transform = partial(
                _element_transform_wrap,
                field_names=field_names,
                element_transform=element_transform
            )

        arr_col = iter_sorted_(
            arr_col,
            key=key,
            reverse=reverse,
            no_sort_if_key_is_none=True,
            element_transform=element_transform,
            sort_before_transform=sort_before_transform
        )
        if flatten_if_single_field and len(field_names) == 1:
            return [row[field_names[0]] for row in arr_col]
        else:
            return [{
                sub_field_name: row[sub_field_name] for sub_field_name in field_names
            } for row in arr_col]


def array_transform(
        df,
        arr_col,
        field_names_to_keep: Union[str, List[str]] = '*',
        field_names_to_exclude: List[str] = None,
        fields_to_add: List[StructField] = None,
        element_transform: Callable[[Mapping], Mapping] = None,
        flatten_if_single_sub_field=False,
        sort_key=None,
        sort_reverse: bool = False,
        sort_before_transform: bool = True,
        out_arr_colname: str = None,
):
    """
    Replaces an StructType array column in the dataframe by a new StructType array of its subfields.
    If `flatten_if_single_sub_field` is True, and only a single subfield is selected, then the new column's data type is an array of the subfield's data type.

    This method ensures the new array is sorted according to `sort_key` and `sort_reverse`.

    Args:
        df: the dataframe.
        arr_col: the ArrayType column.
        field_names_to_keep: keep these sub fields in the new array.
        field_names_to_exclude: excludes these sub fields from the new array.
        flatten_if_single_sub_field: flatten the struct if only single sub field is selected, and in this case the return type is an array of the selected sub field's type.
        sort_key: can be one or more sub field names, or a sorting key function; sort the array using this key before selecting sub fields.
        sort_reverse: True if the sorting is reversed.

    Returns: The dataframe with the specified struct array column replaced by a new array;
    if `flatten_if_single_sub_field` is False, then the new array is a struct array of the selected subfields;
    otherwise, the new array is an array of the single selected sub field's type.

    """
    all_sub_fields = df.select(arr_col).schema[0].dataType.elementType.fields
    if field_names_to_keep == '*':
        sub_fields_to_keep = all_sub_fields
    else:
        sub_fields_to_keep = []
        for sub_field in all_sub_fields:
            if field_names_to_exclude is None or not (sub_field.name in field_names_to_exclude):
                if field_names_to_keep is None or (sub_field.name in field_names_to_keep):
                    sub_fields_to_keep.append(sub_field)
    if fields_to_add:
        sub_fields_to_keep.extend(fields_to_add)
    if not sub_fields_to_keep:
        raise ValueError(f"empty sub fields")

    return_type = ArrayType(
        sub_fields_to_keep[0].dataType
        if flatten_if_single_sub_field and len(sub_fields_to_keep) == 1
        else StructType(fields=sub_fields_to_keep))
    if not out_arr_colname:
        out_arr_colname = str(arr_col)
    return df.withColumn(
        out_arr_colname,
        F.udf(
            partial(
                _sorted_arr_element_transform,
                field_names=[sub_field.name for sub_field in sub_fields_to_keep],
                key=sort_key,
                reverse=sort_reverse,
                flatten_if_single_field=flatten_if_single_sub_field,
                element_transform=element_transform,
                sort_before_transform=sort_before_transform
            ),
            returnType=return_type
        )(out_arr_colname)
    )


def array_self_cartesian_product(
        df: DataFrame,
        arr_col: NameOrColumn,
        output_arr_colname: str,
        item1_fieldname: str,
        item2_fieldname: str = None,
        arr_sort_func: Callable = None,
        bidirection_product: bool = False,
        include_self_product: bool = False
):
    """
    Creates a new array column by Cartesian-product of elements in an existing array.

    Args:
        df: Input DataFrame containing the array column.
        arr_col: Column name or column object of the array column.
        output_arr_colname: Name of the output column containing the Cartesian product.
        item1_fieldname: Name of the first field in the output struct.
        item2_fieldname: Name of the second field in the output struct. If not
            provided or same as `item1_fieldname`, a default name will be generated.
            Defaults to None.
        arr_sort_func: A function to sort the input array before computing
            the Cartesian product.
        bidirection_product: If True, include both (a, b) and (b, a) in the result.
            Defaults to False.
        include_self_product: If True, include pairs with identical elements (a, a)
            in the result. Defaults to False.

    Returns: A DataFrame with a new array column containing the Cartesian product of the input
            array column's elements.

    Raises:
        ValueError: If the input column is not of ArrayType.

    Examples:
        >>> from pyspark.sql import SparkSession
        >>> from pyspark.sql.functions import col
        >>> spark = SparkSession.builder.master("local").appName("array_self_cartesian_product").getOrCreate()

        >>> data = [
        ...     (1, ["a", "b", "c"]),
        ...     (2, ["d", "e"])
        ... ]
        >>> input_df = spark.createDataFrame(data, ["id", "letters"])

        >>> output_df = array_self_cartesian_product(
        ...     input_df,
        ...     arr_col=col("letters"),
        ...     output_arr_colname="letter_pairs",
        ...     item1_fieldname="letter1",
        ...     item2_fieldname="letter2",
        ...     bidirection_product=False,
        ...     include_self_product=False
        ... )

        >>> output_df.show(truncate=False)
        +---+---------+------------------------------------------+
        |id |letters  |letter_pairs                              |
        +---+---------+------------------------------------------+
        |1  |[a, b, c]|[{a, b}, {a, c}, {b, c}]                   |
        |2  |[d, e]   |[{d, e}]                                  |
        +---+---------+------------------------------------------+

    """
    coltype = get_coltype(df, arr_col)
    if item2_fieldname is None or item2_fieldname == item1_fieldname:
        item2_fieldname = f'{item1_fieldname}_2'
        item1_fieldname = f'{item1_fieldname}_1'

    if isinstance(coltype, ArrayType):
        return df.withColumn(
            output_arr_colname,
            F.udf(
                partial(
                    get_cartesian_product,
                    arr_sort_func=arr_sort_func,
                    bidirection_product=bidirection_product,
                    include_self_product=include_self_product
                ),
                returnType=ArrayType(
                    elementType=StructType(
                        fields=[
                            StructField(item1_fieldname, coltype.elementType),
                            StructField(item2_fieldname, coltype.elementType)
                        ]
                    )
                )
            )(arr_col)
        )
    else:
        raise ValueError(f"must provide an array column to 'arr_col'; got {arr_col}")



# region array


def _merge_two_arrays(arr_col1, arr_col2, arr1_field_names, arr2_field_names):
    out = []

    if arr_col1 is None:
        for item2 in arr_col2:
            new_row = {}
            for field in arr1_field_names:
                new_row[field] = None
            for field in arr2_field_names:
                new_row[field] = item2[field] if isinstance(item2, Row) else item2
            out.append(new_row)
    elif arr_col2 is None:
        for item1 in arr_col1:
            new_row = {}
            for field in arr1_field_names:
                new_row[field] = item1[field] if isinstance(item1, Row) else item1
            for field in arr2_field_names:
                new_row[field] = None
            out.append(new_row)
    else:
        for item1, item2 in zip(arr_col1, arr_col2):
            new_row = {}
            for field in arr1_field_names:
                new_row[field] = item1[field] if isinstance(item1, Row) else item1
            for field in arr2_field_names:
                new_row[field] = item2[field] if isinstance(item2, Row) else item2
            out.append(new_row)
    return out


def _merge_arrays(arr_cols_and_field_names):
    pass


def _get_selected_array_fields(df, arr_col, selected_array_field_names):
    arr_element_type = df.schema[arr_col].dataType.elementType
    if isinstance(arr_element_type, StructType):
        arr1_all_fields = {field.name: field for field in arr_element_type.fields}
        if selected_array_field_names is None:
            _array_fields = list(arr1_all_fields.values())
        else:
            _array_fields = []
            for field_name in selected_array_field_names:
                if field_name not in arr1_all_fields:
                    raise ValueError(f"{field_name} is not a sub-field of array column {arr_col}")
                else:
                    _array_fields.append(arr1_all_fields[field_name])
    else:
        if selected_array_field_names is None:
            raise ValueError(f"must provide a field name for non-struct array elements")
        if isinstance(selected_array_field_names, str):
            _array_fields = [
                StructField(name=selected_array_field_names, dataType=arr_element_type)
            ]
        else:
            _array_fields = [
                StructField(name=field_name, dataType=arr_element_type)
                for field_name in selected_array_field_names
            ]
    return _array_fields


def merge_two_arrays(
        df,
        arr_col1,
        arr_col2,
        output_arr_col_name,
        select_arr1_field_names=None,
        select_arr2_field_names=None,
):
    array1_fields = _get_selected_array_fields(df, arr_col1, select_arr1_field_names)
    array2_fields = _get_selected_array_fields(df, arr_col2, select_arr2_field_names)
    return_type = ArrayType(StructType(fields=array1_fields + array2_fields))
    return df.withColumn(
        output_arr_col_name,
        F.udf(
            partial(
                _merge_two_arrays,
                arr1_field_names=[field.name for field in array1_fields],
                arr2_field_names=[field.name for field in array2_fields],
            ),
            returnType=return_type,
        )(arr_col1, arr_col2),
    )

# endregion
