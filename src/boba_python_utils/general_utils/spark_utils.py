import json

from pyspark.sql.types import (
    IntegerType,
    FloatType,
    BooleanType,
    StringType,
    Row,
)

from boba_python_utils.spark_utils.array_operations import *
from boba_python_utils.spark_utils.join_and_filter import *
from boba_python_utils.spark_utils.analysis import *


# region join

def add_overlap_tags(
        df: DataFrame,
        df_overlaps: Union[DataFrame, List[DataFrame]],
        key_colnames: List[str],
        tag_colname: str,
        tag_values: List = None,
        key_colnames_for_overlaps: Optional[List[str]] = None
):
    if df_overlaps is None:
        return df

    df_overlaps = make_list_(df_overlaps)

    if key_colnames is None:
        key_colnames = df_overlaps[0].columns

    if key_colnames_for_overlaps is None:
        key_colnames_for_overlaps = key_colnames

    if tag_values is None:
        tag_values = range(1, len(df_overlaps) + 1)

    df_overlaps_merged = priority_union(
        key_colnames_for_overlaps,
        *(
            df_overlap.select(key_colnames_for_overlaps).distinct().withColumn(
                tag_colname, F.lit(tag_value)
            ) for df_overlap, tag_value in zip(df_overlaps, tag_values)
        )
    )

    return join_on_columns(
        df, df_overlaps_merged, key_colnames, key_colnames_for_overlaps, how='left'
    )


# endregion

# region sample

def sample(
        df: DataFrame,
        sample_ratio_or_size: Union[int, float],
        seed: int = 0
) -> Union[None, DataFrame]:
    """
    Randomly samples from the current dataframe.
    Can specify a sample ratio between 0 and 1 or a fixed sample size.

    Args:
        df: the current dataframe.
        sample_ratio_or_size: a float number between 0 and 1 as sample ratio,
            or an non-negative integer for a fixed sample size.
            If this argument is set as 0, None will be returned.
        seed: the random sample seed.

    Returns: a random sample from the current dataframe;
        None will be returned if argument `sample` is 0.

    """
    if sample_ratio_or_size < 0:
        raise ValueError(f"argument 'sample' must be a non-negative number; got {sample_ratio_or_size}")  # noqa: E501
    if isinstance(sample_ratio_or_size, float):
        if sample_ratio_or_size == 0.0:
            return None
        elif sample_ratio_or_size == 1.0:
            return df
        elif 0.0 < sample_ratio_or_size < 1.0:
            return df.sample(sample_ratio_or_size, seed)
        else:
            sample_ratio_or_size = int(sample_ratio_or_size)

    if isinstance(sample_ratio_or_size, int):
        if sample_ratio_or_size == 0:
            return None
        else:
            return df.orderBy(F.rand(seed)).limit(sample_ratio_or_size)
    else:
        raise ValueError(f"argument 'sample' must be a float or int; got {sample_ratio_or_size}")


# endregion

# region struct


def fold_as_struct(df, cols_to_keep, folded_col_name, cols_to_fold=None):
    if isinstance(cols_to_keep, str):
        cols_to_keep = [cols_to_keep]
    if isinstance(cols_to_fold, str):
        cols_to_fold = [cols_to_fold]

    if not cols_to_keep:
        if not cols_to_fold:
            raise ValueError("Must specify at least one of 'cols_to_keep' and 'cols_to_fold'")
        else:
            cols_to_keep = [_col for _col in df.columns if _col not in cols_to_fold]
    elif not cols_to_fold:
        cols_to_fold = [_col for _col in df.columns if _col not in cols_to_keep]

    return df.select(*cols_to_keep, F.struct(*cols_to_fold).alias(folded_col_name))


def _fill_null_struct_field(struct_field, sub_field_names, values, fill_null_struct_by_values=True):
    if struct_field is None:
        if fill_null_struct_by_values:
            return json.loads(values)
    else:
        sub_field_names = json.loads(sub_field_names)
        values = json.loads(values)
        out = {}
        for sub_field_name in sub_field_names:
            try:
                if struct_field[sub_field_name] is None and sub_field_name in values:
                    out[sub_field_name] = values[sub_field_name]
                else:
                    out[sub_field_name] = struct_field[sub_field_name]
            except:
                raise ValueError(str(struct_field) + '\n' + sub_field_name, '\n' + str(values[sub_field_name]))

        return out


def fill_null_struct_field(df, struct_field_name, values, fill_null_struct_by_values=True):
    field_names = [field.name for field in df.schema[struct_field_name].dataType.fields]
    return df.withColumn(
        struct_field_name,
        F.udf(
            partial(_fill_null_struct_field, fill_null_struct_by_values=fill_null_struct_by_values),
            returnType=df.schema[struct_field_name].dataType,
        )(struct_field_name, F.lit(json.dumps(field_names)), F.lit(json.dumps(values))),
    )


def _merge_fields(*struct_fields, struct_field_names=None, default_values=None):
    # if struct_field_names is not None:
    #     struct_field_names = json.dumps(struct_field_names)
    # if default_values is not None:
    #     default_values = json.dumps(default_values)

    out = {}
    if struct_field_names:
        for struct_field, struct_field_name in zip(struct_fields, struct_field_names):
            if struct_field is None:
                if default_values and struct_field_name in default_values:
                    out.update(default_values[struct_field_name])
            else:
                out.update(struct_field.asDict())
    else:
        for struct_field in struct_fields:
            out.update(struct_field.asDict())
    return out


def merge_structs(
        df, struct_field_names, merged_col_name, drop_input_struct_fields=False, default_values=None
):
    merged_struct_fields = []
    for struct_field_name in struct_field_names:
        merged_struct_fields.extend(df.schema[struct_field_name].dataType.fields)
    merged_struct = StructType(fields=merged_struct_fields)

    df_out = df.withColumn(
        merged_col_name,
        F.udf(
            partial(
                _merge_fields,
                struct_field_names=(struct_field_names if default_values else None),
                default_values=default_values,
            ),
            returnType=merged_struct,
        )(*struct_field_names),
    )
    if drop_input_struct_fields:
        df_out = df_out.drop(*struct_field_names)
    return df_out


def any_sub_field_is_null(df, struct_field_name):
    cond = None
    for field in df.schema[struct_field_name].dataType.fields:
        if cond is None:
            cond = F.col(f'{struct_field_name}.{field.name}').isNull()
        else:
            cond = cond | F.col(f'{struct_field_name}.{field.name}').isNull()
    return cond

# endregion
