from pyspark.sql import DataFrame

import boba_python_utils.spark_utils.data_transform
import boba_python_utils.spark_utils as sparku
import boba_python_utils.spark_utils.spark_functions as F
from boba_python_utils.general_utils.console_util import hprint_message


def spark_dataframe_has_id_columns(
        df_data,
        data_id_col_name,
        index_item_id_colname=None,
        index_list_colname=None
):
    if data_id_col_name not in df_data.columns:
        return False

    if not index_item_id_colname:
        return True

    if not index_list_colname:
        raise ValueError(
            "must specify 'index_list_colname' when 'index_item_id_colname' is specified"
        )

    is_folded_data = (index_list_colname in df_data.columns)
    if is_folded_data:
        df_hist_exp = sparku.explode_as_flat_columns(df_data.select(data_id_col_name, index_list_colname), col_to_explode=index_list_colname)
    else:
        df_hist_exp = df_data

    return index_item_id_colname in df_hist_exp.columns


def spark_dataframe_add_data_id(
        df_data: DataFrame,
        data_id_colname: str,
        max_retry_for_distinct_data_id=3,
        data_id_func=F.uuid4,
        overwrite_existing_ids=False
):
    if data_id_colname not in df_data.columns or overwrite_existing_ids:
        while True:
            df_data = sparku.cache__(
                df_data.withColumn(data_id_colname, data_id_func()),  # we use uuid to id a test case
                name=f'add data id column {data_id_colname})',
                unpersist=df_data  # we would not need the input dataframe after this method
            )

            if df_data.select(data_id_colname).distinct().count() == df_data.count():
                break
            max_retry_for_distinct_data_id -= 1
            if max_retry_for_distinct_data_id == 0:
                raise ValueError(f'unable to generate unique {data_id_colname}')

    return df_data


def spark_dataframe_add_index_item_id(
        df_data: DataFrame,
        data_id_colname: str,
        index_item_id_colname: str = None,
        index_list_colname: str = None,
        index_item_id_func=F.uuid4,
        overwrite_existing_ids=False
):
    if (not index_item_id_colname) or (not index_list_colname):
        return df_data

    is_folded_data = (index_list_colname in df_data.columns)
    if is_folded_data:
        sparku.show_counts(df_data, (F.size(index_list_colname) == 0).alias('data without index'))
        df_data_flat = sparku.explode_as_flat_columns(df_data.select(data_id_colname, index_list_colname), col_to_explode=index_list_colname)
    else:
        df_data_flat = df_data

    if (index_item_id_colname in df_data_flat.columns) and (not overwrite_existing_ids):
        return df_data
    else:
        df_data_flat = df_data_flat.withColumn(index_item_id_colname, index_item_id_func())

    if is_folded_data:
        df_data_out = sparku.cache__(
            df_data.drop(index_list_colname).join(boba_python_utils.spark_utils.data_transform.fold(
                df_data_flat, group_cols=[data_id_colname], fold_colname=index_list_colname,
            ), [data_id_colname]),
            name=f'folded dataframe; add index item id column {index_item_id_colname}'
        )
    else:
        df_data_out = sparku.cache__(
            df_data_flat,
            name=f'flat dataframe; add index item id column {index_item_id_colname}'
        )

    if df_data_out.count() != df_data.where(
            F.col(index_list_colname).isNotNull() & (F.size(index_list_colname) != 0)
    ).count():
        raise ValueError(f"number of data records changed "
                         f"after adding {data_id_colname} and {index_item_id_colname}")

    df_data.unpersist()
    return df_data_out


def add_ids_to_data(
        df_data,
        data_id_colname,
        index_item_id_colname,
        index_list_colname,
        output_format='json',
        add_index_item_id=True,
        overwrite_existing_ids: bool = False,
        output_path=None,
        spark=None,
        verbose=True
):
    if isinstance(df_data, str) and output_path is None:
        output_path = df_data
    df_data = sparku.solve_input(df_data, spark=spark)
    data_id_missing = not spark_dataframe_has_id_columns(
        df_data=df_data,
        data_id_col_name=data_id_colname,
        index_item_id_colname=index_item_id_colname,
        index_list_colname=index_list_colname
    )
    if verbose:
        hprint_message('data_id_missing', data_id_missing)
    if data_id_missing:
        df_data = spark_dataframe_add_data_id(
            df_data=df_data,
            data_id_colname=data_id_colname,
            overwrite_existing_ids=overwrite_existing_ids
        )
        if verbose:
            hprint_message('add_index_item_id', add_index_item_id)
        if add_index_item_id:
            df_data = spark_dataframe_add_index_item_id(
                df_data=df_data,
                data_id_colname=data_id_colname,
                index_item_id_colname=index_item_id_colname,
                index_list_colname=index_list_colname,
                overwrite_existing_ids=overwrite_existing_ids
            )
        if output_path:
            sparku.write_df(
                df_data,
                output_path=output_path,
                format=output_format
            )

            # ! must reload data;
            # otherwise `df_data` might become empty
            # because its data source is the old data without ids,
            # which has been deleted at this point
            df_data.unpersist()
            del df_data
            df_data = sparku.solve_input(output_path, spark=spark)
    return df_data
