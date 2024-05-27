from os import path
from typing import Tuple

from pyspark.sql import DataFrame

import boba_python_utils.spark_utils.data_transform
from boba_python_utils.general_utils.general import make_list_
import boba_python_utils.production_utils.pdfs.constants as c
import boba_python_utils.spark_utils as sparku
import boba_python_utils.spark_utils.spark_functions as F
from boba_python_utils.spark_utils.specialized.indexed_data.constants import KEY_DATA_ID


def _save(
        output_path: str,
        df_turn_pairs: DataFrame,
        df_history_index: DataFrame,
        save_requests: bool,
        save_history_index: bool,
        save_combined: bool,
        keep_no_hist_reqs=False,
        extra_join_keys=(),
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    df_combined = df_turn_pairs.join(
        df_history_index,
        [c.KEY_CUSTOMER_ID, *extra_join_keys],
        how=('left' if keep_no_hist_reqs else 'inner'),
    )
    if save_requests:
        sparku.write_df(
            df_turn_pairs,
            output_path=path.join(output_path, 'requests'),
            num_files=80,
            repartition=True,
        )

    if save_history_index:
        sparku.write_df(
            df_history_index,
            output_path=path.join(output_path, 'history_index'),
            num_files=80,
            repartition=True,
        )

    if save_combined:
        sparku.write_df(
            df_combined,
            output_path=path.join(output_path, 'combined'),
            num_files=80,
            repartition=True,
        )
    return df_combined, df_turn_pairs, df_history_index


def save_turn_pairs_with_customer_history(
        spark,
        df_rephrase,
        customer_history_index_dataframes_or_paths,
        output_paths,
        customer_history_index_format=None,
        save_requests=True,
        save_history_index=True,
        save_combined=True,
        extra_join_keys=None,
):
    if isinstance(customer_history_index_dataframes_or_paths, DataFrame):
        customer_history_index_dataframes_or_paths = [customer_history_index_dataframes_or_paths]
    else:
        customer_history_index_dataframes_or_paths = make_list_(
            customer_history_index_dataframes_or_paths
        )
    output_paths = make_list_(output_paths)

    out = []
    for i, (_customer_history_dataframe_or_path, _output_path) in enumerate(zip(
            customer_history_index_dataframes_or_paths, output_paths
    )):
        if extra_join_keys:
            if isinstance(extra_join_keys[0], (list, tuple)):
                _extra_join_keys = extra_join_keys[i]
            else:
                _extra_join_keys = extra_join_keys
        else:
            _extra_join_keys = ()
        df_history_index = sparku.solve_input(
            _customer_history_dataframe_or_path,
            spark,
            input_format=customer_history_index_format
        )

        df_history_index = sparku.cache__(
            sparku.filter_by_inner_join_on_columns(
                # TODO online/offline history index column name is different
                df_history_index.withColumnRenamed(c.KEY_HISTORY_LIST_PROD, c.KEY_HISTORY_LIST),
                df_rephrase,
                [c.KEY_CUSTOMER_ID, *_extra_join_keys],
            ),
            name=f'df_history_index',
        )

        df_turn_pairs = sparku.cache__(
            sparku.filter_by_inner_join_on_columns(
                df_rephrase, df_history_index, [c.KEY_CUSTOMER_ID, *_extra_join_keys]
            ),
            name='df_turn_pairs (joint with history)',
        )

        out.append(
            _save(
                output_path=_output_path,
                df_turn_pairs=df_turn_pairs,
                df_history_index=df_history_index,
                save_requests=save_requests,
                save_history_index=save_history_index,
                save_combined=save_combined,
                keep_no_hist_reqs=False,
                extra_join_keys=_extra_join_keys,
            )
        )
        df_rephrase.unpersist()
        df_turn_pairs.unpersist()
        df_history_index.unpersist()
    return out


def is_dataset_dataframe(df):
    """
    We require any pDFS test data to have a 'request_first'
    Args:
        df:

    Returns:

    """
    return c.KEY_REQUEST_FIRST in df.columns


def dataset_has_id_columns(
        df_data,
        data_id_col_name=KEY_DATA_ID,
        hist_id_col_name=c.KEY_INDEX_ITEM_ID,
        history_col_name=c.KEY_HISTORY_LIST
):
    if data_id_col_name not in df_data.columns:
        return False
    is_folded_data = (history_col_name in df_data.columns)
    if is_folded_data:
        df_hist_exp = sparku.explode_as_flat_columns(df_data.select(data_id_col_name, history_col_name), col_to_explode=history_col_name)
    else:
        df_hist_exp = df_data

    return hist_id_col_name in df_hist_exp.columns


def add_id_columns_to_dataset(
        df_data,
        data_id_col_name=KEY_DATA_ID,
        hist_id_col_name=c.KEY_INDEX_ITEM_ID,
        history_col_name=c.KEY_HISTORY_LIST,
        max_retry_for_distinct_data_id=3,
        data_id_func=F.uuid4,
        hist_id_func=F.monotonically_increasing_id,
        overwrite_existing_ids=False
):
    if data_id_col_name not in df_data.columns or overwrite_existing_ids:
        while True:
            df_data = sparku.cache__(
                df_data.withColumn(data_id_col_name, data_id_func()),  # we use uuid to id a test case
                name=f'add data id column {data_id_col_name})',
                unpersist=df_data  # we would not need the input dataframe after this method
            )

            if df_data.select(data_id_col_name).distinct().count() == df_data.count():
                break
            max_retry_for_distinct_data_id -= 1
            if max_retry_for_distinct_data_id == 0:
                raise ValueError(f'unable to generate unique {data_id_col_name}')

    sparku.show_counts(df_data, (F.size(c.KEY_HISTORY_LIST) == 0).alias('no history'))

    is_folded_data = (history_col_name in df_data.columns)
    if is_folded_data:
        df_hist_exp = sparku.explode_as_flat_columns(df_data.select(data_id_col_name, history_col_name), col_to_explode=history_col_name)
    else:
        df_hist_exp = df_data

    if (hist_id_col_name in df_hist_exp.columns) and (not overwrite_existing_ids):
        return df_data
    else:
        df_hist_exp = df_hist_exp.withColumn(hist_id_col_name, hist_id_func())

    if is_folded_data:
        df_data_out = sparku.cache__(
            df_data.drop(history_col_name).join(boba_python_utils.spark_utils.data_transform.fold(
                df_hist_exp, group_cols=[data_id_col_name], fold_colname=history_col_name,
            ), [data_id_col_name]),
            name=f'folded dataframe; add history id column {hist_id_col_name}'
        )
    else:
        df_data_out = sparku.cache__(
            df_hist_exp,
            name=f'flat dataframe; add history id column {hist_id_col_name}'
        )

    if df_data_out.count() != df_data.where(
            F.col(c.KEY_HISTORY_LIST).isNotNull() & (F.size(c.KEY_HISTORY_LIST) != 0)
    ).count():
        raise ValueError(f"number of data records changed "
                         f"after adding {data_id_col_name} and {hist_id_col_name}")

    df_data.unpersist()
    return df_data_out
