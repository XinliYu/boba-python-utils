from typing import Optional, Union, Tuple

from pyspark.sql import DataFrame

from boba_python_utils.spark_utils.aggregation import top_from_each_group
from boba_python_utils.spark_utils.data_transform import explode_as_flat_columns, fold
from boba_python_utils.spark_utils.join_and_filter import join_on_columns
from boba_python_utils.spark_utils.specialized.indexed_data.constants import (
    KEY_DATA_ID, KEY_INDEX_LIST, KEY_INDEX_ITEM_ID
)


def is_flat_indexed_data(
        df_dataset: DataFrame,
        data_id_colname: str = KEY_DATA_ID,
        index_colname: str = KEY_INDEX_LIST,
        dataset_size: Optional[int] = None
) -> bool:
    if dataset_size is None:
        dataset_size = df_dataset.count()
    is_dataset_flat = (index_colname not in df_dataset.columns)
    count_distinct_data_id = df_dataset.select(data_id_colname).distinct().count()
    if (not is_dataset_flat) and count_distinct_data_id != dataset_size:
        raise ValueError(
            f"duplicated data IDs found in the data set; "
            f"dataset size {dataset_size} with {count_distinct_data_id} distinct data IDs"
        )
    return is_dataset_flat


def explode_indexed_data_if_folded(
        df_dataset: DataFrame,
        data_id_colname: str = KEY_DATA_ID,
        index_colname: str = KEY_INDEX_LIST,
        index_item_id_colname: str = KEY_INDEX_ITEM_ID,
        explode_colname_or_prefix: Optional[str] = None,
        return_is_input_dataset_flat: bool = False
) -> Union[
    DataFrame,
    Tuple[DataFrame, bool]
]:
    _is_flat_indexed_data = is_flat_indexed_data(
        df_dataset=df_dataset,
        data_id_colname=data_id_colname,
        index_colname=index_colname
    )
    if not _is_flat_indexed_data:
        df_dataset = explode_as_flat_columns(
            df_dataset,
            col_to_explode=index_colname,
            explode_colname_or_prefix=explode_colname_or_prefix,
            prefix_suffix_exempted_struct_field_names=[index_item_id_colname]
        )

    if return_is_input_dataset_flat:
        return df_dataset, _is_flat_indexed_data
    else:
        return df_dataset


def _fold_flat_index_data(
        df_dataset_flat,
        data_id_colname,
        index_colname,
        top_level_colnames
):
    top_level_colnames = [
        _colname for _colname in df_dataset_flat.columns
        if (_colname != data_id_colname and _colname in top_level_colnames)
    ]
    to_fold_colnames = [
        _colname for _colname in df_dataset_flat.columns
        if (
                _colname != data_id_colname and
                (_colname not in top_level_colnames)
        )
    ]
    return join_on_columns(
        df_dataset_flat.select(data_id_colname, *top_level_colnames),
        fold(
            df_dataset_flat.select(data_id_colname, *to_fold_colnames),
            group_cols=[data_id_colname],
            fold_colname=index_colname
        ),
        [data_id_colname]
    )


def combine_two_indexed_data_of_same_ids(
        df_dataset1: DataFrame,
        df_dataset2: DataFrame,
        data_id_colname1: str = KEY_DATA_ID,
        data_id_colname2: str = KEY_DATA_ID,
        index_item_id_colname1: str = KEY_INDEX_ITEM_ID,
        index_item_id_colname2: str = KEY_INDEX_ITEM_ID,
        index_colname1: str = KEY_INDEX_LIST,
        index_colname2: str = KEY_INDEX_LIST,
        explode_colname_or_prefix: Optional[str] = None,
        fold_after_combine: Optional[bool] = None
):
    df_dataset1_flat, is_flat_indexed_data1 = explode_indexed_data_if_folded(
        df_dataset=df_dataset1,
        data_id_colname=data_id_colname1,
        index_colname=index_colname1,
        index_item_id_colname=index_item_id_colname1,
        explode_colname_or_prefix=explode_colname_or_prefix,
        return_is_input_dataset_flat=True
    )

    df_dataset2_flat, is_flat_indexed_data2 = explode_indexed_data_if_folded(
        df_dataset=df_dataset2,
        data_id_colname=data_id_colname2,
        index_colname=index_colname2,
        index_item_id_colname=index_item_id_colname2,
        explode_colname_or_prefix=explode_colname_or_prefix,
        return_is_input_dataset_flat=True
    )

    df_dataset_combined = join_on_columns(
        df_dataset1_flat,
        df_dataset2_flat,
        [data_id_colname1, index_item_id_colname1],
        [data_id_colname2, index_item_id_colname2],
        avoid_column_name_conflict=True,
        how='left'
    )

    if fold_after_combine is None:
        if is_flat_indexed_data1 and is_flat_indexed_data2:
            fold_after_combine = False
        else:
            fold_after_combine = not is_flat_indexed_data1

    if fold_after_combine:
        df_dataset_combined = _fold_flat_index_data(
            df_dataset_flat=df_dataset_combined,
            data_id_colname=data_id_colname1,
            index_colname=index_colname1,
            top_level_colnames=df_dataset1.columns + df_dataset2.columns
        )

    return df_dataset_combined


def get_indexed_data_top_results(
        df_dataset: DataFrame,
        rank_cols,
        topk: int,
        data_id_colname: str = KEY_DATA_ID,
        index_colname: str = KEY_INDEX_LIST,
        index_item_id_colname: str = KEY_INDEX_ITEM_ID,
        explode_colname_or_prefix: Optional[str] = None,
        fold_top_results: Optional[bool] = None
):
    df_dataset_flat, _is_flat_indexed_data = explode_indexed_data_if_folded(
        df_dataset=df_dataset,
        data_id_colname=data_id_colname,
        index_colname=index_colname,
        index_item_id_colname=index_item_id_colname,
        explode_colname_or_prefix=explode_colname_or_prefix,
        return_is_input_dataset_flat=True
    )

    if fold_top_results is None:
        fold_top_results = not _is_flat_indexed_data

    df_dataset_top_results = top_from_each_group(
        df_dataset_flat,
        top=topk,
        group_cols=[data_id_colname],
        order_cols=rank_cols
    )

    if fold_top_results:
        df_dataset_top_results = _fold_flat_index_data(
            df_dataset_flat=df_dataset_top_results,
            data_id_colname=data_id_colname,
            index_colname=index_colname,
            top_level_colnames=df_dataset.columns
        )
    return df_dataset_top_results
