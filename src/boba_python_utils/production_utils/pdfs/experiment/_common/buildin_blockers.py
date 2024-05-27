import importlib
from functools import partial
from typing import List, Union, Iterable, Tuple, Mapping

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql.types import BooleanType

import boba_python_utils.general_utils.strex as strex
from boba_python_utils.spark_utils import with_columns


def no_overlap_tokens(query_utterance, candidate_utterance, version='v1'):
    return len(

        set(query_utterance.replace('play', '').split()) &
        set(candidate_utterance.replace('play', "").split())
    ) == 0


def get_blockers_columns_for_spark_dataframe(
        blocker_module: str,
        blockers: Iterable[str],
        query_col: Union[str, Column],
        rewrite_col: Union[str, Column],
        blocker_label_col_name_prefix='blocked'
) -> Mapping:
    pdfs_filters = importlib.import_module(blocker_module)

    blocker_label_cols = {}

    for _block_func_str in blockers:
        if ':' in _block_func_str:
            _filter_name, filter_version = _block_func_str.rsplit(':', maxsplit=1)
        else:
            _filter_name = _block_func_str
            filter_version = None
        filter_udf = F.udf(
            partial(getattr(pdfs_filters, _filter_name), version=filter_version),
            returnType=BooleanType()
        )
        blocker_label_colname = strex.add_prefix(_filter_name, blocker_label_col_name_prefix)
        blocker_label_cols[blocker_label_colname] = filter_udf(
            query_col,
            rewrite_col
        )

    return blocker_label_cols


def apply_blockers_to_spark_dataframe(
        df_data: DataFrame,
        blocker_module: str,
        blockers: Iterable[str],
        query_col: Union[str, Column],
        rewrite_col: Union[str, Column],
        blocker_label_col_name_prefix='blocked'
) -> Tuple[DataFrame, List[str]]:
    blocker_label_cols = get_blockers_columns_for_spark_dataframe(
        blocker_module=blocker_module,
        blockers=blockers,
        query_col=query_col,
        rewrite_col=rewrite_col,
        blocker_label_col_name_prefix=blocker_label_col_name_prefix
    )

    df_data = with_columns(
        df_data,
        blocker_label_cols
    )
    return df_data, list(blocker_label_cols.keys())


