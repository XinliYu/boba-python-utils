from collections import defaultdict
from datetime import datetime
from typing import Callable, Union, Mapping
from typing import List

from attr import attrs, attrib
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import boba_python_utils.spark_utils as sparku
from slab_aggregation.aggregators.base_aggregators import MultiDayAggregator
from boba_python_utils.general_utils.graph_util import GraphRelation
from boba_python_utils.spark_utils.aggregation import add_agg_cols
from boba_python_utils.spark_utils.graph import (
    get_relation_node_cols,
    join_triplets_of_two_relations,
    get_relation_info,
    get_triplets_of_relation,
    TripletJoinMode,
    GraphTripletDataInfo,
    RelationSearchArgs
)
from boba_python_utils.production_utils.graph.common.data_paths import get_graph_aggregation_path
import boba_python_utils.production_utils.pdfs.constants as c


def get_node_counts(
        df_graph_triplets: DataFrame,
        triplet_data_info: GraphTripletDataInfo
) -> Mapping[str, int]:
    _TMP_KEY_NODE_TYPE = sparku.get_internal_colname('node_type')
    _TMP_KEY_NODE_INDEX = sparku.get_internal_colname('node_index')
    df_graph_triplets_unique_index = sparku.cache__(
        df_graph_triplets.select(
            F.col(
                triplet_data_info.source_node_type_field_name
            ).alias(
                _TMP_KEY_NODE_TYPE
            ),
            F.col(
                triplet_data_info.source_node_index_field_name
            ).alias(
                _TMP_KEY_NODE_INDEX
            )
        ).union(
            df_graph_triplets.select(
                F.col(
                    triplet_data_info.destination_node_type_field_name
                ).alias(
                    _TMP_KEY_NODE_TYPE
                ),
                F.col(
                    triplet_data_info.destination_node_index_field_name
                ).alias(
                    _TMP_KEY_NODE_INDEX
                )
            )
        ).dropDuplicates(),
        name='df_graph_triplets_unique_index'
    )

    node_counts = {
        row[0]: row[1]
        for row in df_graph_triplets_unique_index.groupBy(
            _TMP_KEY_NODE_TYPE
        ).agg(
            F.size(
                F.collect_set(_TMP_KEY_NODE_INDEX)
            )
        ).collect()
    }

    df_graph_triplets_unique_index.unpersist()
    return node_counts


def get_max_node_index(
        df_graph_triplets: DataFrame,
        triplet_data_info: GraphTripletDataInfo,
        shift: int = 0
) -> Mapping[str, int]:
    _TMP_KEY_NODE_TYPE = sparku.get_internal_colname('node_type')
    _TMP_KEY_NODE_INDEX = sparku.get_internal_colname('node_index')
    node_max_index = {
        row[0]: (row[1] + shift)
        for row in df_graph_triplets.groupBy(
            F.col(triplet_data_info.source_node_type_field_name).alias(
                _TMP_KEY_NODE_TYPE
            )
        ).agg(
            F.max(triplet_data_info.source_node_index_field_name).alias(
                f'{_TMP_KEY_NODE_INDEX}1'
            )
        ).join(
            df_graph_triplets.groupBy(
                F.col(triplet_data_info.destination_node_type_field_name).alias(
                    _TMP_KEY_NODE_TYPE
                )
            ).agg(
                F.max(triplet_data_info.destination_node_index_field_name).alias(
                    f'{_TMP_KEY_NODE_INDEX}2'
                )
            ),
            [_TMP_KEY_NODE_TYPE],
            how='outer'
        ).fillna(0, [f'{_TMP_KEY_NODE_INDEX}1', f'{_TMP_KEY_NODE_INDEX}2']).withColumn(
            _TMP_KEY_NODE_INDEX,
            F.greatest(f'{_TMP_KEY_NODE_INDEX}1', f'{_TMP_KEY_NODE_INDEX}2')
        ).drop(
            f'{_TMP_KEY_NODE_INDEX}1', f'{_TMP_KEY_NODE_INDEX}2'
        ).collect()
    }

    return node_max_index


def get_nodes_to_index_map(
        df_triplets: DataFrame,
        triplet_data_info: GraphTripletDataInfo,
        collect_map=False
) -> Union[DataFrame, Mapping[str, Mapping[str, int]]]:
    _TMP_KEY_NODE_TYPE = 'node_type'
    _TMP_KEY_NODE = 'node'
    _TMP_KEY_NODE_INDEX = 'index'

    df_node_index_map = sparku.union(
        df_triplets.select(
            F.col(triplet_data_info.source_node_type_field_name).alias(_TMP_KEY_NODE_TYPE),
            F.col(triplet_data_info.source_node_field_name).alias(_TMP_KEY_NODE),
            F.col(triplet_data_info.source_node_index_field_name).alias(_TMP_KEY_NODE_INDEX),
        ),
        df_triplets.select(
            F.col(triplet_data_info.destination_node_type_field_name).alias(_TMP_KEY_NODE_TYPE),
            F.col(triplet_data_info.destination_node_field_name).alias(_TMP_KEY_NODE),
            F.col(triplet_data_info.destination_node_index_field_name).alias(_TMP_KEY_NODE_INDEX),
        )
    ).distinct().orderBy(_TMP_KEY_NODE_TYPE, _TMP_KEY_NODE_INDEX)

    if collect_map:
        from tqdm import tqdm
        out = defaultdict(lambda: defaultdict(dict))
        for row in tqdm(df_node_index_map.collect(), desc='collecting node indexes'):
            out[row[_TMP_KEY_NODE_TYPE]][row[_TMP_KEY_NODE]] = row[_TMP_KEY_NODE_INDEX]
        return out
    else:
        return df_node_index_map
