from enum import Enum
from typing import List, Any, Union, Tuple, Optional, Iterable

import pyspark.sql.functions as F
from attr import attrs, attrib
from pyspark.sql import DataFrame, Column

from boba_python_utils.general_utils.modeling_utility.graph.graph_data_info import GraphTripletDataInfo
from boba_python_utils.spark_utils.aggregation import top_from_each_group
from boba_python_utils.spark_utils.common import get_internal_colname
from boba_python_utils.spark_utils.data_transform import explode_as_flat_columns
from boba_python_utils.spark_utils.join_and_filter import where, join_on_columns
from boba_python_utils.general_utils.strex import solve_name_conflict


@attrs(slots=True)
class RelationSearchArgs:
    relation = attrib(type=str)
    filter = attrib(type=Any, default=None)
    order_by = attrib(type=List, default=None)
    top = attrib(type=int, default=None)
    keep_connect_node = attrib(type=bool, default=False)
    keep_connect_strength = attrib(type=bool, default=True)

    def get_node_cols(self, df_graph_triplets, graph_triplet_info):
        return get_relation_node_cols(
            df_graph_triplets,
            self.relation,
            graph_triplet_info=graph_triplet_info,
            order_by=self.order_by,
            top=self.top
        )

    def __str__(self):
        return self.relation


def get_relation_info(
        df_graph_triplets: DataFrame,
        relation: Union[str, Any],
        graph_triplet_info: GraphTripletDataInfo
) -> Tuple[Optional[str], Optional[str], Optional[bool]]:
    """
    Gets information about a relation from the graph triplet data:

    Args:
        df_graph_triplets: the graph triplet data.
        relation: the relation, either a string as the relation name,
            or an object with a field 'relation'.
        graph_triplet_info: a `GraphTripletDataInfo` object
            providing necessary information (e.g. field names) of the graph triplet data.

    Returns: a 3-tuple consisting of
        1. the source node type;
        2. the target node type;
        3. if the relation is a reversed relation in the triplet data.

    """
    if not isinstance(relation, str):
        if hasattr(relation, 'relation'):
            relation = relation.relation
    _tmp_row = df_graph_triplets.where(
        F.col(graph_triplet_info.relation_field_name) == relation
    ).head()
    if _tmp_row is None:
        _tmp_row = df_graph_triplets.where(
            F.col(graph_triplet_info.reversed_relation_field_name) == relation
        ).head()
        if _tmp_row is None:
            return None, None, None
        else:
            _src_node_type = _tmp_row[graph_triplet_info.destination_node_type_field_name]
            _dst_node_type = _tmp_row[graph_triplet_info.source_node_type_field_name]
            return _src_node_type, _dst_node_type, True
    else:
        _src_node_type = _tmp_row[graph_triplet_info.source_node_type_field_name]
        _dst_node_type = _tmp_row[graph_triplet_info.destination_node_type_field_name]
        return _src_node_type, _dst_node_type, False


def get_relation_node_cols(
        df_graph_triplets: DataFrame,
        relation: Union[str, RelationSearchArgs],
        graph_triplet_info: GraphTripletDataInfo,
        filter: Any = None,
        order_by: Iterable[Union[str, Column]] = None,
        top: Optional[int] = None
):
    """
    Extracts pairwise node data of the specified relation from graph triplet data.

    Args:
        df_graph_triplets: the graph triplet data.
        relation: the relation, either a string as the relation name,
            or a `RelationSearchArgs` object.
        graph_triplet_info: a `GraphTripletDataInfo` object
            providing necessary information (e.g. field names) of the graph triplet data.
        filter: provides an argument for filtering;
            can be any object compatible with the
            `boba_python_utils.spark_utils.filtering.where` function;
            filter is applied before top selection.
        order_by: order by the specified columns and select top pairwise node data.
        top: order by the specified columns and select top pairwise node data.

    Returns: pairwise node data extracted from the graph triplet data of the specified relation,
        filtering and top selection are applied if specified.

    """
    if isinstance(relation, RelationSearchArgs):
        if filter is None:
            filter = relation.filter
        order_by = order_by or relation.order_by
        top = top or relation.top
        relation = relation.relation
    _df = df_graph_triplets.where(F.col(graph_triplet_info.relation_field_name) == relation)
    _src_node_type, _dst_node_type, _is_reversed_relation = get_relation_info(
        df_graph_triplets, relation, graph_triplet_info
    )
    if _is_reversed_relation:
        _df = where(_df, filter)
        if order_by and top:
            _df = top_from_each_group(
                _df,
                group_cols=[graph_triplet_info.destination_node_field_name],
                order_cols=order_by,
                top=top
            )
        _df = _df.select(
            F.col(graph_triplet_info.destination_node_field_name).alias(graph_triplet_info.source_node_field_name),
            F.col(graph_triplet_info.source_node_field_name).alias(graph_triplet_info.destination_node_field_name),
        )
    else:
        _df = where(_df, filter)
        if order_by and top:
            _df = top_from_each_group(
                _df,
                group_cols=[graph_triplet_info.source_node_field_name],
                order_cols=order_by,
                top=top
            )
        _df = _df.select(
            F.col(graph_triplet_info.destination_node_field_name),
            F.col(graph_triplet_info.source_node_field_name),
        )
    return _df


class TripletJoinMode(str, Enum):
    DstOnSrc = 'dst_on_src'
    DstOnDst = 'dst_on_dst'


def get_triplets_of_relation(
        df_graph_triplets: DataFrame,
        graph_triplet_info: GraphTripletDataInfo,
        relation: Union[str, RelationSearchArgs]
):
    if isinstance(relation, RelationSearchArgs):
        relation = relation.relation
    _, _, _is_reversed_relation = get_relation_info(df_graph_triplets, relation, graph_triplet_info)
    if _is_reversed_relation:
        df = df_graph_triplets.where(F.col(graph_triplet_info.reversed_relation_field_name) == relation)
    else:
        df = df_graph_triplets.where(F.col(graph_triplet_info.relation_field_name) == relation)
    return df, _is_reversed_relation


def join_triplets_of_two_relations(
        df_triplets1: DataFrame,
        df_triplets2: DataFrame,
        join_mode: Union[TripletJoinMode, str],
        src_node_colname: str = None,
        dst_node_colname: str = None,
        keep_connect_node: bool = False,
        connect_node_colname: str = None,
        collect_connect_nodes=True,
        count_connection_strength=False,
        connection_strength_colname=None,
        repartition_before_join: bool = False,
        graph_triplet_info: GraphTripletDataInfo = None
) -> DataFrame:
    """
    Joints two triplet dataframe.

    Args:
        df_triplets1: the first triplet dataframe.
        df_triplets2: the second triplet dataframe.
        join_mode: an option about how to join the two triplet dataframes;
            for example, join the destination node of the first triplet dataframe on the
            source node of the second triplet dataframe, or join the destination node of
            the first triplet dataframe on the destination node of the second triplet dataframe.
        src_node_colname: the column name for the source node;
            overrides the source node column name specified in `graph_triplet_info`.
        dst_node_colname: the column name for the destination node;
            overrides the destination node column name specified in `graph_triplet_info`.
        keep_connect_node: True to keep the connection node in the return.
        connect_node_colname: the column name for the connection node.
        collect_connect_nodes: collects the connection nodes into an array;
            the source node and destination node might be connected through multiple paths,
            if this parameter is set True,
            then the connection nodes are stored in an array column `connect_node_colname`;
            otherwise,
            a flat dataframe with each connection  node in the `connect_node_colname`.
        count_connection_strength:
        connection_strength_colname:
        repartition_before_join:
        graph_triplet_info: an `GraphTripletDataInfo` object providing triplet data information
            like field names of the triplet data.

    Returns:

    """
    src_node_colname = src_node_colname or graph_triplet_info.source_node_field_name
    dst_node_colname = dst_node_colname or graph_triplet_info.destination_node_field_name
    _tmp_dst_node_colname = get_internal_colname(dst_node_colname)
    if join_mode == TripletJoinMode.DstOnSrc:
        df = join_on_columns(
            df_triplets1,
            df_triplets2.withColumnRenamed(dst_node_colname, _tmp_dst_node_colname),
            [dst_node_colname],
            [src_node_colname],
            repartition_before_join=repartition_before_join
        )
    elif join_mode == TripletJoinMode.DstOnDst:
        df = join_on_columns(
            df_triplets1,
            df_triplets2.withColumnRenamed(src_node_colname, _tmp_dst_node_colname),
            [dst_node_colname],
            repartition_before_join=repartition_before_join
        )
    else:
        raise ValueError

    if not connect_node_colname:
        connect_node_colname = 'connect_node'
    if not connection_strength_colname:
        connection_strength_colname = 'connect_strength'

    _ori_connect_node_colname = connect_node_colname
    if keep_connect_node:
        connect_node_colname = solve_name_conflict(
            name=connect_node_colname,
            existing_names=df.columns,
            always_with_suffix=True,
            suffix_sep='_'
        )
        _df = df.groupBy(
            src_node_colname, _tmp_dst_node_colname
        ).agg(F.collect_list(dst_node_colname).alias(connect_node_colname))
        if count_connection_strength:
            connection_strength_colname = solve_name_conflict(
                name=connection_strength_colname,
                existing_names=df.columns,
                always_with_suffix=True,
                suffix_sep='_'
            )
            _df = _df.withColumn(connection_strength_colname, F.size(connect_node_colname))
        if len(df.columns) > 3:
            num_partitions = df.rdd.getNumPartitions()
            df = df.repartition(num_partitions, [src_node_colname, _tmp_dst_node_colname]).join(
                _df.repartition(num_partitions, [src_node_colname, _tmp_dst_node_colname]),
                [src_node_colname, _tmp_dst_node_colname]
            )
        else:
            df = _df

        if not collect_connect_nodes:
            df = explode_as_flat_columns(df, col_to_explode=connect_node_colname, explode_colname_or_prefix=_ori_connect_node_colname)
        else:
            df = df.withColumnRenamed(connect_node_colname, _ori_connect_node_colname)
    else:
        df = df.drop(dst_node_colname)
        if count_connection_strength:
            connection_strength_colname = solve_name_conflict(
                name=connection_strength_colname,
                existing_names=df.columns,
                always_with_suffix=True,
                suffix_sep='_'
            )
            _df = df.groupBy(
                src_node_colname, _tmp_dst_node_colname
            ).agg(F.count('*').alias(connection_strength_colname))
            if len(df.columns) > 3:
                num_partitions = df.rdd.getNumPartitions()
                df = df.repartition(num_partitions, [src_node_colname, _tmp_dst_node_colname]).join(
                    _df.repartition(num_partitions, [src_node_colname, _tmp_dst_node_colname]),
                    [src_node_colname, _tmp_dst_node_colname]
                )
            else:
                df = _df

    return df.withColumnRenamed(_tmp_dst_node_colname, dst_node_colname).distinct()
