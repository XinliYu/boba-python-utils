from datetime import datetime
from typing import Callable, Union
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


def expand_relation_through_neighborhood_for_graph_triplet_data(
        df_graph_triplets: DataFrame,
        graph_triplet_data_info: GraphTripletDataInfo,
        target_relation: str,
        expansion_paths: List[Union[RelationSearchArgs, str]],
        connect_node_colname: str = 'connect_node',
        connect_strength_colname: str = 'connect_strength',
        hop_colname: str = 'hop',
        final_process: Callable = None
):
    expansion_source_node_type, _, _ = get_relation_info(
        df_graph_triplets, target_relation, graph_triplet_data_info
    )
    exp_results = []
    for expansion_path in expansion_paths:
        _relation = expansion_path[0]
        _src_node_type, _dst_node_type, _ = get_relation_info(
            df_graph_triplets, _relation, graph_triplet_data_info
        )
        _df = get_relation_node_cols(
            df_graph_triplets,
            _relation,
            graph_triplet_data_info
        )

        for _relation in expansion_path[1:]:
            _prev_df, _prev_src_node_type, _prev_dst_node_type = _df, _src_node_type, _dst_node_type
            _src_node_type, _dst_node_type, _ = get_relation_info(
                df_graph_triplets, _relation, graph_triplet_data_info
            )
            _df = get_relation_node_cols(
                df_graph_triplets,
                _relation,
                graph_triplet_data_info
            )
            if _src_node_type == _prev_dst_node_type:
                triplets_join_mode = TripletJoinMode.DstOnSrc
            elif _dst_node_type == _prev_dst_node_type:
                triplets_join_mode = TripletJoinMode.DstOnDst
            else:
                raise ValueError()
            _df = join_triplets_of_two_relations(
                df_triplets1=_prev_df,
                df_triplets2=_df,
                join_mode=triplets_join_mode,
                graph_triplet_info=graph_triplet_data_info,
                keep_connect_node=(
                    False if isinstance(_relation, str)
                    else _relation.keep_connect_node
                ),
                connect_node_colname=connect_node_colname,
                count_connection_strength=(
                    True if isinstance(_relation, str)
                    else _relation.keep_connect_strength
                ),
                connection_strength_colname=connect_strength_colname
            )

        if _dst_node_type != expansion_source_node_type:
            _df = sparku.cache__(
                _df.repartition(
                    _df.rdd.getNumPartitions(),
                    [graph_triplet_data_info.destination_node_field_name]
                ),
                name=f'ready to expand through relation {_relation}'
            )
            _df = sparku.cache__(
                join_triplets_of_two_relations(
                    df_triplets1=_df,
                    df_triplets2=_df,
                    join_mode=TripletJoinMode.DstOnDst,
                    graph_triplet_info=graph_triplet_data_info,
                    keep_connect_node=(
                        False if isinstance(expansion_path[0], str)
                        else expansion_path[0].keep_connect_node
                    ),
                    connect_node_colname=connect_node_colname,
                    count_connection_strength=True,
                    connection_strength_colname=connect_strength_colname,
                    repartition_before_join=(
                        True if isinstance(expansion_path[0], str)
                        else expansion_path[0].keep_connect_strength
                    )
                ),
                name=f'expanded through relation {_relation}',
                unpersist=_df
            )
        exp_results.append(_df)

    df_source_node_expansion_map = sparku.cache__(
        sparku.union(*exp_results).distinct(),
        name='df_src_node_expansion_map',
        unpersist=exp_results
    ) if len(exp_results) > 1 else exp_results[0]

    df_graph_triplets_target_relation, _is_reversed_relation = get_triplets_of_relation(
        df_graph_triplets, graph_triplet_data_info, target_relation
    )

    df_graph_triplets_target_relation = sparku.cache__(
        df_graph_triplets_target_relation.repartition(
            df_graph_triplets_target_relation.rdd.getNumPartitions(),
            [graph_triplet_data_info.source_node_field_name]
        ),
        name='df_graph_triplets_target_relation',
        unpersist=df_graph_triplets
    )

    if _is_reversed_relation:
        df_graph_triplets_target_relation_exp = join_triplets_of_two_relations(
            df_triplets1=df_source_node_expansion_map,
            df_triplets2=df_graph_triplets_target_relation,
            join_mode=TripletJoinMode.DstOnDst,
            src_node_colname=graph_triplet_data_info.source_node_field_name,
            dst_node_colname=graph_triplet_data_info.destination_node_field_name,
            keep_connect_node=True,
            connect_node_colname=connect_node_colname,
            count_connection_strength=True,
            connection_strength_colname=connect_strength_colname
        )
    else:
        df_graph_triplets_target_relation_exp = join_triplets_of_two_relations(
            df_triplets1=df_source_node_expansion_map,
            df_triplets2=df_graph_triplets_target_relation,
            join_mode=TripletJoinMode.DstOnSrc,
            src_node_colname=graph_triplet_data_info.source_node_field_name,
            dst_node_colname=graph_triplet_data_info.destination_node_field_name,
            keep_connect_node=False,
            connect_node_colname=connect_node_colname,
            count_connection_strength=True,
            connection_strength_colname=connect_strength_colname
        )

    df_graph_triplets_target_relation_exp = sparku.join_on_columns(
        df_graph_triplets_target_relation_exp,
        df_graph_triplets_target_relation.select(
            graph_triplet_data_info.source_node_field_name,
            graph_triplet_data_info.destination_node_field_name,
            F.lit(1).alias(hop_colname)
        ),
        [
            graph_triplet_data_info.source_node_field_name,
            graph_triplet_data_info.destination_node_field_name
        ],
        how='left'
    ).fillna(2, [hop_colname])

    df_graph_triplets_target_relation_exp_final = sparku.cache__(
        (
            df_graph_triplets_target_relation_exp if final_process is None
            else final_process(df_graph_triplets_target_relation_exp)
        ),
        name='df_graph_triplets_target_relation_exp_final',
        unpersist=(
            df_source_node_expansion_map,
            df_graph_triplets_target_relation,
        )
    )

    sparku.show_counts(df_graph_triplets_target_relation_exp_final, 'hop')

    return df_graph_triplets_target_relation_exp_final


def subgraph_expansion_through_relations(
        df_subgraph_triplets: DataFrame,
        df_whole_graph_triplets: DataFrame,
        triplet_data_info: GraphTripletDataInfo,
        graph_relations: List[GraphRelation],
        num_hops: int = 1,
        verbose: bool = False
):
    tmp_colname = sparku.get_internal_colname('tmp')
    _df_graph_triplets_expansion = []
    relations_enabled_for_graph_expansion = filter(
        lambda x: x.enabled_for_subgraph_expansion, graph_relations
    )
    for graph_relation in relations_enabled_for_graph_expansion:
        _df_subgraph_triplets_select = sparku.cache__(
            df_subgraph_triplets.where(
                F.col(triplet_data_info.relation_field_name)
                == graph_relation.relation
            ).select(
                triplet_data_info.source_node_field_name,
                triplet_data_info.destination_node_field_name
            ),
            name='df_subgraph_triplets_select'
        )
        _df_whole_graph_triplets_select = sparku.cache__(
            sparku.where(
                df_whole_graph_triplets.where(
                    F.col(triplet_data_info.relation_field_name)
                    == graph_relation.relation
                ),
                graph_relation.subgraph_expansion_filter
            ).select(
                triplet_data_info.source_node_field_name,
                triplet_data_info.destination_node_field_name
            ),
            name='df_whole_graph_triplets_select'
        )

        for i in range(num_hops):
            _df_graph_triplets_select2 = sparku.cache__(
                join_triplets_of_two_relations(
                    df_triplets1=_df_subgraph_triplets_select,
                    df_triplets2=_df_whole_graph_triplets_select,
                    join_mode=TripletJoinMode.DstOnDst,
                    keep_connect_node=True,
                    connect_node_colname=tmp_colname,
                    graph_triplet_info=triplet_data_info,
                    collect_connect_nodes=False
                ).select(
                    F.col(
                        triplet_data_info.destination_node_field_name
                    ).alias(
                        triplet_data_info.source_node_field_name
                    ),
                    F.col(tmp_colname).alias(
                        triplet_data_info.destination_node_field_name
                    )
                ).distinct(),
                unpersist=_df_subgraph_triplets_select
            )

            if verbose:
                sparku.show_count_ratio(
                    _df_graph_triplets_select2.select(
                        triplet_data_info.source_node_field_name
                    ).distinct(),
                    df_whole_graph_triplets.where(
                        F.col(triplet_data_info.relation_field_name)
                        == graph_relation.relation
                    ).select(
                        triplet_data_info.source_node_field_name
                    ).distinct(),
                    title1=f'subgraph {graph_relation.source_type} expansion {i}',
                    title2=f'whole graph {graph_relation.source_type}'
                )
            _df_subgraph_triplets_select = _df_graph_triplets_select2

        _df_graph_triplets_expansion.append(
            sparku.cache__(
                sparku.filter_by_inner_join_on_columns(
                    df_whole_graph_triplets.where(
                        F.col(triplet_data_info.source_node_type_field_name)
                        == graph_relation.source_type
                    ),
                    _df_subgraph_triplets_select,
                    [triplet_data_info.source_node_field_name]
                )
            )
        )
        _df_graph_triplets_expansion.append(
            sparku.cache__(
                sparku.filter_by_inner_join_on_columns(
                    df_whole_graph_triplets.where(
                        F.col(triplet_data_info.destination_node_type_field_name)
                        == graph_relation.source_type
                    ),
                    _df_subgraph_triplets_select,
                    [triplet_data_info.destination_node_field_name],
                    [triplet_data_info.source_node_field_name]
                ),
                unpersist=(
                    _df_subgraph_triplets_select,
                    _df_whole_graph_triplets_select
                )
            )
        )

    return sparku.cache__(
        sparku.union(
            df_subgraph_triplets,
            *_df_graph_triplets_expansion
        ).dropDuplicates(
            [
                triplet_data_info.source_node_field_name,
                triplet_data_info.destination_node_field_name,
                triplet_data_info.relation_field_name
            ]
        ),
        name='df_subgraph_triplets (expanded)',
        unpersist=(df_subgraph_triplets, *_df_graph_triplets_expansion),
    )


@attrs(slots=True)
class SparkGraphRelationExpansionThroughNeighbors(MultiDayAggregator):
    """
    This aggregator constructs graph triplet representation from raw node data
    consisting of the following columns,
    >>> {'relation_id': '024e46da-0e1d-473b-a23f-5132d8be61b8',
    >>>  'customer_id': 'ACGYVTSHFC2SY',
    >>>  'domain': 'Music',
    >>>  'intent': 'PlayMusicIntent',
    >>>  'customer_count': 1,
    >>>  'node_type': 'Request',
    >>>  'node_value': 'play happier'}

    We usually have the following columns.
        1. `node_type` and  `node_value` reprsent the node, and together they should be able to
            uniquely identify a node in the graph;
        3. (required column) `relation_id` is used to connect the nodes,
            i.e. nodes witht he same `relation_id` is considered to have a link between them;
            the link definitions are specified by the `_graph_relations` argument of this class;
        3. (required column by a personalized graph) `customer_id` column represents
            the customer nodes.

    Names of above required columns can be configured through aggregator's parameters
    `node_type_colname`, `node_colname`, `relation_id_colname`.

    Other columns can be treated as either nodes or metadata. Specify node columns
    in aggregator parameter `node_type_to_node_colname_map`.
        * For examples, if we would like the `domain` or `intent` to be treated as nodes
            and we consider links between them and those represented by `node_type` and `node_value`
            in the same row, then specicy the following in the `node_type_to_node_colname_map` in a
            {'node_type': 'node_value_column'} format,
            >>> {'domain': 'domain', 'intent': 'intent'}
            which mean a node type 'domain' is mapped to the column 'domain',
            and a node type 'intent' is mapped to the column 'intent'.
        * We can consider above nodes indicated in above ways as node with dedicated columns,
            rather than saved in `node_type` / `node_value` columns.
            Again the 'node_type'/'node_value' pairs defined through dedicated columns
            should also uniquely identify a node in the graph.
        * Columns not specified in `node_type_to_node_colname_map` are treated as metadata columns.

    It is easier to derive above data format from non-graph data.
    This aggregator then constructs a graph triplet representation out of the above format.

    """
    data_name = attrib(type=str)
    data_version = attrib(type=str)
    _graph_triplet_data_info = attrib(type=GraphTripletDataInfo)
    _target_relation = attrib(type=str)
    _expansion_paths = attrib(type=List[List[Union[str, RelationSearchArgs]]])
    connect_node_colname = attrib(type=str)
    connect_strength_colname = attrib(type=str)
    hop_colname = attrib(type=str)

    def _get_aggregation_source(self, end_date: Union[datetime, str], num_days: int):
        return sparku.cache__(
            get_graph_aggregation_path(
                workspace=self._workspace,
                region=self._region,
                locale=self._locale,
                data=self.data_name,
                date=end_date,
                version=self.data_version,
                multi_day_data=True,
                num_days_backward=num_days
            ),
            spark=self._spark,
            name='df_graph_triplets'
        )

    def get_output_path(self, end_date: Union[datetime, str], num_days: int) -> str:
        return get_graph_aggregation_path(
            workspace=self._workspace,
            region=self._region,
            locale=self._locale,
            data=self.data_name,
            date=end_date,
            version=f'{self.data_version}-{self._target_relation}_expanded',
            multi_day_data=True,
            num_days_backward=num_days
        )

    def _final_process(self, df_graph_triplets_expanded: DataFrame):
        KEY_CUSTOMER_COUNT_SUM = f'{c.KEY_CUSTOMER_COUNT}_sum'
        KEY_CUSTOMER_COUNT_MAX = f'{c.KEY_CUSTOMER_COUNT}_max'
        KEY_CUSTOMER_COUNT_AVG = f'{c.KEY_CUSTOMER_COUNT}_avg'
        return sparku.one_from_each_group(
            add_agg_cols(
                df_graph_triplets_expanded,
                group_cols=[
                    self._graph_triplet_data_info.source_node_field_name,
                    self._graph_triplet_data_info.destination_node_field_name
                ],
                agg_cols=[
                    F.sum(c.KEY_CUSTOMER_COUNT).alias(KEY_CUSTOMER_COUNT_SUM),
                    F.max(c.KEY_CUSTOMER_COUNT).alias(KEY_CUSTOMER_COUNT_MAX),
                    F.avg(c.KEY_CUSTOMER_COUNT).alias(KEY_CUSTOMER_COUNT_AVG)
                ],
                by_join=True
            ),
            group_cols=[
                self._graph_triplet_data_info.source_node_field_name,
                self._graph_triplet_data_info.destination_node_field_name
            ],
            order_cols=[
                F.col(KEY_CUSTOMER_COUNT_SUM).desc(),
                F.col(KEY_CUSTOMER_COUNT_AVG).desc()
            ]
        )

    def _aggregate(self, df_agg_source_processed):
        # TODO: duplicate `connect_strength_1`
        return expand_relation_through_neighborhood_for_graph_triplet_data(
            df_graph_triplets=df_agg_source_processed,
            graph_triplet_data_info=self._graph_triplet_data_info,
            target_relation=self._target_relation,
            expansion_paths=self._expansion_paths,
            connect_node_colname=self.connect_node_colname,
            connect_strength_colname=self.connect_strength_colname,
            hop_colname=self.hop_colname,
            final_process=self._final_process
        )

# spark = sparku.get_spark(variable_dict=globals())
# relation_colname = c.KEY_RELATION
# reversed_relation_colname = c.KEY_REVERSED_RELATION
# src_node_colname = c.KEY_ENTITY_FIRST
# dst_node_colname = c.KEY_ENTITY_SECOND
# src_node_type_colname = c.KEY_ENTITY_TYPE_FIRST
# dst_node_type_colname = c.KEY_ENTITY_TYPE_SECOND
# connect_node_colname = 'connect_node'
# connect_strength_colname = 'connect_strength'
# target_relation = 'customer_saves_recipe'
# hop_colname = 'hop'
# graph_triplet_data_info = GraphTripletDataInfo(
#     source_node_field_name=src_node_colname,
#     destination_node_field_name=dst_node_colname,
#     source_node_type_field_name=src_node_type_colname,
#     destination_node_type_colname=dst_node_type_colname,
#     relation_field_name=relation_colname,
#     reversed_relation_field_name=reversed_relation_colname
# )
#
# expansion_paths = [
#     [
#         RelationSearchArgs(
#             relation='customer_saves_recipe',
#             order_by=[F.col(c.KEY_CUSTOMER_COUNT).desc(), F.col(c.KEY_POPULARITY).desc(), F.col(c.KEY_GLOBAL_COUNT).desc()],
#             top=50,
#             keep_connect_node=False
#         )
#     ]
# ]
#
# self = GraphRelationExpansionThroughNeighbors(
#     name='GraphRelationExpansionThroughNeighbors',
#     spark=spark,
#     region='na',
#     locale='en_US',
#     workspace='science',
#     data_name='graph_triplets',
#     data_version='hho_recipe_graph_v1',
#     data_format='json',
#     graph_triplet_data_info=graph_triplet_data_info,
#     target_relation=target_relation,
#     expansion_paths=expansion_paths,
#     connect_node_colname=connect_node_colname,
#     connect_strength_colname=connect_strength_colname,
#     hop_colname=hop_colname
# )
#
# df_graph_triplets = self._get_aggregation_source(
#     end_date='02/28/2021',
#     num_days=60
# )
# graph_triplet_data_info = self._graph_triplet_data_info
# target_relation = self._target_relation
# expansion_paths = self._expansion_paths
# connect_node_colname = self.connect_node_colname
# connect_strength_colname = self.connect_strength_colname
# hop_colname = self.hop_colname
# final_process = self._final_process
#
# sparku.write_df(df_graph_triplets_target_relation_exp_final, self.get_output_path(
#     end_date='02/28/2021',
#     num_days=60
# ), num_files=500)


# s3abml://abml-workspaces-na/slab_graphs/science/graph_triplets/hho_recipe_graph_v1-customer_saves_recipe_expanded/60days/en_US/2021/02/28/
