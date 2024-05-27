from abc import ABC
from abc import ABC
from collections import defaultdict
from typing import Any, Callable, Union, Tuple, Iterable, Set
from typing import List
from typing import Mapping

from attr import attrs, attrib
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType

import boba_python_utils.spark_utils.spark_functions as F
import boba_python_utils.spark_utils.common
import boba_python_utils.spark_utils.spark_functions.common
import boba_python_utils.spark_utils as sparku
import boba_python_utils.general_utils.strex as strex
import boba_python_utils.production_utils.pdfs.constants as c
from boba_python_utils.common_utils.typing_helper import make_tuple_, enumerate_
from boba_python_utils.general_utils.graph_util import GraphRelation
from boba_python_utils.general_utils.modeling_utility.graph.graph_data_info import GraphTripletDataInfo, FIELD_NODE_INDEX, FIELD_NODE_TYPE, FIELD_RELATION, FIELD_REVERSED_RELATION
from boba_python_utils.general_utils.modeling_utility.graph.spark_graph_triplet_databuilder import SparkGraphTripletDataBuilder
from boba_python_utils.general_utils.modeling_utility.graph.spark_graph_triplet_relation_expansion import subgraph_expansion_through_relations
from boba_python_utils.spark_utils.common import _solve_name_for_exploded_column
from boba_python_utils.spark_utils.graph import TripletJoinMode, join_triplets_of_two_relations

SpecialTypeCustomer = 'Customer'
DEFAULT_FILENAME_GRAPH_DGL = 'graph.dgl'

def _graph_sample_expansion_filter_by_onehop_dst_node(
        df_graph_full,
        df_graph_sample,
        src_node_type,
        dst_node_type,
        node_col_name,
        node_type_colname,
        node_type_to_node_colname_map,
        relation_id_colname
):
    """
    One-hop expansion of the graph sample.
    `df_graph_sample` must be a subgraph of `df_graph` and they share the relation identifiers;

    First searches `df_graph` for nodes of `src_node_type` that appear in `df_graph_sample`,
    and then retrieves all their corresponding nodes of `dst_node_type`.
    """
    # `df_graph` consists of graph nodes with a relation identifier column
    # (See `GraphTripletBuilder` for the details)
    # `df_graph_sample` is either a subgraph of `df_graph`,
    # or a mapping from node type to single-column node-value dataframes

    if (
            node_type_to_node_colname_map is not None and
            src_node_type in node_type_to_node_colname_map
    ):
        # source node type has dedicated node column,
        # which mean the node of this type appears in every row of `df_graph`,
        # in this case we use node value to filter the full graph
        if isinstance(df_graph_sample, Mapping):
            # when `df_graph_sample` is a mapping from node type to node value dataframe,
            # use this node value dataframe to match the full graph
            df_graph_sample = df_graph_sample[src_node_type]
        else:
            # when `df_graph_sample` is a subgraph dataframe,
            # use the node values in the node column to match the full graph
            src_node_colname = node_type_to_node_colname_map[src_node_type]
            df_graph_sample = df_graph_sample.where(
                F.col(src_node_colname).isNotNull()
            ).select(src_node_colname)

        df_graph_expanded_sample = sparku.filter_by_inner_join_on_columns(
            df_graph_full,
            df_graph_sample,
            prevent_resolved_attribute_missing_error=True
        )
    else:
        # source node type is not saved in a dedicated node column,
        # source node type and value are saved in 'node_type' and 'node_value' columns,
        # and this node type does not appear in every row of `df_graph`.
        # in this case we use relation id to match the full graph
        if isinstance(df_graph_sample, Mapping):
            # when `df_graph_sample` is a mapping from node type to node value dataframe,
            # we use the node value to match the full graph;
            # the matched subgraph has the relation ids we can use to expand the graph sample
            df_graph_sample = sparku.filter_by_inner_join_on_columns(
                df_graph_full.where(F.col(node_type_colname) == src_node_type),
                df_graph_sample[src_node_type],
                prevent_resolved_attribute_missing_error=True
            )
        else:
            # otherwise search `df_graph_sample` for the source node type,
            # and the filtered subgraph has the relation ids we can use to expand the graph sample
            df_graph_sample = df_graph_sample.where(F.col(node_type_colname) == src_node_type)

        # matching the full graph through the relation id
        df_graph_expanded_sample = sparku.filter_by_inner_join_on_columns(
            df_graph_full,
            df_graph_sample,
            [relation_id_colname],
            prevent_resolved_attribute_missing_error=True
        )

    if (
            node_type_to_node_colname_map is not None and
            dst_node_type in node_type_to_node_colname_map
    ):
        dst_node_colname = node_type_to_node_colname_map[dst_node_type]
        return df_graph_expanded_sample.where(
            F.col(dst_node_colname).isNotNull()
        ).select(dst_node_colname)
    else:
        return df_graph_expanded_sample.where(
            F.col(node_type_colname) == dst_node_type
        ).select(node_col_name)


def build_expanded_graph_sample_node_filter(
        df_graph_full,
        df_graph_sample,
        graph_relations: List[GraphRelation],
        node_col_name,
        relation_id_colname,
        node_type_col_name=None,
        node_type_to_node_colname_map=None,
        always_consider_reverse_relation=True,
        num_hops=1,
        cache_options: boba_python_utils.spark_utils.common.CacheOptions = boba_python_utils.spark_utils.common.CacheOptions.IMMEDIATE,
):
    """
    Constructs a filter for `df_graph_full` that would expand the `df_graph_sample`
    by hops within the `df_graph_full`.

    In each one-hop expansion, we iterates through all relations defined in `graph_relations`,
    where each graph relation consists of a `src_node_type` and a `dst_node_type`.
    We search `df_graph_full` for nodes of `src_node_type` that appear in `df_graph_sample`,
    and then retrieves all their corresponding nodes of `dst_node_type`.
    Then we use the expanded nodes from this search as the new `df_graph_sample`,
    and starts the next one-hop expansion.

    Args:
        df_graph_full: the full graph;
        df_graph_sample: the graph sample, must be either
            1. a subgraph of `df_graph_full` that shares the same relation identifiers, or
            2. a mapping between node types an single-column dataframes of node values;
        graph_relations: expands the graph sample using these the graph relations;
        node_col_name: the name of the column that stores node values;
        relation_id_colname: the name of the relation identifier column;
        node_type_col_name: the name of the column that stores node types;
            if not provided, we use '{node_col_name}_type';
        node_type_to_node_colname_map: a mapping from node types to node column names;
            we need this mapping if certain nodes are saved in dedicated node columns.
        always_consider_reverse_relation: True to always consider one-hop expansion through
            the reversed relation regardless of the setting in the graph relation;
        num_hops: how many hops to expand the graph sample;
        cache_options: cache option for the returned filter dataframes.

    Returns: a mapping from node types to single-column dataframes of node values;
        this mapping can be used to filter the full graph `df_graph_full` later
        to complete the expansion of `df_graph_sample`.

    """
    if num_hops <= 0:
        return df_graph_sample
    if not node_type_col_name:
        node_type_col_name = f'{node_col_name}_type'

    out_filters = defaultdict(list)
    while num_hops > 0:
        for graph_relation in graph_relations:
            src_type, dst_type = graph_relation.source_type, graph_relation.destination_type

            out_filters[dst_type].append(
                _graph_sample_expansion_filter_by_onehop_dst_node(
                    df_graph_full=df_graph_full,
                    df_graph_sample=df_graph_sample,
                    src_node_type=src_type,
                    dst_node_type=dst_type,
                    node_col_name=node_col_name,
                    node_type_colname=node_type_col_name,
                    node_type_to_node_colname_map=node_type_to_node_colname_map,
                    relation_id_colname=relation_id_colname
                )
            )
            if always_consider_reverse_relation or graph_relation.reversed_relation:
                out_filters[src_type].append(
                    _graph_sample_expansion_filter_by_onehop_dst_node(
                        df_graph_full=df_graph_full,
                        df_graph_sample=df_graph_sample,
                        src_node_type=dst_type,
                        dst_node_type=src_type,
                        node_col_name=node_col_name,
                        node_type_colname=node_type_col_name,
                        node_type_to_node_colname_map=node_type_to_node_colname_map,
                        relation_id_colname=relation_id_colname
                    )
                )
        df_graph_sample = out_filters
        num_hops -= 1

    return {
        node_type:
            sparku.cache__(
                sparku.union(df_filters).distinct(),
                name=f'extended_node_sample_filter_by_one_hop ({node_type})',
                cache_option=cache_options
            )
        for node_type, df_filters in out_filters.items()
    }


def add_node_index_to_graph_triplets(
        df_graph_triplets,
        node_colname,
        node_type_colname=None,
        out_node_index_colname=None,
        triplet_data_info: GraphTripletDataInfo = None
):
    """
    Adds node indices to a graph triplet dataframe.

    Args:
        df_graph_triplets: the graph triplet dataframe.
        node_colname: the name of the node column.
        node_type_colname: the name of the node type column.
        out_node_index_colname: the name of the column that saves the node indices.

    Returns: a graph triplet dataframe with node indices.

    """
    if not node_type_colname:
        node_type_colname = f'{node_colname}_type'
    if not out_node_index_colname:
        out_node_index_colname = f'{node_colname}_index'

    if triplet_data_info is None:
        node_col_name_first = f'{node_colname}_first'
        node_col_name_second = f'{node_colname}_second'
        node_type_col_name_first = f'{node_type_colname}_first'
        node_type_col_name_second = f'{node_type_colname}_second'
        out_node_index_col_name_first = f'{out_node_index_colname}_first'
        out_node_index_col_name_second = f'{out_node_index_colname}_second'
    else:
        node_col_name_first = triplet_data_info.source_node_field_name
        node_col_name_second = triplet_data_info.destination_node_field_name
        node_type_col_name_first = triplet_data_info.source_node_type_field_name
        node_type_col_name_second = triplet_data_info.destination_node_type_field_name
        out_node_index_col_name_first = triplet_data_info.source_node_index_field_name
        out_node_index_col_name_second = triplet_data_info.destination_node_index_field_name

    # By default we order nodes of each node type by the node values in a descending order,
    # and then assign a graph index to each node under a node type;
    # graph library like DGL also assign node indices in this way.
    df_node_index = sparku.add_group_order_index(
        df_graph_triplets.select(
            F.col(node_type_col_name_first).alias(node_type_colname),
            F.col(node_col_name_first).alias(node_colname)
        ).union(
            df_graph_triplets.select(
                F.col(node_type_col_name_second).alias(node_type_colname),
                F.col(node_col_name_second).alias(node_colname)
            )
        ).dropDuplicates(),
        order_index_col_name=out_node_index_colname,
        group_cols=node_type_colname,
        order_cols=node_colname
    )

    sparku.show_counts(df_node_index, node_type_colname, extra_agg_cols=[F.min(out_node_index_colname), F.max(out_node_index_colname)])

    df_graph_triplets_with_index = sparku.cache__(
        sparku.join_on_columns(
            sparku.join_on_columns(
                df_graph_triplets,
                df_node_index.withColumnRenamed(out_node_index_colname, out_node_index_col_name_first),
                [node_type_col_name_first, node_col_name_first],
                [node_type_colname, node_colname],
                prevent_resolved_attribute_missing_error=True
            ),
            df_node_index.withColumnRenamed(out_node_index_colname, out_node_index_col_name_second),
            [node_type_col_name_second, node_col_name_second],
            [node_type_colname, node_colname],
            prevent_resolved_attribute_missing_error=True
        ),
        name='df_triplets_merged_with_index',
        unpersist=(df_graph_triplets, df_node_index)
    )

    return df_graph_triplets_with_index


def _solve_node_col_from_raw_graph_data(
        df_raw_graph_data: DataFrame,
        node_colname: str,
        output_node_colname: str
):
    """
    Solving node column referred by name `node_colname` in `df_raw_graph_data`;
    assigns alias `output_node_colname` to the resolved node column;
    used in `build_graph_triplets` function.

    The `df_raw_graph_data` will be exploded if `node_colname`
    represents an an array column or a field in an array of structs.
    The method will resolve the node colum to the right one after explosion.
    """
    node_colname_primary, node_colname_secondary = sparku.nested_colname_bisplit(
        df_raw_graph_data, node_colname
    )
    coltype = sparku.get_coltype(df_raw_graph_data, node_colname_primary)
    if isinstance(coltype, ArrayType):
        df_raw_graph_data = sparku.explode_as_flat_columns(df_raw_graph_data.where(F.col(node_colname_primary).isNotNull()), col_to_explode=node_colname_primary, explode_colname_or_prefix=node_colname_primary)
        if node_colname_secondary:
            solved_node_col = F.col(
                f'{node_colname_primary}_{node_colname_secondary}'
            )
            df_raw_graph_data = df_raw_graph_data.where(solved_node_col.isNotNull())
            solved_node_col = solved_node_col.alias(output_node_colname)
        else:
            df_raw_graph_data = df_raw_graph_data.where(F.col(node_colname_primary).isNotNull())
            solved_node_col = F.col(node_colname_primary).alias(output_node_colname)
    else:
        df_raw_graph_data = df_raw_graph_data.where(F.col(node_colname).isNotNull())
        solved_node_col = F.col(node_colname).alias(output_node_colname)
    return df_raw_graph_data, solved_node_col


def build_graph_triplets(
        df_raw_graph_data: DataFrame,
        graph_relations: List[GraphRelation],
        relation_id_colname: str,
        raw_data_node_colname,
        raw_data_node_type_colname=None,
        node_type_to_raw_data_node_colname_map=None,
        triplet_data_info: GraphTripletDataInfo = None,
        output_metadata_colnames: Union[List[str], Tuple[str, ...]] = None,
        essential_raw_graph_data_colnames: Set[str] = None,
        node_filters: Mapping[str, DataFrame] = None,
        enable_metadata: bool = True
):
    """

    Args:
        df_raw_graph_data:
        graph_relations:
        relation_id_colname:
        raw_data_node_colname:
        raw_data_node_type_colname:
        node_type_to_raw_data_node_colname_map:
        output_node_colname:
        output_node_type_colname:
        output_relation_colname:
        output_reversed_relation_colname:
        output_metadata_colnames: meta data column names in the returned triplet dataframe.
        node_filters:

    Returns:

    """
    if not raw_data_node_type_colname:
        raw_data_node_type_colname = f'{raw_data_node_colname}_type'

    if triplet_data_info is None:
        out_src_node_colname = f'{raw_data_node_colname}_first'
        out_trg_node_colname = f'{raw_data_node_colname}_second'
        out_src_node_type_colname = f'{raw_data_node_type_colname}_first'
        out_trg_node_type_colname = f'{raw_data_node_type_colname}_second'
        output_relation_colname = FIELD_RELATION
        output_reversed_relation_colname = FIELD_REVERSED_RELATION
    else:
        out_src_node_colname = triplet_data_info.source_node_field_name
        out_trg_node_colname = triplet_data_info.destination_node_field_name
        out_src_node_type_colname = triplet_data_info.source_node_type_field_name
        out_trg_node_type_colname = triplet_data_info.destination_node_type_field_name
        output_relation_colname = triplet_data_info.relation_field_name
        output_reversed_relation_colname = triplet_data_info.reversed_relation_field_name

    triplets = []
    for graph_relation in graph_relations:
        # region STEP0: determines metadata columns
        def _get_output_metadata_columns(_df_graph_selected):
            _output_metadata_colnames = output_metadata_colnames
            if _output_metadata_colnames is None:
                _output_metadata_colnames = ()

            _output_metadata_colnames = [
                _colname for _colname in _output_metadata_colnames
                if _colname not in essential_raw_graph_data_colnames
            ]

            if not graph_relation.common_metadata_colnames:
                output_metadata_cols = _output_metadata_colnames
            elif isinstance(graph_relation.common_metadata_colnames, Mapping):
                output_metadata_cols = [
                    F.col(
                        _solve_name_for_exploded_column(_df_graph_selected, _colname)
                    ).alias(_output_colname)
                    for _colname, _output_colname
                    in graph_relation.common_metadata_colnames.items()
                ]
            elif isinstance(graph_relation.common_metadata_colnames, (tuple, list)):
                graph_relation_metadata_colnames = [
                    _colname for _colname in graph_relation.common_metadata_colnames
                    if _colname not in essential_raw_graph_data_colnames
                ]
                if len(graph_relation_metadata_colnames) != len(_output_metadata_colnames):
                    raise ValueError(
                        f"specified {len(graph_relation_metadata_colnames)}"
                        f"metadata columns for the augmentation data;"
                        f"expect {len(_output_metadata_colnames)}"
                        f"metadata columns"
                    )

                output_metadata_cols = [
                    F.col(
                        _solve_name_for_exploded_column(_df_graph_selected, _colname)
                    ).alias(_output_colname)
                    for _colname, _output_colname in zip(
                        graph_relation_metadata_colnames,
                        _output_metadata_colnames
                    )
                ]
            else:
                raise ValueError("'metadata_colnames' in a 'GraphRelation' object "
                                 "must be none, a mapping, tuple or list; "
                                 f"got {graph_relation.common_metadata_colnames}")
            return output_metadata_cols

        def _get_extra_metadata_col(_df_graph_selected):
            return boba_python_utils.spark_utils.spark_functions.common_functions.to_str(
                *(
                    _solve_name_for_exploded_column(_df_graph_selected, _colname)
                    for _colname in graph_relation.extra_metadata_colnames
                ),
                concat='|',
                flat_str_for_single_column=True
            )

        # endregion

        src_node_type, trg_node_type = graph_relation.source_type, graph_relation.destination_type
        if (
                src_node_type in node_type_to_raw_data_node_colname_map
                or trg_node_type in node_type_to_raw_data_node_colname_map
        ):
            # when one of `src_node_type` or `dst_node_type` has dedicated node column,
            # then select and filter rows depending on whether `src_node_type` or `dst_node_type`
            # have dedicated node columns

            # region STEP1: row selection (i.e. determine `df_graph_selected`)
            # based on `src_node_type` and `dst_node_type`
            # and whether these node types have dedicated columns in `df_raw_graph_data`
            if src_node_type in node_type_to_raw_data_node_colname_map:
                src_node_colname = node_type_to_raw_data_node_colname_map[src_node_type]
                df_graph_selected, src_node_col = _solve_node_col_from_raw_graph_data(
                    df_raw_graph_data, src_node_colname, out_src_node_colname
                )

                if trg_node_type in node_type_to_raw_data_node_colname_map:
                    # when both `src_node_type` and `trg_node_type` have dedicated node columns
                    trg_node_colname = node_type_to_raw_data_node_colname_map[trg_node_type]
                    df_graph_selected, trg_node_col = _solve_node_col_from_raw_graph_data(
                        df_graph_selected, trg_node_colname, out_trg_node_colname
                    )

                else:
                    # when only `src_node_type` has dedicated node column;
                    # then we filter rows by `dst_node_type`
                    trg_node_col = F.col(raw_data_node_colname).alias(out_trg_node_colname)
                    df_graph_selected = df_graph_selected.where(
                        (F.col(raw_data_node_type_colname) == trg_node_type) &
                        F.col(raw_data_node_colname).isNotNull()
                    )
            else:
                # when only `trg_node_type` has dedicated node column;
                # then we filter rows by `src_node_type`
                src_node_col = F.col(raw_data_node_colname).alias(out_src_node_colname)
                trg_node_colname = node_type_to_raw_data_node_colname_map[trg_node_type]

                df_graph_selected, trg_node_col = _solve_node_col_from_raw_graph_data(
                    df_raw_graph_data, trg_node_colname, out_trg_node_colname
                )
                # dst_node_col = F.col(dst_node_colname).alias(out_node_col_name_second)

                df_graph_selected = df_graph_selected.where(
                    (F.col(raw_data_node_type_colname) == src_node_type) &
                    F.col(raw_data_node_colname).isNotNull()
                )
            # endregion

            # region STEP2: applies filter and constructs triplets
            output_metadata_cols = _get_output_metadata_columns(df_graph_selected)
            # graph_relation_filter = _get_solved_graph_relation_filter(
            #     graph_relation.filter, df_graph_selected
            # )
            if graph_relation.extra_metadata_colnames:
                output_metadata_cols.append(_get_extra_metadata_col(df_graph_selected))

            df_triplets = sparku.where(
                df_graph_selected, graph_relation.filter,
                prevent_resolved_attribute_missing_error=True
            ).select(
                src_node_col,
                trg_node_col,
                F.lit(src_node_type).alias(out_src_node_type_colname),
                F.lit(trg_node_type).alias(out_trg_node_type_colname),
                F.lit(graph_relation.relation).alias(output_relation_colname),
                F.lit(graph_relation.reversed_relation).alias(output_reversed_relation_colname),
                *(output_metadata_cols if enable_metadata else ())
            )
            # endregion

            # region STEP3: applies node filters
            if node_filters is not None:
                _node_filters = {}
                if src_node_type in node_filters:
                    _node_filters[out_src_node_colname] = node_filters[src_node_type]
                if trg_node_type in node_filters:
                    _node_filters[out_trg_node_colname] = node_filters[trg_node_type]
                df_triplets = sparku.where(df_triplets, _node_filters)
            # endregion
        else:
            # when neither `src_node_type` or `dst_node_type` has dedicated node column,
            # then select and filter rows for both `src_node_type` and `dst_node_type`,
            # and then join by the relation identifier

            # region STEP1: row selection and filtering
            df_raw_graph_data = df_raw_graph_data.where(F.col(raw_data_node_colname).isNotNull())

            df_graph_selected1 = sparku.where(
                df_raw_graph_data.where(F.col(raw_data_node_type_colname) == src_node_type),
                graph_relation.filter,
                prevent_resolved_attribute_missing_error=True
            )
            df_graph_selected2 = sparku.where(
                df_raw_graph_data.where(F.col(raw_data_node_type_colname) == trg_node_type),
                graph_relation.filter,
                prevent_resolved_attribute_missing_error=True
            )

            if node_filters is not None:
                df_graph_selected1 = sparku.where(
                    df_graph_selected1,
                    {raw_data_node_colname: node_filters.get(src_node_type, None)},
                    prevent_resolved_attribute_missing_error=True
                )
                df_graph_selected2 = sparku.where(
                    df_graph_selected2,
                    {raw_data_node_colname: node_filters.get(trg_node_type, None)},
                    prevent_resolved_attribute_missing_error=True
                )
            # endregion

            # region STEP2: constructs triplets by joining on the relation identifier
            output_metadata_cols = _get_output_metadata_columns(df_graph_selected1)
            if graph_relation.extra_metadata_colnames:
                output_metadata_cols.append(_get_extra_metadata_col(df_graph_selected1))
            df_triplets = sparku.with_columns(
                sparku.join_on_columns(
                    df_graph_selected1.select(
                        relation_id_colname,
                        F.col(raw_data_node_colname).alias(out_src_node_colname),
                        *(output_metadata_cols if enable_metadata else ())
                    ),
                    df_graph_selected2.select(
                        relation_id_colname,
                        F.col(raw_data_node_colname).alias(out_trg_node_colname),
                    ),
                    [relation_id_colname],
                    prevent_resolved_attribute_missing_error=True
                ),
                {
                    out_src_node_type_colname: F.lit(src_node_type),
                    out_trg_node_type_colname: F.lit(trg_node_type),
                    output_relation_colname: F.lit(graph_relation.relation),
                    output_reversed_relation_colname: F.lit(graph_relation.reversed_relation)
                }
            ).drop(relation_id_colname)
            # endregion

        triplets.append(df_triplets)

    return sparku.union(triplets, allow_missing_columns=True).dropDuplicates([
        out_src_node_colname,
        out_trg_node_colname,
        out_src_node_type_colname,
        out_trg_node_type_colname,
        output_relation_colname,
        output_reversed_relation_colname
    ])


@attrs(slots=True)
class SparkGraphTripletAggregator(SparkGraphTripletDataBuilder, ABC):
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

    Filter:
        Data filtering can be applied in two places during the triplet data construction.

        Graph data can be filtered in the pre-processing phase
        by specifiying arguments for `preprocess_graph_data_filter` and `graph_data_filter_not_apply_to_null`.
        The pre-process graph raw data and augmentation data are exposed to these filters.

        The tirplet data can be filtered by specifying the `filter` in each `GraphRelation` object
        of `graph_relations`, which is intended for relation specific filtering; and the
        `node_filters` of this class, which is intended for node-type specific filtering.

    Metadata:
        Metadata for each triplet are extracted from the raw graph data. Metadata of a filtered/sample
        subgraph is the same as the original graph for the same triplet.

    """
    _graph_relations = attrib(type=List[GraphRelation], default=None)

    _is_personalized_graph = attrib(type=bool, default=False)
    _customer_id_node_type = attrib(type=str, default=SpecialTypeCustomer)
    _common_metadata_colnames = attrib(type=List[str], default=None)
    _augmentation_data_source = attrib(type=Any, default=None)

    _relation_id_colname = attrib(type=str, default='data_id')
    _node_type_colname = attrib(type=str, default='entity_type')
    _node_colname = attrib(type=str, default='entity')
    _node_type_to_node_colname_map = attrib(type=dict, default=None)

    _augmentation_relation_id_colname = attrib(type=str, default='data_id')
    _augmentation_data_node_type_colname = attrib(type=str, default='entity_type')
    _augmentation_data_node_colname = attrib(type=str, default='entity')
    _augmentation_data_colname_suffix = attrib(type=str, default=None)
    _augmentation_data_node_type_to_node_colname_map = attrib(type=dict, default=None)

    # graph data filter/sample options
    _preprocess_graph_data_filter = attrib(type=Union[str, Callable, Iterable[Union[str, Callable]]], default=None)
    _preprocess_graph_data_filter_not_apply_to_null = attrib(type=bool, default=False)
    _node_filters = attrib(type=Mapping[str, Any], default=None)
    _subgraph_expansion_always_consider_reverse_relation = attrib(type=bool, default=True)
    _subgraph_expansion_num_hops = attrib(type=int, default=1)
    # endregion

    _load_external_graph_triplets = attrib(type=Callable, default=None)

    # region abstract methods

    def _pre_process_graph_and_augmentation_data(self, df_graph_raw, df_aug_data):
        """
        Takes in the raw graph dataframe and the augmentation dataframe,
        and returns two dataframes, referred to as `df_graph_processed`
        and `df_augmentation_data_processed` in the following requirement specification.

        The returned `df_graph_processed` must contain the following columns,
        1. an id column to identify each unique relations,
           column name defined by `self._relation_id_colname`;
        2. a column for node type, column name defined by `self._node_type_colname`;
        3. a column for node value, column name defined by `self._node_colname`;
        4. a column to identify each customer, if the graph is personalized.
        The `df_graph_processed` dataframe may contain other metadata columns.

        We might need to add additional nodes from a dataset for training or testing,
        and that is why we need the other dataframe `df_augmentation_data_processed`.
        1. the augmentation dataframe must also have an id column to identify unique relations,
           defined by `self._augmentation_relation_id_colname`.
        2. additional nodes of a particular node type may be stored in a dedicated column,
           defined in node type to column name map
           `self._augmentation_data_node_type_to_node_colname_map`;
        3. additional nodes of multiple types may be stored in a column,
           column name defined by `self._augmentation_data_node_colname`,
           and `self._augmentation_data_colname_suffix`;
           meanwhile there is a node type column,
           column name defined by `self._augmentation_data_node_type_colname`,
           and `self._augmentation_data_colname_suffix`;
        4. the metadata columns must be of the same names as those in `df_graph_processed`,
           with a possible name suffix defined by `self._augmentation_data_colname_suffix`.

        Args:
            df_graph_raw: the raw graph dataframe.
            df_aug_data: the augmentation dataframe.

        Returns: two dataframes, one from the graph, and the other for the augmentation data.

        """
        df_graph_processed, df_augmentation_data_processed = df_graph_raw, df_aug_data
        return df_graph_processed, df_augmentation_data_processed

    def _graph_sample(self, df_graph_processed_augmented, df_augmentation_data_processed):
        return df_graph_processed_augmented

    # endregion

    # region non-abstract methods
    def _apply_preprocesss_graph_filter(
            self,
            df_graph_processed_augmented,
            df_augmentation_data_processed
    ):
        """
        Applies preprocess graph filter.
        Currently we support the following sampling,
            * "filter_by_augmentation_data_customer":
               only keep graph related to customers that appear in the augmentation data;
               effective only if `self._is_personalized_graph` is set True
               and `df_augmentation_data_processed` is provided.

        """
        if not self._preprocess_graph_data_filter:
            return df_graph_processed_augmented

        if self._debug_mode:
            count_df_graph_processed_augmented: int = df_graph_processed_augmented.count()

        df_graph_processed_augmented_filtered = df_graph_processed_augmented
        for filter_idx, _graph_filter_method in enumerate_(self._preprocess_graph_data_filter):
            if callable(_graph_filter_method):
                df_filter: DataFrame = _graph_filter_method(
                    df_graph_processed_augmented_filtered,
                    df_augmentation_data_processed
                )
                filtering_name = f'df_graph_processed_augmented_filtered (filter {filter_idx})'
            elif isinstance(_graph_filter_method, str):
                if self._preprocess_graph_data_filter == 'filter_by_augmentation_data_customer':
                    if self._is_personalized_graph:
                        df_filter = df_augmentation_data_processed.select(
                            self._node_type_to_node_colname_map[self._customer_id_node_type]
                        )
                        filtering_name = f'df_graph_processed_augmented_filtered (filter {filter_idx}: {_graph_filter_method})'
                    else:
                        raise ValueError()
                else:
                    raise ValueError(f"the filter name {_graph_filter_method} is not recognized")
            else:
                raise ValueError("a filter can only be a function that generates a filter dataframe, "
                                 f"or the name of a build-in filter; got {_graph_filter_method}")

            if df_filter is not None:
                df_graph_processed_augmented_filtered = sparku.cache__(
                    sparku.filter_by_inner_join_on_columns(
                        df_graph_processed_augmented_filtered,
                        df_filter,
                        do_not_filter_if_null=self._preprocess_graph_data_filter_not_apply_to_null
                    ),
                    name=filtering_name,
                    unpersist=df_graph_processed_augmented_filtered
                )
                if self._debug_mode:
                    sparku.show_count_ratio(
                        df_graph_processed_augmented_filtered,
                        count_df_graph_processed_augmented,
                        'df_graph_processed_augmented_filtered',
                        'df_graph_processed_augmented'
                    )

        return self._graph_sample(df_graph_processed_augmented_filtered, df_augmentation_data_processed)

    def _get_essential_colnames_of_raw_graph_data(self, customer_colname) -> set:
        if self._is_personalized_graph:
            essential_graph_colnames = {
                customer_colname,
                self._relation_id_colname,
                self._node_type_colname,
                self._node_colname
            }
        else:
            essential_graph_colnames = {
                self._relation_id_colname,
                self._node_type_colname,
                self._node_colname
            }
        return essential_graph_colnames

    def _pre_process(self, df_agg_source):
        # region STEP1: pre-process aggregation source
        if isinstance(df_agg_source, (tuple, list)):
            df_graph_raw, df_aug_data = df_agg_source
        else:
            df_graph_raw = df_agg_source
            df_aug_data = None

        # see the doc of this `_pre_process_graph_and_augmentation_data`
        # for detailed requirement on the returned dataframes.
        (
            df_graph_processed, df_augmentation_data_processed
        ) = self._pre_process_graph_and_augmentation_data(
            df_graph_raw, df_aug_data
        )

        # endregion

        # region STEP2: graph node augmentation
        # extracting additional node data from `df_augmentation_data_processed`
        # and add to the `df_graph_processed`.
        if df_augmentation_data_processed is not None:
            # determines the essential graph columns and the metadata columns
            customer_colname = self._node_type_to_node_colname_map[SpecialTypeCustomer]
            essential_raw_graph_data_colnames = self._get_essential_colnames_of_raw_graph_data(
                customer_colname
            )

            # we augment the graph for every graph relation in `self._graph_relations`
            # with `graph_relation.is_augmented_by_data` being True;
            # we extract node data for both `src_type` and `dst_type` if it is not a customer node.
            node_augmentation_from_data = {}
            for graph_relation in self._graph_relations:
                if graph_relation.is_augmented_by_data:
                    src_type, dst_type = graph_relation.source_type, graph_relation.destination_type
                    for node_type in (src_type, dst_type):
                        if node_type == SpecialTypeCustomer:
                            # TODO: we do not augment customer nodes right now
                            # but this might be needed in some case
                            continue
                        elif node_type not in node_augmentation_from_data:
                            node_colname = self._augmentation_data_node_type_to_node_colname_map.get(
                                node_type, None
                            )
                            aug_data_colname_suffix = (
                                    graph_relation.augmentation_data_colname_suffix
                                    or self._augmentation_data_colname_suffix
                            )

                            if not graph_relation.augmentation_data_metadata_colnames:
                                # no metadata columns to extract from the augmentation data
                                additional_data_col_for_node_augmentation = ()
                            elif isinstance(
                                    graph_relation.augmentation_data_metadata_colnames,
                                    Mapping
                            ):
                                # `graph_relation.metadata_colnames` is explicitly specified
                                # with metadata column names in the augmetnaetion data
                                # and their mapped column names
                                additional_data_col_for_node_augmentation = [
                                    F.col(
                                        strex.add_suffix(
                                            _aug_data_metada_colname,
                                            aug_data_colname_suffix
                                        )
                                    ).alias(_graph_metadata_colname)
                                    for _aug_data_metada_colname, _graph_metadata_colname
                                    in graph_relation.augmentation_data_metadata_colnames.items()
                                ]
                            elif isinstance(
                                    graph_relation.augmentation_data_metadata_colnames,
                                    (tuple, list)
                            ):
                                # gets metadata column names for this relation;
                                # any essential columns specified there will be excluded;
                                # if it is not specified,
                                # then we assume all non-essential columns in `df_graph_processed`
                                # are metadata columns.
                                graph_meta_data_colnames = [
                                    _col for _col in
                                    (
                                            graph_relation.common_metadata_colnames
                                            or self._common_metadata_colnames
                                            or df_graph_processed.columns
                                    )
                                    if _col not in essential_raw_graph_data_colnames
                                ]
                                aug_data_metadata_colnames = [
                                    _col for _col in
                                    graph_relation.augmentation_data_metadata_colnames
                                    if _col not in essential_raw_graph_data_colnames
                                ]
                                if len(aug_data_metadata_colnames) != \
                                        len(graph_meta_data_colnames):
                                    raise ValueError(
                                        f"specified {len(aug_data_metadata_colnames)}"
                                        f"metadata columns for the augmentation data;"
                                        f"expect {len(graph_meta_data_colnames)}"
                                        f"metadata columns"
                                    )

                                additional_data_col_for_node_augmentation = [
                                    F.col(
                                        strex.add_suffix(
                                            _aug_data_metada_colname,
                                            aug_data_colname_suffix
                                        )
                                    ).alias(_graph_metadata_colname)
                                    for _aug_data_metada_colname, _graph_metadata_colname in zip(
                                        aug_data_metadata_colnames,
                                        graph_meta_data_colnames
                                    )
                                ]

                            if node_colname is None:
                                # when a node is stored in a tuple of node type/value columns,
                                # we search the rows for the current `node_type`
                                df_graph_augmentation = df_augmentation_data_processed.where(
                                    F.col(
                                        strex.add_suffix(
                                            self._augmentation_data_node_type_colname,
                                            aug_data_colname_suffix
                                        )
                                    ) == node_type
                                )
                                data_entity_col = F.col(
                                    strex.add_suffix(
                                        self._augmentation_data_node_colname,
                                        aug_data_colname_suffix
                                    )
                                ).alias(self._node_colname)
                            else:
                                # when we have a dedicated column for the current `node_type`
                                data_entity_col = F.col(node_colname).alias(self._node_colname)
                                df_graph_augmentation = df_augmentation_data_processed

                            node_augmentation_from_data[node_type] = df_graph_augmentation.select(
                                self._augmentation_relation_id_colname,
                                *((customer_colname,) if self._is_personalized_graph else ()),
                                *additional_data_col_for_node_augmentation,
                                F.lit(node_type).alias(self._node_type_colname),
                                data_entity_col
                            ).dropDuplicates([self._augmentation_relation_id_colname])

            df_graph_processed_augmented = sparku.cache__(
                sparku.union(
                    df_graph_processed,
                    node_augmentation_from_data,
                    allow_missing_columns=True
                ),
                name='df_graph_processed_augmented'
            )

            if self._debug_mode:
                sparku.show_counts(df_graph_processed_augmented, self._node_type_colname)
        else:
            # if `df_augmentation_data_processed` is None,
            # then node augmentation from data is not possible,
            # and any relation requiring data augmentation will trigger an error.
            for graph_relation in self._graph_relations:
                if graph_relation.is_augmented_by_data:
                    raise ValueError(f"graph relation '{graph_relation}' "
                                     f"requires augmentation data; "
                                     f"but no augmentation data is provided")

            df_graph_processed_augmented = df_graph_processed

        # endregion

        # region STEP3: applies preprocess graph filter
        df_graph_processed_augmented_filtered = self._apply_preprocesss_graph_filter(
            df_graph_processed_augmented, df_augmentation_data_processed
        )
        # endregion

        has_subgraph_expansion = any(
            x.enabled_for_subgraph_expansion for x in self._graph_relations
        )
        if (not has_subgraph_expansion) or self._subgraph_expansion_num_hops <= 0:
            return df_graph_processed_augmented_filtered, None
        else:
            return df_graph_processed_augmented_filtered, df_graph_processed_augmented

    def _aggregate(self, df_agg_source_processed):
        df_processed_raw_graph_data, df_processed_raw_graph_data_before_filter = df_agg_source_processed

        # region STEP1: build graph triplets from the augmented graph, and apply node filters

        essential_raw_graph_data_colnames = self._get_essential_colnames_of_raw_graph_data(
            self._node_type_to_node_colname_map[SpecialTypeCustomer]
        )

        df_graph_triplets = sparku.cache__(
            build_graph_triplets(
                df_raw_graph_data=df_processed_raw_graph_data,
                graph_relations=self._graph_relations,
                relation_id_colname=self._relation_id_colname,
                raw_data_node_colname=self._node_colname,
                raw_data_node_type_colname=self._node_type_colname,
                node_type_to_raw_data_node_colname_map=self._node_type_to_node_colname_map,
                triplet_data_info=self._triplet_data_info,
                output_metadata_colnames=self._common_metadata_colnames,
                essential_raw_graph_data_colnames=essential_raw_graph_data_colnames,
                node_filters=self._node_filters
            ),
            name='df_graph_triplets',
            unpersist=df_processed_raw_graph_data
        )

        if df_processed_raw_graph_data_before_filter is not None:
            df_graph_triplets_without_filter = sparku.cache__(
                build_graph_triplets(
                    df_raw_graph_data=df_processed_raw_graph_data_before_filter,
                    graph_relations=self._graph_relations,
                    relation_id_colname=self._relation_id_colname,
                    raw_data_node_colname=self._node_colname,
                    raw_data_node_type_colname=self._node_type_colname,
                    node_type_to_raw_data_node_colname_map=self._node_type_to_node_colname_map,
                    triplet_data_info=self._triplet_data_info,
                    output_metadata_colnames=self._common_metadata_colnames,
                    essential_raw_graph_data_colnames=essential_raw_graph_data_colnames,
                    node_filters=None
                ),
                name='df_graph_triplets_without_filter'
            )

            df_graph_triplets = subgraph_expansion_through_relations(
                df_subgraph_triplets=df_graph_triplets,
                df_whole_graph_triplets=df_graph_triplets_without_filter,
                graph_relations=self._graph_relations,
                triplet_data_info=self._triplet_data_info,
                num_hops=self._subgraph_expansion_num_hops,
                verbose=self._debug_mode
            )
            df_graph_triplets_without_filter.unpersist()

        # endregion

        # region STEP2: merges with any provided external triplets
        if self._load_external_graph_triplets is not None:
            if isinstance(self._load_external_graph_triplets, str):
                df_graph_triplets_external = sparku.solve_input(
                    self._load_external_graph_triplets,
                    spark=self._spark
                )
            else:
                df_graph_triplets_external = self._load_external_graph_triplets()
            df_graph_triplets = sparku.cache__(
                sparku.union(
                    df_graph_triplets,
                    df_graph_triplets_external
                ),
                name='df_graph_triplets (union with external triplets)',
                unpersist=df_graph_triplets
            )

        # endregion

        # region STEP3: adds node indices for each node type
        df_graph_triplets_with_index = sparku.cache__(
            add_node_index_to_graph_triplets(
                df_graph_triplets=df_graph_triplets,
                node_colname=self._node_colname,
                node_type_colname=self._node_type_colname,
                triplet_data_info=self._triplet_data_info
            ),
            name='df_graph_triplets_with_index',
            unpersist=df_graph_triplets
        )

        # endregion

        if self._debug_mode:
            # displays minimum and maximum node index for each node type
            sparku.show_counts(
                df_graph_triplets_with_index.select(
                    F.col(
                        self._triplet_data_info.source_node_type_field_name
                    ).alias(FIELD_NODE_TYPE),
                    F.col(
                        self._triplet_data_info.source_node_index_field_name
                    ).alias(FIELD_NODE_INDEX)
                ).union(
                    df_graph_triplets_with_index.select(
                        F.col(
                            self._triplet_data_info.destination_node_type_field_name
                        ).alias(FIELD_NODE_TYPE),
                        F.col(
                            self._triplet_data_info.destination_node_index_field_name
                        ).alias(FIELD_NODE_INDEX)
                    )
                ).dropDuplicates(),
                [FIELD_NODE_TYPE],
                extra_agg_cols=[
                    F.min(FIELD_NODE_INDEX),
                    F.max(FIELD_NODE_INDEX)
                ]
            )

            sparku.show_counts(
                df_graph_triplets_with_index,
                [
                    self._triplet_data_info.relation_field_name,
                    self._triplet_data_info.reversed_relation_field_name
                ]
            )

            df_graph_triplets_with_index.orderBy(F.rand()).show(50, False)
        return df_graph_triplets_with_index
    # endregion
