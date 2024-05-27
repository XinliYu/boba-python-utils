import json
from abc import ABC
from collections import defaultdict
from datetime import datetime
from functools import partial
from itertools import chain
from os import path
from typing import Callable, Optional, Union, Tuple
from typing import List

from attr import attrs, attrib
from tqdm import tqdm

import boba_python_utils.spark_utils as sparku
from slab_aggregation.aggregator import MultiDayAggregator
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.general_utils.general import make_list_
from boba_python_utils.io_utils.pickle_io import pickle_save, pickle_load, read_all_text
from boba_python_utils.io_utils.json_io import write_json
from boba_python_utils.general_utils.modeling_utility.graph.graph_data_info import GraphTripletDataInfo
from boba_python_utils.general_utils.modeling_utility.graph.spark_graph_triplet_stats import get_node_counts, get_max_node_index
from boba_python_utils.path_utils.path_listing import get_paths_by_pattern
from boba_python_utils.spark_utils.parallel_compute import parallel_compute


def build_dgl_graph_triplet_dicts_from_graph_triplets(
        partition,
        triplet_data_info: GraphTripletDataInfo,
        meta_data_compute=None
):
    relation_dict = {}
    index_to_node_and_metadata_map = {}

    def _add_index_to_node_and_metadata(node_type, node, node_index, meta_data):
        has_metadata = (meta_data is not None)

        if node_type not in index_to_node_and_metadata_map:
            index_to_node_and_metadata_map[node_type] = {}

        _node_and_metadata_dict = index_to_node_and_metadata_map[node_type]
        if node_index not in _node_and_metadata_dict:
            if has_metadata:
                _node_and_metadata_dict[node_index] = (node, meta_data)
            else:
                _node_and_metadata_dict[node_index] = node

    def _add_relation(node_type1, node_type2, relation, node_index1, node_index2, meta_data):
        relation_key = (node_type1, relation, node_type2)
        has_metadata = (meta_data is not None)
        if relation_key not in relation_dict:
            if has_metadata:
                relation_dict[relation_key] = ([node_index1], [node_index2], [meta_data])
            else:
                relation_dict[relation_key] = ([node_index1], [node_index2])
        else:
            id_list_tup = relation_dict[relation_key]
            id_list_tup[0].append(node_index1)
            id_list_tup[1].append(node_index2)
            if has_metadata:
                id_list_tup[2].append(meta_data)

    triplet_data_basic_field_names = (
        triplet_data_info.source_node_type_field_name,
        triplet_data_info.destination_node_type_field_name,
        triplet_data_info.source_node_field_name,
        triplet_data_info.destination_node_field_name,
        triplet_data_info.source_node_index_field_name,
        triplet_data_info.destination_node_index_field_name,
        triplet_data_info.relation_field_name,
        triplet_data_info.reversed_relation_field_name
    )

    for row in partition:
        try:
            (
                src_type,
                dst_type,
                src_node,
                dst_node,
                src_index,
                dst_index,
                relation,
                reversed_relation
            ) = (
                row[_colname] for _colname in triplet_data_basic_field_names
            )
        except Exception as ex:
            print(row)
            raise ex
        _add_index_to_node_and_metadata(
            src_type,
            src_node,
            src_index,
            tuple(
                row[_colname]
                for _colname
                in triplet_data_info.get_node_metadata_field_names(src_type)
            ) if triplet_data_info.has_node_metadata(src_type) else None
        )
        _add_index_to_node_and_metadata(
            dst_type,
            dst_node,
            dst_index,
            tuple(
                row[_colname]
                for _colname
                in triplet_data_info.get_node_metadata_field_names(dst_type)
            ) if triplet_data_info.has_node_metadata(dst_type) else None
        )

        _add_relation(
            src_type,
            dst_type,
            relation,
            src_index,
            dst_index,
            tuple(
                row[_colname]
                for _colname
                in triplet_data_info.get_relation_metadata_field_names(relation)
            ) if triplet_data_info.has_relation_metadata(relation) else None
        )

        if reversed_relation:
            _add_relation(
                dst_type,
                src_type,
                reversed_relation,
                dst_index,
                src_index,
                tuple(
                    row[_colname]
                    for _colname
                    in triplet_data_info.get_relation_metadata_field_names(reversed_relation)
                ) if triplet_data_info.has_relation_metadata(reversed_relation) else None
            )

    node_and_metadata_dict2 = {}
    for node_type, index_to_node_map in index_to_node_and_metadata_map.items():
        node_indexes, nodes_and_metadata = zip(*index_to_node_map.items())
        has_metadata = triplet_data_info.has_node_metadata(node_type)
        if has_metadata:
            nodes, metadata = zip(*nodes_and_metadata)
            metadata = tuple(zip(*metadata))
        else:
            nodes = nodes_and_metadata
            metadata = None
        if meta_data_compute is not None:
            metadata = meta_data_compute(node_type, nodes, metadata)

        if metadata is not None:
            node_and_metadata_dict2[node_type] = list(zip(node_indexes, nodes, *metadata))
        else:
            node_and_metadata_dict2[node_type] = list(zip(node_indexes, nodes))

    return relation_dict, node_and_metadata_dict2


def load_dgl_graph_triplet_dicts(input_dir_path, file_pattern='*.bin', node_count_file='node_count.json'):
    node_count = json.loads(read_all_text(path.join(input_dir_path, node_count_file)))

    all_files = sum(
        (
            get_paths_by_pattern(_input_dir_path, pattern=file_pattern, recursive=False)
            for _input_dir_path in make_list_(path.join(input_dir_path, 'triplets'))
        ), []
    )
    if len(all_files) == 0:
        raise ValueError(f"no graph triplet data files found under dir(s) {input_dir_path}")

    merged_relation_dict, merged_node_and_metadata_dict, merged_relation_metadata_dict = {}, {}, {}
    for data_file in tqdm(all_files, desc='loading graph triplets'):
        relation_dict, node_and_metadata_dict = pickle_load(data_file)
        for k, v in relation_dict.items():
            if k not in merged_relation_dict:
                merged_relation_dict[k] = ([v[0]], [v[1]])
                if (len(v)) == 3:
                    merged_relation_metadata_dict[k] = [v[2]]
            else:
                merged_relation_dict[k][0].append(v[0])
                merged_relation_dict[k][1].append(v[1])
                if (len(v)) == 3:
                    merged_relation_metadata_dict[k].append(v[2])

        for k, v in node_and_metadata_dict.items():
            if k not in merged_node_and_metadata_dict:
                merged_node_and_metadata_dict[k] = [v]
            else:
                merged_node_and_metadata_dict[k].append(v)

    for k, v in tqdm(merged_relation_dict.items(), desc='merged relations'):
        merged_relation_dict[k] = (sum(v[0], []), sum(v[1], []))

    if merged_relation_metadata_dict:
        for k, v in tqdm(merged_relation_metadata_dict.items(), desc='merged relation metadata'):
            merged_relation_metadata_dict[k] = tuple(zip(*sum(v, [])))

    for node_type, node_data in tqdm(merged_node_and_metadata_dict.items(), desc='merged node data'):
        node_index_set = set()

        def _new_node_index(node_index):
            if node_index in node_index_set:
                return False
            else:
                node_index_set.add(node_index)
                return True

        merged_node_and_metadata_dict[node_type] = tuple(zip(*sorted(
            [item for _node_data in node_data for item in _node_data if _new_node_index(item[0])]
        )))

    for node_type, count in node_count.items():
        if count != len(merged_node_and_metadata_dict[node_type][0]):
            raise ValueError()
        if merged_node_and_metadata_dict[node_type][0][0] != 0 and merged_node_and_metadata_dict[node_type][0][-1] != count - 1:
            raise ValueError()

    return merged_relation_dict, merged_node_and_metadata_dict, merged_relation_metadata_dict, node_count


def add_self_relations_by_nodes(relation_dict, self_relation_node_types: List[str]):
    self_relation_node_index = defaultdict(list)
    for k, v in relation_dict.items():
        src_type, relation, dst_type = k
        if src_type in self_relation_node_types:
            self_relation_node_index[src_type].append(v[0])
        if dst_type in self_relation_node_types:
            self_relation_node_index[dst_type].append(v[1])
    for k, v in self_relation_node_index.items():
        v = list(set(chain(*v)))
        relation_dict[(k, f'self_{k}', k)] = (v, v)


def add_self_relations_by_edges(relation_dict, self_relation_edge_types):
    self_relation_node_index = defaultdict(list)
    for k, v in relation_dict.items():
        if k in self_relation_edge_types:
            src_type, relation, dst_type = k
            self_relation_node_index[src_type].append(v[0])
            self_relation_node_index[dst_type].append(v[1])
    for k, v in self_relation_node_index.items():
        v = sum(v, [])
        relation_dict[(k, f'self_{k}', k)] = (v, v)


@attrs(slots=True)
class SparkGraphTripletDataBuilder(MultiDayAggregator, ABC):
    _triplet_data_info = attrib(type=GraphTripletDataInfo, default=None)
    _meta_data_compute = attrib(type=Callable, default=None)
    _output_tmp_path = attrib(type=Union[bool, str], default=None)
    _self_relations = attrib(type=List[Tuple[Union[str, Tuple[str, str, str]]]], default=None)

    def __attrs_post_init__(self):
        super(MultiDayAggregator, self).__attrs_post_init__()
        if self._triplet_data_info is None:
            self._triplet_data_info = GraphTripletDataInfo()

    def get_triplet_local_data_dump_path(self, end_date: datetime, num_days: int) -> Optional[str]:
        return None

    def dump_triplet_data_local(self, end_date: Union[str, datetime], num_days: int, num_files=1000, overwrite=True):
        triplet_local_data_dump_path = self.get_triplet_local_data_dump_path(
            end_date=end_date, num_days=num_days
        )
        if triplet_local_data_dump_path:
            df_graph_triplets = sparku.cache__(
                self.get_output_path(end_date=end_date, num_days=num_days),
                input_format=self._data_format,
                spark=self._spark
            )

            node_counts = get_node_counts(
                df_graph_triplets=df_graph_triplets,
                triplet_data_info=self._triplet_data_info
            )

            hprint_message('node counts', node_counts)
            node_counts2 = get_max_node_index(
                df_graph_triplets=df_graph_triplets,
                triplet_data_info=self._triplet_data_info,
                shift=1
            )

            hprint_message('node counts inferred from node index', node_counts2)

            if node_counts != node_counts2:
                raise ValueError(
                    "node counts are inconsistent with node index; "
                    "usually this is caused by passing a subset the original triplet data; "
                    "reconstruct the triplets data with filters rather than "
                    "dumping a subsets of constructed triplet data"
                )
            else:
                hprint_message('node counts are consistent with node index')

            parallel_compute(
                df=df_graph_triplets,
                partition_transform_func=partial(
                    build_dgl_graph_triplet_dicts_from_graph_triplets,
                    triplet_data_info=self._triplet_data_info,
                    meta_data_compute=self._meta_data_compute
                ),
                combine_partition_transform_func=None,
                repartition=num_files,
                file_based_combine=True,
                output_result_to_files=True,
                output_path=path.join(triplet_local_data_dump_path, 'triplets'),
                output_overwrite=overwrite,
                output_tmp_path=self._output_tmp_path
            )

            write_json(
                node_counts,
                file_path=path.join(triplet_local_data_dump_path, 'node_count.json')
            )

    def load_triplet_data_local_dump(self, end_date: Union[str, datetime], num_days: int):
        triplet_local_data_dump_path = self.get_triplet_local_data_dump_path(
            end_date=end_date, num_days=num_days
        )

        triplet_local_data_dump_path_merged_file = path.join(
            triplet_local_data_dump_path, 'triplets.bin'
        )

        if not path.exists(triplet_local_data_dump_path_merged_file):
            relation_dict, node_metadata_dict, relation_metadata_dict, node_count = load_dgl_graph_triplet_dicts(
                input_dir_path=triplet_local_data_dump_path,
                file_pattern='*.bin',
                node_count_file='node_count.json'
            )
            pickle_save(
                (relation_dict, node_metadata_dict, relation_metadata_dict, node_count),
                triplet_local_data_dump_path_merged_file
            )
        else:
            _tmp = pickle_load(
                triplet_local_data_dump_path_merged_file
            )
            if len(_tmp) == 4:
                relation_dict, node_metadata_dict, relation_metadata_dict, node_count = _tmp
            else:
                relation_dict, node_metadata_dict, node_count = _tmp
                relation_metadata_dict = {}

        self_relation_node_types = [x for x in self._self_relations if isinstance(x, str)]
        self_relation_edge_types = [x for x in self._self_relations if isinstance(x, tuple)]

        add_self_relations_by_nodes(relation_dict, self_relation_node_types)
        for node_type in self_relation_node_types:
            if len(
                    relation_dict[(node_type, f'self_{node_type}', node_type)][0]
            ) != len(
                node_metadata_dict[node_type][0]
            ):
                raise ValueError()

        add_self_relations_by_edges(relation_dict, self_relation_edge_types)
        return relation_dict, node_metadata_dict, relation_metadata_dict, node_count
