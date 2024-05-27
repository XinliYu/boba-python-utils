from typing import List, Mapping, Union

from attr import attrs, attrib

FIELD_NODE_FIRST = 'node_first'
FIELD_NODE_SECOND = 'node_second'
FIELD_NODE_TYPE_FIRST = 'node_type_first'
FIELD_NODE_TYPE_SECOND = 'node_type_second'
FIELD_NODE_INDEX_FIRST = 'node_index_first'
FIELD_NODE_INDEX_SECOND = 'node_index_second'
FIELD_RELATION = 'relation'
FIELD_REVERSED_RELATION = 'reversed_relation'
FIELD_NODE_TYPE = 'node_type'
FIELD_NODE_INDEX = 'node_index'


@attrs(slots=True)
class GraphTripletDataInfo:
    source_node_field_name = attrib(type=str, default=FIELD_NODE_FIRST)
    destination_node_field_name = attrib(type=str, default=FIELD_NODE_SECOND)
    source_node_type_field_name = attrib(type=str, default=FIELD_NODE_TYPE_FIRST)
    destination_node_type_field_name = attrib(type=str, default=FIELD_NODE_TYPE_SECOND)
    source_node_index_field_name = attrib(type=str, default=FIELD_NODE_INDEX_FIRST)
    destination_node_index_field_name = attrib(type=str, default=FIELD_NODE_INDEX_SECOND)
    relation_field_name = attrib(type=str, default=FIELD_RELATION)
    reversed_relation_field_name = attrib(type=str, default=FIELD_REVERSED_RELATION)
    node_metadata_field_names = attrib(type=Union[Mapping[str, List[str]], List], default=None)
    relation_metadata_field_names = attrib(type=Union[Mapping[str, List[str]], List], default=None)

    def get_node_metadata_field_names(self, node_type):
        return self.node_metadata_field_names.get(node_type, None) \
            if isinstance(self.node_metadata_field_names, Mapping) \
            else self.node_metadata_field_names

    def get_relation_metadata_field_names(self, relation):
        return self.relation_metadata_field_names.get(relation, None) \
            if isinstance(self.relation_metadata_field_names, Mapping) \
            else self.relation_metadata_field_names

    def has_node_metadata(self, node_type):
        return self.node_metadata_field_names and \
               (
                       isinstance(self.node_metadata_field_names, (list, tuple))
                       or (node_type in self.node_metadata_field_names)
               )

    def has_relation_metadata(self, relation):
        return self.relation_metadata_field_names and \
               (
                       isinstance(self.relation_metadata_field_names, (list, tuple))
                       or (relation in self.relation_metadata_field_names)
               )
