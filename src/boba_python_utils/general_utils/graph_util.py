from typing import List, Mapping, Any

from attr import attrs, attrib

from boba_python_utils.general_utils.strex import add_prefix, add_suffix


@attrs(slots=True)
class GraphRelation:
    source_type = attrib(type=str)
    destination_type = attrib(type=str)
    relation = attrib(type=str)
    reversed_relation = attrib(type=str, default=None)
    common_metadata_colnames = attrib(type=List[str], default=None)
    extra_metadata_colnames = attrib(type=List[str], default=None)
    is_augmented_by_data = attrib(type=bool, default=False)
    augmentation_data_metadata_colnames = attrib(type=List[str], default=None)
    augmentation_data_colname_suffix = attrib(type=str, default=None)
    filter = attrib(type=Mapping[str, Any], default=None)
    enabled_for_subgraph_expansion = attrib(type=bool, default=False)
    subgraph_expansion_filter = attrib(type=Mapping[str, Any], default=None)

    def __attrs_post_init__(self):
        source_type = self.source_type.lower()
        destination_type = self.destination_type.lower()
        if self.relation:
            if not self.relation.startswith(source_type):
                self.relation = add_prefix(self.relation, prefix=source_type)
            if not self.relation.endswith(destination_type):
                self.relation = add_suffix(self.relation, suffix=destination_type)
        if self.reversed_relation:
            if not self.reversed_relation.startswith(destination_type):
                self.reversed_relation = add_prefix(self.reversed_relation, prefix=destination_type)
            if not self.reversed_relation.endswith(source_type):
                self.reversed_relation = add_suffix(self.reversed_relation, suffix=source_type)
