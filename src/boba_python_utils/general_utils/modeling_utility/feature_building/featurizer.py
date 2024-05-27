from abc import ABC
from typing import List
from typing import Union

from attr import attrs, attrib
from pyspark.sql import DataFrame, SparkSession

from boba_python_utils.common_utils.typing_helper import make_list_


@attrs(slots=False)
class Featurizer(ABC):
    """
    Base featurizer class for exacting features from data.

    The data must have an id field to uniquely identify each data item.

    The output features of this featurizer must be organized in groups called "feature sets"
    where each feature set contains one or more features.

    Attributes:
        featurizer_name: a name for this featurizer; used in logging and messages.
        feature_set_names: the names of the feature sets this featurizer will produce.
        num_feature_sets: the number of feature sets this featurizer will produce.

    """
    featurizer_name = attrib(type=str)
    feature_set_names = attrib(type=Union[str, List[str]])

    def __attrs_post_init__(self):
        self.feature_set_names = make_list_(self.feature_set_names)
        self.num_feature_sets = len(self.feature_set_names)

    def _select_feature_fields(self, data, *field_names):
        raise NotImplementedError

    def _get_features(
            self, data,
            data_id_field_name: str,
            feature_id_field_names: List[str],
            **kwargs
    ):
        """
        Implement this function to get features from the data.

        Must return a feature data object that contains
            1) feature id fields to uniquely identify each feature entry, and
            2) the feature set fields.

        See Also: `get_features`.

        """
        raise NotImplementedError

    def get_features(
            self, data,
            data_id_field_name: str,
            feature_id_field_names: List[str],
            **kwargs
    ):
        """
        Gets features from the data.

        Args:
            data: the data to extract features from.
            data_id_field_name: the field name of data id in `data`.
            feature_id_field_names: the fields names in the generated feature data
                to uniquely identify each feature entry.

        Returns: feature data extracted from the provided `data`;
            it includes feature id fields, and the feature set fields.

        """
        return self._select_feature_fields(
            self._get_features(
                data,
                data_id_field_name=data_id_field_name,
                feature_id_field_names=feature_id_field_names,
                **kwargs
            ),
            *feature_id_field_names,
            *self.feature_set_names
        )


@attrs(slots=False)
class IndexedDataFeaturizer(Featurizer, ABC):
    """
    Base featurizer class for extracting features from indexed data.

    We assume each data entry has an index list, and a typical task is to rank the list and
    determine the top index item for the data entry.

    Each item in the index must have a unique id within the index list
    to uniquely identify the index item within the list;
    the index item id does not require global uniqueness.

    Attributes:
        requires_flat_data:
            True if this featurizer requires the data index being exploded,
                i.e. there might be multiple data entries with the same data id,
                but they contain difference index item with different index item id from the index;
            False if each data entry has unique data id, and there is a list field containing
                all index items for the data entry.

    """

    requires_flat_data = attrib(type=bool, default=True)
    requires_non_flat_data = attrib(type=bool, default=False)

    def _get_features(
            self, data,
            data_id_field_name: str,
            feature_id_field_names: List[str],
            index_list_field_name: str = None,
            index_item_id_field_name: str = None,
            is_index_only_data=False,
            **kwargs
    ):
        raise NotImplementedError

    def get_features(
            self, data,
            data_id_field_name: str,
            feature_id_field_names: List[str],
            index_list_field_name: str = None,
            index_item_id_field_name: str = None,
            is_index_only_data=False,
            **kwargs
    ):
        """
        Gets features from the indexed data.

        Args:
            data: the data to extract features from.
            data_id_field_name: the field name of data id in `data`.
            feature_id_field_names: the fields names in the generated feature data
                to uniquely identify each feature entry.
            index_list_field_name: the field name of the index field.
            index_item_id_field_name: the field name for the index item id.
            is_index_only_data: True if the `data` contains index list only;
                in this case, `data` does not contain non-index data,
                and the data id identifies each index list,
                and features dependent on the non-index data will not be available.

        Returns: feature data extracted from the provided `data`;
            it includes feature id fields, and the feature set fields.

        """
        return super().get_features(
            data=data,
            data_id_field_name=data_id_field_name,
            feature_id_field_names=feature_id_field_names,
            index_list_field_name=index_list_field_name,
            index_item_id_field_name=index_item_id_field_name,
            is_index_only_data=is_index_only_data,
            **kwargs
        )


@attrs(slots=False)
class SparkFeaturizer(Featurizer, ABC):
    _spark = attrib(type=SparkSession, default=None)

    def _select_feature_fields(self, data: DataFrame, *field_names):
        return data.select(*field_names)


@attrs(slots=False)
class SparkIndexedDataFeaturizer(SparkFeaturizer, IndexedDataFeaturizer, ABC):
    pass
