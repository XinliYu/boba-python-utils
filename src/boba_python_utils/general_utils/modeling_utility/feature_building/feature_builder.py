import random
import shutil
from abc import ABC
from os import path
from typing import List, Iterable, Callable, Mapping, Tuple
from typing import Union

from attr import attrs, attrib
from pyspark.sql import DataFrame

from boba_python_utils.general_utils.common_objects.debuggable import Debuggable
from boba_python_utils.common_utils.typing_helper import make_list_
from boba_python_utils.general_utils.console_util import hprint_message, hprint, hprint_pairs
from boba_python_utils.general_utils.experiment_utility.experiment_base import ExperimentBase
from boba_python_utils.io_utils.text_io import write_all_lines
from boba_python_utils.spark_utils.specialized.indexed_data.constants import KEY_DATA_ID, KEY_INDEX_ITEM_ID, KEY_INDEX_LIST
from boba_python_utils.general_utils.modeling_utility.feature_building.constants import KEY_MERGED_FEATURES, KEY_LABEL, KEY_FEATURE_GROUP, FILENAME_FEATURE_LIST, DIRNAME_FEATURE_DATA
from boba_python_utils.general_utils.modeling_utility.feature_building.feature_selection import FeatureSelection
from boba_python_utils.general_utils.modeling_utility.feature_building.featurizer import IndexedDataFeaturizer
from boba_python_utils.path_utils.common import ensure_dir_existence
from boba_python_utils.string_utils.prefix_suffix import add_suffix


@attrs(slots=True)
class FeatureBuilderResult:
    features_field_name = attrib(type=str)
    feature_data_or_path = attrib(type=Union[str, DataFrame])


@attrs(slots=False)
class FeatureBuilder(Debuggable, ABC):
    featurizers = attrib(type=List[IndexedDataFeaturizer], default=None)
    data_id_field_name = attrib(type=str, default=KEY_DATA_ID)
    feature_id_field_names = attrib(type=List[str], default=None)

    save_individual_featurizer_results = attrib(type=bool, default=True)
    extra_feature_group_data_fields = attrib(type=List[str], default=[])
    num_files_for_features = attrib(type=int, default=None)
    num_files_for_data = attrib(type=int, default=500)

    def __attrs_post_init__(self):
        self._num_feature_sets = sum(featurizer.num_feature_sets for featurizer in self.featurizers)

    # region abstract methods

    # region data transformation methods
    def _pre_process_data(self, data):
        raise NotImplementedError

    def _get_featurizer_input_data(self, pre_processed_data, featurizer):
        raise NotImplementedError

    def _merge_features(
            self,
            all_feature_names: List[str],
            features_collection: List,
            merged_features_name: str
    ):
        raise NotImplementedError

    def _final_transform_with_features_and_data(
            self,
            feature_data,
            feature_field_name: str,
            pre_processed_data,
            **kwargs
    ):
        raise NotImplementedError

    # endregion

    # region I/O methods
    def _get_output_path(self, features_name) -> str:
        raise NotImplementedError

    def _load_data(self, data_source, cache: bool, data_name: str = None, **kwargs):
        raise NotImplementedError

    def _write_data(self, data, output_path: str, num_files: int, **kwargs):
        raise NotImplementedError

    def _unload_data(self, data):
        raise NotImplementedError

    def _dump_features(
            self,
            output_path,
            feature_aggregation_method,
            feature_read_method,
            feature_write_method,
            feature_aggregation_name: str,
            logging_method=None,
            force: bool = False,
            save_aggregation: bool = True,
            **kwargs
    ):
        raise NotImplementedError

    # endregion

    # region id-related methods
    def _data_has_ids(self, data):
        raise NotImplementedError

    def _add_data_ids(self, data, overwrite_existing_ids: bool = False):
        raise NotImplementedError

    # endregion

    # region misc methods
    def _get_num_dimensions(self, df_feats):
        raise NotImplementedError

    def _data_has_field(self, data, field_name: str) -> bool:
        raise NotImplementedError

    def _get_feature_name_suffix(self) -> str:
        return ''

    # endregion

    # endregion

    def add_ids_to_data(self, data, overwrite_existing_ids: bool = False):
        return self._add_data_ids(data, overwrite_existing_ids)

    def dump_features(
            self,
            output_path,
            feature_aggregation_method,
            feature_aggregation_name: str,
            logging_method=None,
            force: bool = False,
            save_aggregation: bool = True,
            **kwargs
    ):
        def _feature_read_method(input_path: str) -> DataFrame:
            return self._load_data(
                data_source=input_path,
                cache=False
            )

        def _feature_write_method(features, output_path):
            hprint_message(
                f'{path.basename(output_path)} dimensions',
                self._get_num_dimensions(features),
            )
            self._write_data(
                data=features,
                output_path=output_path,
                num_files=self.num_files_for_features
            )

        return self._dump_features(
            output_path=output_path,
            feature_read_method=_feature_read_method,
            feature_write_method=_feature_write_method,
            feature_aggregation_method=feature_aggregation_method,
            feature_aggregation_name=feature_aggregation_name,
            logging_method=logging_method,
            force=force,
            save_aggregation=save_aggregation,
            **kwargs
        )

    def build_features(
            self,
            data,
            force=False,
            return_output_path=True,
            merged_features_name: str = KEY_MERGED_FEATURES,
            **kwargs
    ) -> FeatureBuilderResult:

        if not self.feature_id_field_names:
            self.feature_id_field_names = [self.data_id_field_name]

        # region STEP1: define needed parameters for `_dump_features` method
        features_name = (
            self.featurizers[0].featurizer_name
            if self._num_feature_sets == 1
            else merged_features_name
        )
        features_output_path = self._get_output_path(
            add_suffix(features_name, suffix=self._get_feature_name_suffix(), sep='-')
        )

        # endregion

        # region STEP2: the main logic is defined in this private function `_feature_aggregation`.
        def _feature_aggregation():
            nonlocal data

            # region STEP1: load data and add id fields
            _df_data = data
            data = self._load_data(data_source=data, cache=True)

            if not self._data_has_ids(data):
                # adds data id to each data entry there is no existing data id field
                if isinstance(_df_data, str):
                    data_path = _df_data
                    data = self.add_ids_to_data(data, overwrite_existing_ids=False)
                    self._write_data(
                        data=data,
                        output_path=data_path,
                        num_files=self.num_files_for_data
                    )

                    self._unload_data(data)
                    data = self._load_data(data_source=data_path, cache=True)
                else:
                    raise ValueError("Required id fields are not found in the input data.")
            # endregion

            # region STEP2: data pre-processing
            pre_processed_data = self._pre_process_data(data)
            # endregion

            # region STEP3: run featurizers to get features
            features_collection = []
            all_feature_names = []

            def feature_aggregation_method():
                try:
                    featurizer_input_data = self._get_featurizer_input_data(
                        pre_processed_data, featurizer
                    )
                except NotImplementedError:
                    featurizer_input_data = pre_processed_data

                return featurizer.get_features(
                    data=featurizer_input_data,
                    data_id_field_name=self.data_id_field_name,
                    feature_id_field_names=self.feature_id_field_names,
                    **kwargs
                )

            if self._num_feature_sets == 1:
                featurizer = self.featurizers[0]
                merged_features = feature_aggregation_method()
                feature_field_name = featurizer.feature_set_names[0]
            else:
                for featurizer in self.featurizers:
                    featurizer_name = featurizer.featurizer_name
                    feature_data_output_path = self._get_output_path(
                        add_suffix(featurizer_name, suffix=self._get_feature_name_suffix(), sep='-')
                    )

                    features = self.dump_features(
                        output_path=feature_data_output_path,
                        feature_aggregation_method=feature_aggregation_method,
                        feature_aggregation_name=featurizer_name,
                        logging_method=None,
                        force=force,
                        save_aggregation=self.save_individual_featurizer_results
                    )
                    features_collection.append(features)
                    all_feature_names.extend(featurizer.feature_set_names)

                merged_features = self._merge_features(
                    all_feature_names=all_feature_names,
                    features_collection=features_collection,
                    merged_features_name=merged_features_name
                )
                feature_field_name = merged_features_name

            # endregion

            # region STEP4: join features with data and return
            return self._final_transform_with_features_and_data(
                feature_data=merged_features,
                feature_field_name=feature_field_name,
                pre_processed_data=pre_processed_data,
                **kwargs
            )
            # endregion

        # endregion

        # region STEP3: execute feature dumping
        data_joint_with_features = self.dump_features(
            output_path=features_output_path,
            feature_aggregation_method=_feature_aggregation,
            feature_aggregation_name=features_name,
            logging_method=None,
            force=force,
            save_aggregation=True
        )
        # endregion

        return FeatureBuilderResult(
            features_field_name=features_name,
            feature_data_or_path=(
                features_output_path
                if return_output_path
                else data_joint_with_features
            )
        )


@attrs(slots=False)
class IndexedDataFeatureBuilder(FeatureBuilder, ABC):
    index_item_id_field_name = attrib(type=str, default=KEY_INDEX_ITEM_ID)
    index_list_field_name = attrib(type=str, default=KEY_INDEX_LIST)
    is_index_item_id_unique = attrib(type=bool, default=False)

    def __attrs_post_init__(self):
        super(IndexedDataFeatureBuilder, self).__attrs_post_init__()
        self._num_featurizers_requires_flat_data = sum(featurizer.requires_flat_data for featurizer in self.featurizers)
        self._num_featurizers_not_requires_flat_data = len(self.featurizers) - self._num_featurizers_requires_flat_data
        self.expects_data_to_have_index = bool(self.index_item_id_field_name) and bool(self.index_list_field_name)

    # region abstract methods

    # region data transformation methods
    def _pre_process_before_data_flatten(self, data):
        pass

    def _pre_process_after_data_flatten(self, data_flat):
        raise NotImplementedError

    def _get_flattened_data(self, data):
        raise NotImplementedError

    # endregion

    # region id-related methods
    def _add_index_item_ids(self, data, overwrite_existing_ids: bool = False):
        raise NotImplementedError

    # endregion

    # region misc methods
    def _is_flat_data(self, data) -> bool:
        raise NotImplementedError

    # endregion

    # endregion

    def expects_data_to_have_index(self) -> bool:
        return bool(self.index_item_id_field_name) and bool(self.index_list_field_name)

    def add_ids_to_data(self, data, overwrite_existing_ids: bool = False):
        data = super(IndexedDataFeatureBuilder, self).add_ids_to_data(
            data=data,
            overwrite_existing_ids=overwrite_existing_ids
        )
        if self.expects_data_to_have_index:
            data = self._add_index_item_ids(
                data=data,
                overwrite_existing_ids=overwrite_existing_ids
            )
        return data

    def _pre_process_data(self, data):
        # 1) pre-process data by `_pre_process_before_data_flatten` before any data flattening;
        # 2) tries to flatten the data if necessary by calling `_get_flattened_data`;
        # 3) pre-process data by `_pre_process_after_data_flatten` after any data flattening.
        data_processed = self._pre_process_before_data_flatten(data)
        if data_processed is None:
            data_processed = data
            pre_process_implemented = False
        else:
            pre_process_implemented = True

        if pre_process_implemented and (data is not data_processed):
            self._unload_data(data)

        if self._num_featurizers_requires_flat_data > 0:
            if self._is_flat_data(data_processed):
                # if the processed data is already flat,
                # then we assume the featurizers work with the flat data
                # regardless of their `requires_flat_data` attribute
                data_flat = data_processed
            else:
                data_flat = self._get_flattened_data(data_processed)
                if self._num_featurizers_not_requires_flat_data == 0:
                    self._unload_data(data_processed)
            try:
                data_flat_processed = self._pre_process_after_data_flatten(data_flat)
                pre_process_implemented = True
            except NotImplementedError:
                data_flat_processed = data_flat
                pre_process_implemented = False

            if pre_process_implemented and (data_flat is not data_flat_processed):
                self._unload_data(data_flat)
        else:
            data_flat_processed = None

        return data_processed, data_flat_processed

    def _get_featurizer_input_data(self, pre_processed_data, featurizer):
        if featurizer.requires_flat_data and featurizer.requires_non_flat_data:
            return pre_processed_data
        elif featurizer.requires_non_flat_data:
            return pre_processed_data[0]
        elif featurizer.requires_flat_data:
            return pre_processed_data[1]
        else:
            raise ValueError()

    def build_features(
            self,
            data,
            force=False,
            return_output_path=True,
            merged_features_name: str = KEY_MERGED_FEATURES,
            is_index_only_data: bool = False
    ) -> FeatureBuilderResult:
        if not self.feature_id_field_names:
            if self.expects_data_to_have_index:
                if self.is_index_item_id_unique:
                    self.feature_id_field_names = [self.index_item_id_field_name]
                else:
                    self.feature_id_field_names = [self.data_id_field_name, self.index_item_id_field_name]
            else:
                self.feature_id_field_names = [self.data_id_field_name]

        return super(IndexedDataFeatureBuilder, self).build_features(
            data=data,
            force=force,
            return_output_path=return_output_path,
            merged_features_name=merged_features_name,
            index_list_field_name=self.index_list_field_name,
            index_item_id_field_name=self.index_item_id_field_name,
            is_index_only_data=is_index_only_data
        )


@attrs(slots=False)
class OfflineExperimentIndexedDataFeatureBuilder(IndexedDataFeatureBuilder, ABC):
    """
    Index feature builder for offline experiments. Supports generating binary feature files like
    numpy arrays or tensors for immediate use with modeling frameworks. Supports feature selection.

    """
    experiment = attrib(type=ExperimentBase, default=None)
    feature_selections = attrib(type=List[FeatureSelection], default=None)

    top_selection_feature_field_name = attrib(type=str, default=None)
    top_selection_size = attrib(type=int, default=10)
    top_selection_reverse = attrib(type=bool, default=True)
    shuffle_features = attrib(type=bool, default=True)
    shuffle_feature_groups = attrib(type=bool, default=True)

    label_field_name = attrib(type=str, default=KEY_LABEL)
    feature_group_field_name = attrib(type=str, default=KEY_FEATURE_GROUP)
    extra_feature_group_data_fields = attrib(type=Iterable[str], default=None)
    _tmp_local_dir = attrib(type=str, default=path.abspath(path.expanduser('~/_tmp')))

    _exists_dir_path_method = attrib(type=Callable, default=None)
    _put_dir_method = attrib(type=Callable, default=None)
    _get_dir_method = attrib(type=Callable, default=None)

    feature_list_file_name = attrib(type=str, default=FILENAME_FEATURE_LIST)

    # region abstract methods

    def _add_labels_to_flat_data(self, flat_data, label_field_name: str):
        raise NotImplementedError

    def _select_top_from_labeled_flat_data(
            self,
            labeled_flat_data,
            top_selection_feature_field_name: str,
            top_selection_size: int,
            top_selection_reverse: bool
    ):
        raise NotImplementedError

    def get_feature_list(self):
        raise NotImplementedError

    def _group_feature_data(
            self,
            feature_data_with_labels,
            features_field_name,
            group_id_field_name,
            item_id_field_name,
            extra_feature_group_data_fields,
            output_feature_group_field_name,
            shuffle_features,
            shuffle_feature_groups
    ):
        """
        Implements this method to organize features into groups based on group id.
        """
        raise NotImplementedError

    def get_output_path_local_for_offline_feature_files(
            self,
            feature_selection_name: str
    ) -> str:
        raise NotImplementedError

    def get_output_path_for_offline_feature_files(
            self,
            feature_selection_name: str
    ) -> str:
        raise NotImplementedError

    def _build_offline_experiment_feature_files(
            self,
            grouped_feature_data_with_labels,
            selected_features,
            output_path: str,

    ):
        raise NotImplementedError

    def _get_feature_group_data_by_id(self, grouped_feature_data_with_labels, data_id):
        raise NotImplementedError

    # endregion

    # region base implementation overrides
    def __attrs_post_init__(self):
        super(OfflineExperimentIndexedDataFeatureBuilder, self).__attrs_post_init__()

        self._name = self.experiment.name
        if self.top_selection_size <= 0:
            self.top_selection_size = None

        self.feature_list = self.get_feature_list()

        if not isinstance(self.feature_list, (list, tuple)):
            raise ValueError(
                f"the feature list must be a list of feature names; "
                f"got {self.feature_list}"
            )

        if not self.extra_feature_group_data_fields:
            self.extra_feature_group_data_fields = [self.label_field_name]
        else:
            self.extra_feature_group_data_fields = list(self.extra_feature_group_data_fields)
            if self.label_field_name not in self.extra_feature_group_data_fields:
                self.extra_feature_group_data_fields.append(self.label_field_name)

        if self.feature_selections is None:
            self.feature_selections = [FeatureSelection.all_features()]
        else:
            self.feature_selections = make_list_(self.feature_selections)

    def _pre_process_after_data_flatten(self, flat_data):
        """
        Computes label column if it is not available,
        and selects top candidate features from the customer history
        if `top_selection_score_colname` is specified.

        """

        if self._data_has_field(flat_data, self.label_field_name):
            labeled_flat_data = flat_data
        else:
            labeled_flat_data = self._add_labels_to_flat_data(flat_data, self.label_field_name)

        if self.top_selection_feature_field_name is None:
            return labeled_flat_data
        else:
            return self._select_top_from_labeled_flat_data(
                labeled_flat_data,
                top_selection_feature_field_name=self.top_selection_feature_field_name,
                top_selection_size=self.top_selection_size,
                top_selection_reverse=self.top_selection_reverse
            )

    # endregion

    def _get_feature_name_suffix(self):
        if self.top_selection_size is None:
            return ''
        else:
            return (
                f'-top{self.top_selection_size}-{self.top_selection_feature_field_name}'
                if self.top_selection_reverse
                else f'-top{self.top_selection_size}-{self.top_selection_feature_field_name}_asc'
            )

    def _verify_offline_experiment_feature_data(
            self,
            grouped_feature_data_with_labels,
            offline_experiment_feature_data,
            selected_features
    ):
        hprint_message('debug mode', self._debug_mode)
        feats, labels, group_sizes, data_ids, group_item_ids = offline_experiment_feature_data
        if sum(group_sizes) != len(feats):
            raise ValueError("the sum of group sizes must "
                             "be equal to the number of feature vectors")

        for _ in range(20):
            i = random.randint(0, len(data_ids))
            data_id = data_ids[i]
            hprint_pairs(('testing', i), (self.data_id_field_name, data_id))
            feature_group_data = self._get_feature_group_data_by_id(
                grouped_feature_data_with_labels,
                data_id
            )
            group_size = group_sizes[i]
            if group_size != len(feature_group_data[self.feature_group_field_name]):
                raise ValueError(f"the group size is inconsistent "
                                 f"for feature {i} of id {data_id}")
            j = sum(group_sizes[:i])
            for data_feat, data_label, data_item_id, data_item in zip(
                    feats[j: j + group_size],
                    labels[j: j + group_size],
                    group_item_ids[j: j + group_size],
                    feature_group_data['group']
            ):
                if (
                        data_feat != [data_item[feat_name] for feat_name in selected_features] or
                        data_label != data_item[self.label_field_name] or
                        data_item_id != data_item[self.index_item_id_field_name]
                ):
                    raise ValueError("the group sizes do not align with the features")

    def build_features(
            self,
            data: DataFrame = None,
            force=False,
            force_local=False,
            return_output_path=True,
            merged_features_name: str = 'merged_features',
            is_index_only_data: bool = False
    ):
        # region STEP1: filter out feature selections that already exist
        feature_selections = []
        path_suffix = self._get_feature_name_suffix()
        for feature_selection in self.feature_selections:
            # go through all feature selections;
            # if a feature selection already exists in local,
            #    then we skip the feature building;
            # otherwise if a feature selection already exists in s3,
            #    then we download it and skip the feature building;
            # otherwise we put it in `feature_selections`
            #    and will perform feature building for them.
            feature_selection_name = add_suffix(
                feature_selection.name, suffix=path_suffix, sep='-'
            )
            output_path_local = self.get_output_path_local_for_offline_feature_files(
                feature_selection_name
            )
            output_path_local_feature_list_file = path.join(
                output_path_local, FILENAME_FEATURE_LIST
            )
            output_path_local_feature_data_dir = path.join(
                output_path_local, DIRNAME_FEATURE_DATA
            )
            output_path = self.get_output_path_for_offline_feature_files(feature_selection_name)

            if path.exists(output_path_local):
                if (
                        path.exists(output_path_local_feature_list_file) and
                        path.exists(output_path_local_feature_data_dir)
                ):
                    if force_local:
                        shutil.rmtree(output_path_local)
                    else:
                        hprint_message('feature output path already exists', output_path_local)
                        hprint('to overwrite existing feature files, set `force_local` as True')

                        if not self._exists_dir_path_method(output_path):
                            self._put_dir_method(
                                output_path_local,
                                output_path
                            )
                        continue
                else:
                    shutil.rmtree(output_path_local)
            elif not force_local:
                if self._exists_dir_path_method(output_path):
                    hprint_message('downloading features from', output_path)
                    self._get_dir_method(
                        output_path,
                        output_path_local
                    )
                    continue
            feature_selections.append(feature_selection)

        hprint_message(
            'num feature selections already exist',
            len(self.feature_selections) - len(feature_selections)
        )
        hprint_message(
            'num feature selections to build',
            len(feature_selections)
        )
        if not feature_selections:
            return
        # endregion

        # region STEP2: builds feature data
        if data is None:
            data = self.experiment.input_path_from_datasets()

        base_feature_builder_result = super().build_features(
            data=data,
            force=force,
            return_output_path=return_output_path,
            merged_features_name=merged_features_name,
            is_index_only_data=is_index_only_data
        )

        feature_data_with_labels = self._load_data(
            data_source=base_feature_builder_result.feature_data_or_path,
            cache=False,
            data_name='feature_data_with_labels'
        )
        # endregion

        # region STEP3: groups feature data by data id
        grouped_feature_data_with_labels = self._group_feature_data(
            feature_data_with_labels=feature_data_with_labels,
            features_field_name=base_feature_builder_result.features_field_name,
            group_id_field_name=self.data_id_field_name,
            item_id_field_name=self.index_item_id_field_name,
            extra_feature_group_data_fields=self.extra_feature_group_data_fields,
            output_feature_group_field_name=self.feature_group_field_name,
            shuffle_features=self.shuffle_features,
            shuffle_feature_groups=self.shuffle_feature_groups
        )
        # endregion

        # region STEP4: feature selection
        feature_list = self.feature_list
        path_suffix = self._get_feature_name_suffix()
        for feature_selection in feature_selections:
            # allows multiple feature selections;
            # output a feature dump for each feature selection
            feature_selection_name = add_suffix(
                feature_selection.name, suffix=path_suffix, sep='-'
            )
            output_path_local = self.get_output_path_local_for_offline_feature_files(
                feature_selection_name
            )
            output_path = self.get_output_path_for_offline_feature_files(feature_selection_name)

            if path.exists(output_path_local):
                shutil.rmtree(output_path_local)

            output_path_local_tmp = ensure_dir_existence(
                path.join(
                    self._tmp_local_dir,
                    self.experiment.name,
                    feature_selection_name
                ),
                clear_dir=True
            )

            selected_features: Union[Mapping, List, Tuple] = (
                feature_selection.select_features_from_list(feature_list)
            )

            hprint_message(
                feature_selection.name,
                f'{len(selected_features)} selected features'
            )

            write_all_lines(
                (
                    selected_features.keys()
                    if isinstance(selected_features, Mapping)
                    else selected_features
                ),
                path.join(output_path_local_tmp, FILENAME_FEATURE_LIST)
            )

            offline_experiment_feature_data = self._build_offline_experiment_feature_files(
                grouped_feature_data_with_labels,
                selected_features=selected_features,
                output_path=path.join(output_path_local_tmp, DIRNAME_FEATURE_DATA)
            )

            if self._debug_mode:
                self._verify_offline_experiment_feature_data(
                    grouped_feature_data_with_labels,
                    offline_experiment_feature_data,
                    selected_features=selected_features
                )

            hprint_message('move dumped numpy features to', output_path_local)
            shutil.move(output_path_local_tmp, output_path_local)

            self._put_dir_method(output_path_local, output_path)
        # endregion
