from abc import ABC
from functools import partial
from typing import List, Mapping

from attr import attrs, attrib
from pyspark.sql import DataFrame
from pyspark.sql.types import Row

import boba_python_utils.spark_utils.spark_functions as F
import boba_python_utils.spark_utils.data_transform
import boba_python_utils.spark_utils as sparku
import boba_python_utils.spark_utils.struct_operations
from slab_aggregation.aggregators.base_aggregators import Aggregator
from boba_python_utils.spark_utils.specialized.indexed_data.data_id import spark_dataframe_has_id_columns, spark_dataframe_add_index_item_id, spark_dataframe_add_data_id
from boba_python_utils.general_utils.modeling_utility.feature_building.feature_builder import IndexedDataFeatureBuilder, FeatureBuilder, OfflineExperimentIndexedDataFeatureBuilder
from boba_python_utils.spark_utils.parallel_compute import parallel_compute
from boba_python_utils.production_utils.s3.existence import execute_spark_aggregation_if_not_exist_spark_success


@attrs(slots=False)
class SparkFeatureBuilder(Aggregator, FeatureBuilder):
    # region data transformation methods

    def __attrs_post_init__(self):
        Aggregator.__attrs_post_init__(self)
        FeatureBuilder.__attrs_post_init__(self)

    def _data_has_field(self, data: DataFrame, field_name: str) -> bool:
        return field_name in data.columns

    def _merge_features(
            self,
            all_feature_names: List[str],
            features_collection: List,
            merged_features_name: str
    ):
        feat_list = []
        joint_features = None
        for features in features_collection:
            for feat_set_field in features.schema:
                if feat_set_field.name in all_feature_names:
                    for feat_field in feat_set_field.dataType.fields:
                        feat_list.append(feat_field.name)

            features = features.drop(
                *(_col for _col in features.columns
                  if (_col not in all_feature_names and
                      _col not in self.feature_id_field_names)))

            if joint_features is None:
                joint_features = features
            else:
                joint_features = sparku.cache__(
                    joint_features.join(features, self.feature_id_field_names, how='full'),
                    name='df_merged_feats',
                    unpersist=(joint_features, features),
                )

        for feat_set_col in all_feature_names:
            if self._debug_mode:
                sparku.show_counts(joint_features, F.col(feat_set_col).isNull())
            joint_features = joint_features.where(F.col(feat_set_col).isNotNull())

        return sparku.merge_structs(
            df=joint_features,
            struct_field_names=all_feature_names,
            merged_col_name=merged_features_name,
            drop_input_struct_fields=True,
            default_values=None,
        )

    def _final_transform_with_features_and_data(
            self,
            feature_data,
            feature_field_name: str,
            pre_processed_data,
            **kwargs
    ):
        if self.extra_feature_group_data_fields:
            feature_data = pre_processed_data.select(
                *self.feature_id_field_names, *self.extra_feature_group_data_fields
            ).join(
                feature_data,
                self.feature_id_field_names
            )

        feature_data = sparku.cache__(
            feature_data,
            name='features dataframe',
            unpersist=feature_data
        )

        if self._debug_mode:
            sparku.show_counts(
                feature_data,
                sparku.any_sub_field_is_null(
                    feature_data, feature_field_name
                ).alias('has_null_features'),
            )

        return feature_data

    # endregion

    # region I/O methods
    def _load_data(self, data_source, cache: bool, data_name: str = None, **kwargs):
        data_name = data_name or str(data_source)

        if cache:
            return sparku.cache__(
                data_source,
                spark=self._spark,
                input_format=self._data_format,
                name=data_name,
                cache_option=self._cache_option,
                **kwargs
            )
        else:
            return sparku.solve_input(
                data_source,
                spark=self._spark,
                name=data_name,
                input_format=self._data_format,
                **kwargs
            )

    def _write_data(self, data: DataFrame, output_path: str, num_files: int, **kwargs):
        if not num_files:
            num_files = 100

        sparku.write_df(
            data,
            output_path=output_path,
            num_files=num_files,
            format=self._data_format,
            repartition=True,
            compress=True,
            show_counts=False,
        )

    def _unload_data(self, data: DataFrame):
        data.unpersist()

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
        return execute_spark_aggregation_if_not_exist_spark_success(
            s3path=output_path,
            spark_aggregation_method=feature_aggregation_method,
            spark_read_method=feature_read_method,
            spark_write_method=feature_write_method,
            spark_aggregation_name=feature_aggregation_name,
            log_message_method=None,
            force=force,
            save_aggregation=save_aggregation,
            return_dataframe=True,
        )

    # endregion

    # region id-related methods
    def _data_has_ids(self, data):
        return spark_dataframe_has_id_columns(
            df_data=data,
            data_id_col_name=self.data_id_field_name
        )

    def _add_data_ids(self, data, overwrite_existing_ids: bool = False):
        return spark_dataframe_add_data_id(
            df_data=data,
            data_id_colname=self.data_id_field_name,
            overwrite_existing_ids=overwrite_existing_ids
        )

    # endregion

    # region misc methods
    def _get_num_dimensions(self, features):
        excluded_fields = {
            self.data_id_field_name,
            *self.feature_id_field_names,
            *self.extra_feature_group_data_fields
        }
        return sum(
            len(features.schema[i].dataType)
            for i in range(len(features.columns))
            if features.schema[i].name not in excluded_fields
        )
    # endregion


@attrs(slots=False)
class SparkIndexedDataFeatureBuilder(SparkFeatureBuilder, IndexedDataFeatureBuilder):

    def __attrs_post_init__(self):
        IndexedDataFeatureBuilder.__attrs_post_init__(self)
        SparkFeatureBuilder.__attrs_post_init__(self)

    # region data transformation methods
    def _get_flattened_data(self, df_data: DataFrame):
        if self.index_list_field_name in df_data.columns:
            return sparku.cache__(
                sparku.explode_as_flat_columns(
                    df_data,
                    col_to_explode=self.index_list_field_name,
                    overwrite_exist_column=True
                ),
                name='flattened data',
                cache_option=self._cache_option
            )
        else:
            return df_data

    def _final_transform_with_features_and_data(
            self,
            feature_data,
            feature_field_name: str,
            pre_processed_data,
            **kwargs
    ):
        pre_processed_data = (pre_processed_data[1] or pre_processed_data[0])
        return SparkFeatureBuilder._final_transform_with_features_and_data(
            self,
            feature_data=feature_data,
            feature_field_name=feature_field_name,
            pre_processed_data=pre_processed_data,
            **kwargs
        )

    # endregion

    # region id-related methods
    def _data_has_ids(self, data):
        return spark_dataframe_has_id_columns(
            df_data=data,
            data_id_col_name=self.data_id_field_name,
            index_item_id_colname=self.index_item_id_field_name,
            index_list_colname=self.index_list_field_name
        )

    def _add_index_item_ids(self, data, overwrite_existing_ids: bool = False):
        return spark_dataframe_add_index_item_id(
            df_data=data,
            data_id_colname=self.data_id_field_name,
            index_item_id_colname=self.index_item_id_field_name,
            index_list_colname=self.index_list_field_name,
            overwrite_existing_ids=overwrite_existing_ids
        )

    # endregion

    # region misc methods
    def _is_flat_data(self, data) -> bool:
        return self.index_list_field_name not in data.columns
    # endregion


def combined_lists(result, combined_result):
    if combined_result is None:
        combined_result = result
    else:
        for x, y in zip(result, combined_result):
            y.extend(x)
    return combined_result


def grouped_feat_extractor_from_partition(
        partition,
        feature_list,
        data_id_colname,
        index_item_id_colname,
        label_colname,
        feature_group_colname,
        unlikely_large_group_size_threshold=500
):
    feats, labels, group_sizes, data_ids, group_item_ids = [], [], [], [], []
    for row in partition:
        group = row[feature_group_colname]
        if len(group) > unlikely_large_group_size_threshold:
            raise ValueError(f'unlikely large group with size {len(group)} found')
        data_ids.append(row[data_id_colname])
        group_sizes.append(len(group))
        if isinstance(feature_list, Mapping):
            # if the feature selection is a mapping,
            # then it is a mapping from feature names to "mask values";
            # in this case we mask the original feature values by "mask values" in the feature list
            for item in group:
                feats.append([
                    item[feat_name] if mask_feat_value is None
                    else (
                        None if mask_feat_value in ('none', 'null')
                        else mask_feat_value
                    )
                    for feat_name, mask_feat_value in feature_list.items()]
                )
                labels.append(item[label_colname])
                group_item_ids.append(item[index_item_id_colname])
        else:
            for item in group:
                feats.append([item[feat_name] for feat_name in feature_list])
                labels.append(item[label_colname])
                group_item_ids.append(item[index_item_id_colname])
    return feats, labels, group_sizes, data_ids, group_item_ids


@attrs(slots=False)
class SparkOfflineExperimentIndexedDataFeatureBuilder(
    SparkIndexedDataFeatureBuilder, OfflineExperimentIndexedDataFeatureBuilder, ABC
):
    _num_feature_dump_partitions = attrib(type=str, default=1000)

    def __attrs_post_init__(self):
        SparkIndexedDataFeatureBuilder.__attrs_post_init__(self)
        OfflineExperimentIndexedDataFeatureBuilder.__attrs_post_init__(self)

    def _select_top_from_labeled_flat_data(
            self,
            labeled_flat_data: DataFrame,
            top_selection_feature_field_name,
            top_selection_size,
            top_selection_reverse
    ):
        return sparku.cache__(
            sparku.top_from_each_group(
                labeled_flat_data,
                group_cols=[self.data_id_field_name],
                order_cols=[
                    F.col(top_selection_feature_field_name).desc()
                    if top_selection_reverse
                    else F.col(top_selection_feature_field_name)
                ],
                top=top_selection_size
            ),
            cache_option=self._cache_option,
            name="top selection from labeled flat data",
            unpersist=labeled_flat_data
        )

    def _group_feature_data(
            self,
            feature_data_with_labels: DataFrame,
            features_field_name,
            group_id_field_name,
            item_id_field_name,
            extra_feature_group_data_fields,
            output_feature_group_field_name,
            shuffle_features,
            shuffle_feature_groups
    ):
        feature_data_with_labels_unfolded = boba_python_utils.spark_utils.struct_operations.unfold_struct(
            feature_data_with_labels.select(
                group_id_field_name,
                item_id_field_name,
                *extra_feature_group_data_fields,
                features_field_name
            ),
            struct_colname=features_field_name
        )

        if shuffle_features:
            feature_data_with_labels_unfolded = feature_data_with_labels_unfolded.orderBy(F.rand())

        grouped_feature_data_with_labels = boba_python_utils.spark_utils.data_transform.fold(
            feature_data_with_labels_unfolded,
            group_cols=[group_id_field_name],
            fold_colname=output_feature_group_field_name
        )

        if shuffle_feature_groups:
            grouped_feature_data_with_labels = grouped_feature_data_with_labels.orderBy(F.rand())

        return sparku.cache__(
            grouped_feature_data_with_labels,
            name='grouped feature data with labels',
            cache_option=self._cache_option,
            unpersist=feature_data_with_labels
        )

    def _build_offline_experiment_feature_files(
            self,
            grouped_feature_data_with_labels: DataFrame,
            selected_features,
            output_path: str
    ):
        return parallel_compute(
            df=grouped_feature_data_with_labels,
            partition_transform_func=partial(
                grouped_feat_extractor_from_partition,
                feature_list=selected_features,
                data_id_colname=self.data_id_field_name,
                index_item_id_colname=self.index_item_id_field_name,
                label_colname=self.label_field_name,
                feature_group_colname=self.feature_group_field_name
            ),
            combine_partition_transform_func=combined_lists if self._debug_mode else None,
            repartition=self._num_feature_dump_partitions,
            file_based_combine=True,
            output_result_to_files=True,
            output_path=output_path
        )

    def _get_feature_group_data_by_id(
            self,
            grouped_feature_data_with_labels: DataFrame,
            data_id
    ) -> Row:
        return sparku.where(
            grouped_feature_data_with_labels, {self.data_id_field_name: data_id}
        ).head()
