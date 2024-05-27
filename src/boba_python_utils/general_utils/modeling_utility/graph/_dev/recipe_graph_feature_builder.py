import random
import shutil
from functools import partial
from os import path
from typing import List, Mapping, Tuple
from typing import Union

from attr import attrib, attrs
from pyspark.sql import DataFrame

import boba_python_utils.general_utils.modeling_utility.dataset.constants
import boba_python_utils.general_utils.modeling_utility.feature_building.constants
import boba_python_utils.spark_utils.spark_functions as F
import boba_python_utils.spark_utils.data_transform
import boba_python_utils.spark_utils as sparku
import boba_python_utils.production_utils.pdfs.constants as c
import boba_python_utils.spark_utils.struct_operations
import utix.consolex as consolex
import utix.ioex as ioex
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.general_utils.general import make_list_
from boba_python_utils.production_utils.pdfs.testsets.opportunity_and_guardrail_data import add_label_to_customer_history_index
from hdfs_alpha.l2_ranker_p1_v1.common import PdfsL2Experiment
from pdfs_offline.l2.common.feature_builders.pdfs_feature_builder import PdfsFeatureBuilder
from pdfs_offline.l2.common.feature_selection import FeatureSelection
from utix.sparku import parallel_compute


def grouped_feat_extractor_from_partition(partition, feature_list):
    feats, labels, group_sizes, data_ids, group_item_ids = [], [], [], [], []
    for row in partition:
        group = row['group']
        if len(group) > 500:
            raise ValueError('unlikely large group found')
        data_ids.append(row[KEY_DATA_ID])
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
                labels.append(item[boba_python_utils.general_utils.modeling_utility.feature_building.constants.KEY_LABEL])
                group_item_ids.append(item[c.KEY_INDEX_ITEM_ID])
        else:
            for item in group:
                feats.append([item[feat_name] for feat_name in feature_list])
                labels.append(item[boba_python_utils.general_utils.modeling_utility.feature_building.constants.KEY_LABEL])
                group_item_ids.append(item[c.KEY_INDEX_ITEM_ID])
    return feats, labels, group_sizes, data_ids, group_item_ids


def combined_lists(result, combined_result):
    if combined_result is None:
        combined_result = result
    else:
        for x, y in zip(result, combined_result):
            y.extend(x)
    return combined_result


def get_rewrite_from_customer_history_index_item(item):
    return item[c.KEY_REPLACED_REQUEST] if (c.KEY_REPLACED_REQUEST in item and item[c.KEY_REPLACED_REQUEST]) else item[c.KEY_REQUEST]


@attrs
class PdfsOfflineFeatureBuilder(PdfsFeatureBuilder):
    experiment = attrib(type=PdfsL2Experiment, default=None)
    feature_selections = attrib(type=List[FeatureSelection], default=None)
    _temp_dir = attrib(type=str, default=path.abspath('~/_pdfs_exp_tmp'))
    _label_col_name = attrib(type=str, default=boba_python_utils.general_utils.modeling_utility.feature_building.constants.KEY_LABEL)
    _top_selection_score = attrib(type=str, default=None)
    _top_selection = attrib(type=int, default=10)
    _top_selection_reverse = attrib(type=bool, default=True)

    def _get_path_suffix(self):
        if self._top_selection is None:
            return ''
        else:
            return (
                f'-top{self._top_selection}-{self._top_selection_score}'
                if self._top_selection_reverse
                else f'-top{self._top_selection}-{self._top_selection_score}_asc'
            )

    def get_output_path(self, featurizer_name) -> str:
        return self.experiment.s3_l2_features_path(featurizer_name) + self._get_path_suffix()

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self._top_selection <= 0:
            self._top_selection = None
        self.feature_list = self.experiment.get_l2_feature_list()
        if not isinstance(self.feature_list, (list, tuple)):
            raise ValueError(f"the feature list must be a list of feature names; got {self.feature_list}")

        if self.feature_selections is None:
            self.feature_selections = [FeatureSelection.all_features()]
        else:
            self.feature_selections = make_list_(self.feature_selections)

        if not self._extra_fields_to_save:
            self._extra_fields_to_save = [c.KEY_REQUEST, boba_python_utils.general_utils.modeling_utility.feature_building.constants.KEY_LABEL]
        else:
            self._extra_fields_to_save = list(self._extra_fields_to_save)
            if c.KEY_REQUEST not in self._extra_fields_to_save:
                self._extra_fields_to_save.append(c.KEY_REQUEST)
            if boba_python_utils.general_utils.modeling_utility.feature_building.constants.KEY_LABEL not in self._extra_fields_to_save:
                self._extra_fields_to_save.append(boba_python_utils.general_utils.modeling_utility.feature_building.constants.KEY_LABEL)

    # def _pre_process_before_data_flatten(self, df_data: DataFrame) -> DataFrame:
    # KEY_SCORE_BM25_INTERNAL_USE = f'{INTERNAL_USE_COL_NAME_PREFIX}{c.KEY_SCORE_BM25}'
    #
    # df_data_with_bm25_scores = sparku.cache__(
    #     df_data.withColumn(
    #         KEY_SCORE_BM25_INTERNAL_USE,
    #         F.get_bm_25_scores_udf(
    #             c.KEY_REQUEST_SECOND,
    #             c.KEY_HISTORY,
    #             doc_field_name_or_extractor=get_rewrite_from_customer_history_index_item,
    #             ignore_empty_doc=False
    #         )),
    #     name='df_data_with_bm25_scores',
    #     cache_options=self._cache_options,
    #     unpersist=df_data
    # )
    #
    # df_data_with_bm25_scores_merged = sparku.cache__(
    #     sparku.merge_two_arrays(
    #         df_data_with_bm25_scores,
    #         c.KEY_HISTORY,
    #         KEY_SCORE_BM25_INTERNAL_USE,
    #         output_arr_col_name=c.KEY_HISTORY,
    #         select_arr2_field_names=c.KEY_SCORE_BM25),
    #     name='df_data_with_bm25_scores_merged'
    # )
    #
    # return df_data_with_bm25_scores_merged.drop(KEY_SCORE_BM25_INTERNAL_USE)

    def pre_process_after_data_flatten(self, df_data_flat: DataFrame) -> DataFrame:
        if self._label_col_name not in df_data_flat.columns:
            df_data_flat_labeled = sparku.cache__(
                add_label_to_customer_history_index(
                    df_data_flat,
                    truth_request_key=c.KEY_REQUEST_SECOND,
                    truth_hypothesis_key=c.KEY_NLU_HYPOTHESIS_SECOND
                ),
                name='df_data_flat_labeled',
                cache_option=self._cache_option,
                unpersist=df_data_flat
            )
        else:
            df_data_flat_labeled = df_data_flat

        if self._top_selection_score is None:
            return df_data_flat_labeled
        else:
            return sparku.cache__(
                sparku.top_from_each_group(
                    df_data_flat_labeled,
                    group_cols=[KEY_DATA_ID],
                    order_cols=[F.col(self._top_selection_score).desc() if self._top_selection_reverse else F.col(self._top_selection_score)],
                    top=self._top_selection
                ),
                cache_option=self._cache_option,
                name='df_data_flat_labeled_top',
                unpersist=df_data_flat_labeled
            )

    def build_features(self, df_data: DataFrame = None, force=False, return_dataframe=False, is_index_data=False):
        if df_data is None:
            df_data = self.experiment.input_path_from_datasets()

        base_feature_builder_result = super().build_features(
            data=df_data, force=force, return_dataframe=True, is_index_data=is_index_data
        )

        df_feats_with_labels = base_feature_builder_result.get_feature_dataframe(self._spark)
        feats_col_name = base_feature_builder_result.features_field_name

        df_grouped_feats_with_labels = sparku.cache__(
            boba_python_utils.spark_utils.data_transform.fold(
                boba_python_utils.spark_utils.struct_operations.unfold_struct(
                    df_feats_with_labels.select(
                        self._data_id_col_name,
                        self._hist_id_col_name,
                        c.KEY_REQUEST,
                        boba_python_utils.general_utils.modeling_utility.feature_building.constants.KEY_LABEL,
                        feats_col_name
                    ),
                    struct_colname=feats_col_name
                ),
                group_cols=[KEY_DATA_ID],
                fold_colname='group'
            ).orderBy(F.rand()),
            name='df_grouped_feats_with_labels',
            unpersist=df_feats_with_labels
        )

        feature_list = self.feature_list
        feature_selections = self.feature_selections

        path_suffix = self._get_path_suffix()
        for feature_selection in feature_selections:
            # allows multiple feature selections;
            # output a feature dump for each feature selection
            output_root_tmp = ioex.pathex.ensure_dir_existence(
                path.join(self._temp_dir, self._name, feature_selection.name + path_suffix),
                clear_dir=True
            )

            selected_features: Union[Mapping, List, Tuple] = (
                feature_selection.select_features_from_list(feature_list)
            )

            hprint_message(feature_selection.name, f'{len(selected_features)} selected features')

            ioex.write_all_lines(
                selected_features.keys() if isinstance(selected_features, Mapping) else selected_features,
                path.join(output_root_tmp, 'feature_list.txt')
            )
            extracted_feature_data = parallel_compute(
                df=df_grouped_feats_with_labels,
                partition_transform_func=partial(grouped_feat_extractor_from_partition, feature_list=selected_features),
                combine_partition_transform_func=combined_lists if self._debug_mode else None,
                repartition=1000,
                file_based_combine=True,
                output_result_to_files=True,
                output_path=path.join(output_root_tmp, 'feature_data')
            )

            output_root = self.experiment.local_l2_features_path(feature_selection.name + path_suffix)
            if path.exists(output_root):
                shutil.rmtree(output_root)
            consolex.hprint_message('move dumped numpy features to', output_root)
            shutil.move(output_root_tmp, output_root)

            if self._debug_mode:
                feats, labels, group_sizes, data_ids, group_item_ids = extracted_feature_data
                if sum(group_sizes) != len(feats):
                    raise ValueError()

                for _ in range(20):
                    i = random.randint(0, len(data_ids))
                    data_id = data_ids[i]
                    consolex.hprint_pairs(('testing', i), (KEY_DATA_ID, data_id))
                    data = sparku.where(df_grouped_feats_with_labels, {KEY_DATA_ID: data_id}).head()
                    group_size = group_sizes[i]
                    if group_size != len(data['group']):
                        raise ValueError()
                    j = sum(group_sizes[:i])
                    for data_feat, data_label, data_item_id, data_item in zip(feats[j: j + group_size], labels[j: j + group_size], group_item_ids[j: j + group_size], data['group']):
                        if data_feat != [data_item[feat_name] for feat_name in selected_features] or data_label != data_item[boba_python_utils.general_utils.modeling_utility.feature_building.constants.KEY_LABEL] or data_item_id != data_item[c.KEY_INDEX_ITEM_ID]:
                            raise ValueError()
