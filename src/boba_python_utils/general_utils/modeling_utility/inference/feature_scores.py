import uuid
from os import path
from typing import Callable, Optional

from pyspark.sql import SparkSession

from boba_python_utils.common_utils.iter_helper import iter_, tqdm_wrap
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.io_utils.pickle_io import pickle_load
from boba_python_utils.io_utils.json_io import write_json
from boba_python_utils.general_utils.modeling_utility.dataset.constants import (
    KEY_DATA_ID, KEY_INDEX_ITEM_ID
)
from boba_python_utils.general_utils.modeling_utility.feature_building.constants import (
    KEY_LABEL, KEY_SCORE, KEY_RESULT
)
from boba_python_utils.path_utils.path_listing import get_paths_by_pattern
from boba_python_utils.path_utils.common import ensure_dir_existence
from boba_python_utils.spark_utils.data_loading import solve_input
from boba_python_utils.spark_utils.data_writing import write_df
from boba_python_utils.spark_utils.parallel_compute import parallel_compute


def load_from_feature_files(all_feature_files, use_tqdm=True, description='loading feature data'):
    def _combine_data(result, combined_result):
        if combined_result is None:
            combined_result = result
        else:
            for x, y in zip(result, combined_result):
                y.extend(x)
        return combined_result

    feature_data = None
    for data_file in tqdm_wrap(all_feature_files, use_tqdm=use_tqdm, tqdm_msg=description):
        feature_data = _combine_data(pickle_load(data_file), feature_data)
    return feature_data


def dump_feature_scores(
        paths_feature_data: str,
        spark: SparkSession,
        scorer: Callable,
        model_path: str,
        model_workspace_path: str = None,
        group_id_colname: str = KEY_DATA_ID,
        item_id_colname: str = KEY_INDEX_ITEM_ID,
        label_colname: str = KEY_LABEL,
        output_score_colname: str = KEY_SCORE,
        score_dump_path: Optional[str] = None,
        local_score_dump_path: Optional[str] = None,
        partitions: int = 600,
        is_ranking: bool = False,
        max_group_size: int = None,
        **model_args
):
    # region STEP1: feature data loading
    all_feature_files = sum(
        (
            get_paths_by_pattern(path.join(l2_feature_data, 'feature_data'), pattern='*.bin', recursive=False)
            for l2_feature_data in iter_(paths_feature_data)
        ), []
    )
    if len(all_feature_files) == 0:
        raise ValueError(f"no feature data files found for {paths_feature_data}")

    hprint_message('num_feature_files', len(all_feature_files))

    # endregion

    # region STEP2: compute feature scores
    ensure_dir_existence(local_score_dump_path, clear_dir=True)

    def compute_scores(partition, meta_data_compute=None):
        feature_files = list(map(lambda x: x[0], partition))
        if not feature_files:
            return
        model = scorer(
            model_path=model_path,
            model_workspace_path=(
                path.join(model_workspace_path, str(uuid.uuid4()))
                if model_workspace_path
                else None
            ),
            **model_args
        )

        if hasattr(model, 'predict_proba') and callable(model.predict_proba):
            predict_proba = model.predict_proba
        elif callable(model):
            predict_proba = model
        else:
            raise ValueError(
                "the feature scoring model must either has a method 'predict_proba' "
                f"or be a callable itself; got {model}"
            )

        (
            x_eval, y_eval, group_sizes_eval, data_ids_eval, group_item_ids_eval
        ) = load_from_feature_files(
            feature_files,
            use_tqdm=False
        )

        if group_sizes_eval and max(group_sizes_eval) > max_group_size:
            raise ValueError(
                f'expected max {max_group_size} group size; '
                f'got {max(group_sizes_eval)}'
            )

        scores = predict_proba(
            data=x_eval,
            # NOTE we support group-based ranking;
            # in this case we pass in the `group_sizes_eval`
            group=group_sizes_eval if is_ranking else None
        )

        start = end = 0
        out = []
        for i in range(len(data_ids_eval)):
            data_id, group_size = data_ids_eval[i], group_sizes_eval[i]
            end += group_size
            item = {
                group_id_colname: data_id,
                KEY_RESULT: []
            }
            for item_id, score, label in zip(
                    group_item_ids_eval[start:end],
                    scores[start:end],
                    y_eval[start:end]
            ):
                item[KEY_RESULT].append({
                    item_id_colname: item_id,
                    output_score_colname: float(score),
                    label_colname: label
                })
            out.append(item)
            start = end

        return out

    parallel_compute(
        df=all_feature_files,
        partition_transform_func=compute_scores,
        combine_partition_transform_func=None,
        repartition=partitions,
        file_based_combine=True,
        output_result_to_files=True,
        output_path=local_score_dump_path,
        output_overwrite=True,
        output_write_func=write_json,
        output_file_pattern='{}.json',
        spark=spark
    )
    # endregion

    # region STEP3: copies the computed scores to the final output path
    write_df(
        solve_input(
            local_score_dump_path,
            spark=spark
        ),
        score_dump_path,
        num_files=partitions
    )
    # endregion

    return solve_input(
        score_dump_path,
        spark=spark
    )
