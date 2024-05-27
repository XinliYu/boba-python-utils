from functools import partial
from os import path
from typing import Union

from pyspark.sql import SparkSession, Column, DataFrame
from pyspark.sql.types import FloatType, BooleanType, StructType, StructField, ArrayType, StringType

import boba_python_utils.spark_utils.spark_functions as F
import boba_python_utils.spark_utils.data_transform
import boba_python_utils.spark_utils.spark_functions.common
import boba_python_utils.spark_utils as sparku
import boba_python_utils.production_utils.pdfs.constants as c
from slab_aggregation.aggregators.daily_traffic_based_pair_index.filters.turn_pair_index_filter_config import TurnPairIndexFilterConfig
from boba_python_utils.external.rank_bm25 import BM25Okapi
from boba_python_utils.general_utils.modeling_utility.feature_building.constants import KEY_LABEL
from boba_python_utils.general_utils.nlp_utils import (
    char_edit_distance,
)
from boba_python_utils.spark_utils.specialized.indexed_data.constants import KEY_DATA_ID
from boba_python_utils.spark_utils.specialized.indexed_data.data_id import spark_dataframe_add_data_id
from boba_python_utils.spark_utils import (
    INTERNAL_USE_COL_NAME_PREFIX,
)
from boba_python_utils.production_utils._nlu.entity_pairing import extract_slot_pairs_from_hypothesis_udf
from boba_python_utils.production_utils._nlu.target_slots import NON_STOP_SLOT_TYPES
from boba_python_utils.production_utils.greenwich_data.common import (
    get_time_lag_udf,
    CpdrConfig,
)
from boba_python_utils.production_utils.pdfs.testsets.common import (
    save_turn_pairs_with_customer_history,
)


def get_top_index_requests_by_bm25(
        request_first,
        request_second,
        history_index,
        index_request_keys,
        label_key=KEY_LABEL,
        top=5,
        stop_words=None
):
    if stop_words is None:
        stop_words = {}

    corpus, labels, tokenized_corpus = [], [], []
    for index_item in history_index:
        doc = None
        for index_request_key in index_request_keys:
            doc = index_item[index_request_key]
            if doc:
                break
        if doc:
            label = index_item[label_key]
            if doc != request_first:
                tokens = [token for token in doc.split() if token not in stop_words]
                if tokens:
                    corpus.append(doc)
                    labels.append(label)
                    tokenized_corpus.append(tokens)

    query_tokens = [token for token in request_second.split() if token not in stop_words]

    if query_tokens and tokenized_corpus:
        bm25 = BM25Okapi(tokenized_corpus)
        top_n_doc_idxes = bm25.get_top_n_doc_indexes(query_tokens, corpus, n=top)

        out_pos_docs, out_neg_docs = [], []
        for i in top_n_doc_idxes:
            doc = corpus[i]
            label = labels[i]
            if label:
                out_pos_docs.append(doc)
            else:
                out_neg_docs.append(doc)
        return out_pos_docs, out_neg_docs
    else:
        return [], []


def get_top_index_requests_by_bm25_udf(
        index_request_keys,
        positive_example_colname='positive',
        negative_example_colname='negative',
        label_key=KEY_LABEL,
        top=5,
        stop_words=None
):
    return F.udf(
        partial(
            get_top_index_requests_by_bm25,
            index_request_keys=index_request_keys,
            label_key=label_key,
            top=top,
            stop_words=stop_words
        ),
        returnType=StructType(fields=[
            StructField(name=positive_example_colname, dataType=ArrayType(StringType())),
            StructField(name=negative_example_colname, dataType=ArrayType(StringType()))
        ])
    )


def _get_turn_pair_match_hist_label(
        df_hist_exp: DataFrame,
        df_turn_pairs: DataFrame,
        request_key_hist: str,
        hypothesis_key_hist: str,
        request_key_turn_pairs: str,
        hypothesis_key_turn_pairs: str,
        customer_id_key_hist: str = None,
        customer_id_key_turn_pairs: str = None,
) -> Column:
    hypothesis_match = (F.size(F.split(df_turn_pairs[hypothesis_key_turn_pairs], r'\|')) > 2) & (
            df_turn_pairs[hypothesis_key_turn_pairs]
            == df_hist_exp[hypothesis_key_hist]  # noqa: E501,E126
    )
    if c.KEY_REPLACED_REQUEST in df_hist_exp.columns:
        _rephrase_in_hist_filter = (
                (
                        df_turn_pairs[request_key_turn_pairs] == df_hist_exp[request_key_hist]
                )  # noqa: E501,E126
                | (
                        df_hist_exp[c.KEY_REPLACED_REQUEST].isNotNull()  # noqa: E126
                        & (df_turn_pairs[request_key_turn_pairs] == df_hist_exp[c.KEY_REPLACED_REQUEST])
                )
                | hypothesis_match
        )
    else:
        _rephrase_in_hist_filter = (
                                           df_turn_pairs[request_key_turn_pairs] == df_hist_exp[request_key_hist]
                                   ) | hypothesis_match  # noqa: E501,E126

    if (
            customer_id_key_hist in df_hist_exp.columns
            and customer_id_key_turn_pairs in df_turn_pairs.columns
    ):
        return _rephrase_in_hist_filter & (
                df_turn_pairs[customer_id_key_turn_pairs]  # noqa: E126
                == df_hist_exp[customer_id_key_hist]
        )
    else:
        return _rephrase_in_hist_filter


def add_label_to_customer_history_index(
        df_turn_pairs_with_customer_history_index: DataFrame,
        label_col_name: str = c.KEY_LABEL,
        truth_request_key: str = c.KEY_REQUEST_SECOND,
        truth_hypothesis_key: str = c.KEY_HYPOTHESIS_SECOND,
        data_id_key: str = None,
        history_key: str = c.KEY_HISTORY_LIST,
):
    """
    Adds a label to each customer history index entry.
    Args:
        df_turn_pairs_with_customer_history_index:
        label_col_name:
        truth_request_key:
        truth_hypothesis_key:
        data_id_key:
        history_key:

    Returns:

    """
    if history_key in df_turn_pairs_with_customer_history_index.columns:
        data_id_key = data_id_key or sparku.get_internal_colname(KEY_DATA_ID)
        data_id_key_exists = data_id_key in df_turn_pairs_with_customer_history_index.columns
        if not data_id_key_exists:
            df_turn_pairs_with_customer_history_index = spark_dataframe_add_data_id(
                df_data=df_turn_pairs_with_customer_history_index,
                data_id_colname=data_id_key,
                data_id_func=F.monotonically_increasing_id
            )

        _request_key = sparku.get_internal_colname(truth_request_key)
        _hypothesis_key = sparku.get_internal_colname(truth_hypothesis_key)
        df_turn_pairs_with_customer_history_index_exp = sparku.explode_as_flat_columns(
            df_turn_pairs_with_customer_history_index.select(
                data_id_key,
                F.col(truth_request_key).alias(_request_key),
                F.col(truth_hypothesis_key).alias(_hypothesis_key),
                history_key,
            ),
            col_to_explode=history_key
        )

        df_hist_exp_with_labels = df_turn_pairs_with_customer_history_index_exp.withColumn(
            label_col_name,
            _get_turn_pair_match_hist_label(
                df_turn_pairs_with_customer_history_index_exp,
                df_turn_pairs_with_customer_history_index_exp,
                request_key_hist=c.KEY_REQUEST,
                hypothesis_key_hist=c.KEY_HYPOTHESIS,
                request_key_turn_pairs=_request_key,
                hypothesis_key_turn_pairs=_hypothesis_key,
            ),
        ).drop(_request_key, _hypothesis_key)

        df_hist_exp_with_labels_folded = boba_python_utils.spark_utils.data_transform.fold(
            df_hist_exp_with_labels, group_cols=[data_id_key], fold_colname=history_key
        )

        df_turn_pairs_with_customer_history_index_labeled = (
            df_turn_pairs_with_customer_history_index.drop(history_key).join(
                df_hist_exp_with_labels_folded, [data_id_key]
            )
        )
        if not data_id_key_exists:
            df_turn_pairs_with_customer_history_index_labeled = (
                df_turn_pairs_with_customer_history_index_labeled.drop(data_id_key)
            )
        return df_turn_pairs_with_customer_history_index_labeled
    else:
        return df_turn_pairs_with_customer_history_index.withColumn(
            label_col_name,
            _get_turn_pair_match_hist_label(
                df_turn_pairs_with_customer_history_index,
                df_turn_pairs_with_customer_history_index,
                request_key_hist=c.KEY_REQUEST,
                hypothesis_key_hist=c.KEY_HYPOTHESIS,
                request_key_turn_pairs=truth_request_key,
                hypothesis_key_turn_pairs=truth_hypothesis_key,
            ),
        )


def get_customer_history_index_overlap_with_turn_pairs(
        df_customer_history_index_exploded,
        df_turn_pairs,
        turn_pairs_request_key=c.KEY_REQUEST_SECOND,
        turn_pairs_hypothesis_key=c.KEY_HYPOTHESIS_SECOND,
):
    _customer_id = f'{INTERNAL_USE_COL_NAME_PREFIX}{c.KEY_CUSTOMER_ID}'
    _request_key = f'{INTERNAL_USE_COL_NAME_PREFIX}{turn_pairs_request_key}'
    _hypothesis_key = f'{INTERNAL_USE_COL_NAME_PREFIX}{turn_pairs_hypothesis_key}'
    df_turn_pairs = df_turn_pairs.select(
        F.col(c.KEY_CUSTOMER_ID).alias(_customer_id),
        F.col(turn_pairs_request_key).alias(_request_key),
        F.col(turn_pairs_hypothesis_key).alias(_hypothesis_key),
    ).distinct()

    _rephrase_in_hist_filter = _get_turn_pair_match_hist_label(
        df_hist_exp=df_customer_history_index_exploded,
        df_turn_pairs=df_turn_pairs,
        request_key_hist=c.KEY_REQUEST,
        hypothesis_key_hist=c.KEY_HYPOTHESIS,
        customer_id_key_hist=c.KEY_CUSTOMER_ID,
        request_key_turn_pairs=_request_key,
        hypothesis_key_turn_pairs=_hypothesis_key,
        customer_id_key_turn_pairs=_customer_id,
    )

    # ! we cannot use the following code
    # ! it will cause duplicates
    # return df_hist_overlap_with_rephrase_keys.join(
    #     df_turn_pairs, _rephrase_in_hist_filter  # ! this is not guaranteed one-one mapping
    # ).drop(
    #     *df_turn_pairs.columns
    # )

    _join_keys = [c.KEY_CUSTOMER_ID, c.KEY_REQUEST, c.KEY_HYPOTHESIS]
    df_hist_overlap_with_rephrase_keys = (
        df_customer_history_index_exploded.select(
            *_join_keys, c.KEY_REPLACED_REQUEST  # "replaced_request" is required for labeling
        ).join(
            df_turn_pairs, _rephrase_in_hist_filter
        ).drop(
            *df_turn_pairs.columns, c.KEY_REPLACED_REQUEST
        ).distinct()
    )

    return df_customer_history_index_exploded.join(
        df_hist_overlap_with_rephrase_keys,
        _join_keys
    )


def turn_pair_in_history_filter(
        df_turn_pairs,
        df_customer_history_index_exploded,
        request_key=c.KEY_REQUEST_SECOND,
        hypothesis_key=c.KEY_HYPOTHESIS_SECOND,
):
    _customer_id = f'{INTERNAL_USE_COL_NAME_PREFIX}{c.KEY_CUSTOMER_ID}'
    _request_key = f'{INTERNAL_USE_COL_NAME_PREFIX}{c.KEY_REQUEST}'
    _hypothesis_key = f'{INTERNAL_USE_COL_NAME_PREFIX}{c.KEY_HYPOTHESIS}'
    df_hist_exp = df_customer_history_index_exploded.select(
        F.col(c.KEY_CUSTOMER_ID).alias(_customer_id),
        F.col(c.KEY_REQUEST).alias(_request_key),
        F.col(c.KEY_HYPOTHESIS).alias(_hypothesis_key),
        F.col(c.KEY_REPLACED_REQUEST),  # "replaced_request" is required for labeling
    ).distinct()

    _rephrase_in_hist_filter = _get_turn_pair_match_hist_label(
        df_hist_exp=df_hist_exp,
        df_turn_pairs=df_turn_pairs,
        request_key_hist=_request_key,
        hypothesis_key_hist=_hypothesis_key,
        customer_id_key_hist=_customer_id,
        request_key_turn_pairs=request_key,
        hypothesis_key_turn_pairs=hypothesis_key,
        customer_id_key_turn_pairs=c.KEY_CUSTOMER_ID,
    )

    # ! we cannot use the following code
    # ! it will cause duplicates
    # return df_turn_pairs.join(
    #     df_hist_exp, _rephrase_in_hist_filter  # ! this is not guaranteed one-one mapping
    # ).drop(
    #     *df_hist_exp.columns
    # )

    _join_keys = [c.KEY_CUSTOMER_ID, request_key, hypothesis_key]
    df_turn_pairs_overlap_with_hist_keys = (
        df_turn_pairs.select(*_join_keys).join(
            df_hist_exp, _rephrase_in_hist_filter
        ).drop(
            _customer_id, _request_key, _hypothesis_key, c.KEY_REPLACED_REQUEST
        ).distinct()
    )

    return sparku.join_on_columns(
        df_turn_pairs,
        df_turn_pairs_overlap_with_hist_keys,
        _join_keys,
        prevent_resolved_attribute_missing_error=True  # ! must set this to True to avoid crash
    )


def _flatten_customer_history_index(
        df_customer_history_index,
        history_index_col_name=c.KEY_HISTORY_LIST_PROD
):
    if history_index_col_name in df_customer_history_index.columns:
        df_customer_history_index_exploded = sparku.explode_as_flat_columns(df=df_customer_history_index, col_to_explode=c.KEY_HISTORY_LIST_PROD)
    else:
        df_customer_history_index_exploded = df_customer_history_index
    return df_customer_history_index_exploded


def get_customer_history_index_overlap_by_turn_pairs(
        df_turn_pairs,
        df_customer_history_index,
        history_index_col_name=c.KEY_HISTORY_LIST_PROD
):
    df_customer_history_index_exploded = _flatten_customer_history_index(
        df_customer_history_index,
        history_index_col_name
    )

    df_customer_history_index_overlap_with_turn_pairs = get_customer_history_index_overlap_with_turn_pairs(
        df_customer_history_index_exploded=df_customer_history_index_exploded,
        df_turn_pairs=df_turn_pairs,
        turn_pairs_request_key=c.KEY_REQUEST_SECOND,
        turn_pairs_hypothesis_key=c.KEY_NLU_HYPOTHESIS_SECOND,
    )
    df_customer_history_index_overlap_with_turn_pairs_folded = boba_python_utils.spark_utils.data_transform.fold(
        df_customer_history_index_overlap_with_turn_pairs.withColumn(
            c.KEY_NLU_HYPOTHESIS_SECOND, F.col(c.KEY_HYPOTHESIS)
        ),
        group_cols=[c.KEY_CUSTOMER_ID, c.KEY_NLU_HYPOTHESIS_SECOND],
        fold_colname=c.KEY_HISTORY_LIST,
    )

    return df_customer_history_index_overlap_with_turn_pairs_folded


def get_turn_pairs_overlap_with_customer_history_index(
        df_turn_pairs,
        df_customer_history_index,
        turn1_in_hist_filter=False,
        turn2_in_hist_filter=False,
        history_index_col_name=c.KEY_HISTORY_LIST_PROD
):
    df_customer_history_index_exploded = _flatten_customer_history_index(
        df_customer_history_index,
        history_index_col_name
    )

    if turn2_in_hist_filter:
        df_turn_pairs = turn_pair_in_history_filter(
            df_turn_pairs,
            df_customer_history_index_exploded,
            request_key=c.KEY_REQUEST_SECOND,
            hypothesis_key=c.KEY_NLU_HYPOTHESIS_SECOND,
        )
    if turn1_in_hist_filter:
        df_turn_pairs = turn_pair_in_history_filter(
            df_turn_pairs,
            df_customer_history_index_exploded,
            request_key=c.KEY_REQUEST_FIRST,
            hypothesis_key=c.KEY_NLU_HYPOTHESIS_FIRST,
        )
    return df_turn_pairs


def _edit_distance_ratio(text1, text2, bow=False):
    if len(text1) <= 2 and len(text2) <= 2:
        return 1.0
    else:
        return 1 - char_edit_distance(text1, text2, bow=bow, normalized=True)


def _entity_pair_similarity_score(entity1, entity2):
    if entity1 in entity2 or entity2 in entity1 or len(entity1) == 1 or len(entity2) == 1:
        return 1.0
    return max(
        _edit_distance_ratio(entity1, entity2), _edit_distance_ratio(entity1, entity2, bow=True)
    )


@F.udf(returnType=FloatType())
def _get_entity_pairs_similarity_score_udf(entity_pairs):
    if entity_pairs:
        return min(row[0] for row in entity_pairs)


def build_entity_swap_guardrail_data(
        output_path_for_data_with_selected_customer_index: str,
        output_path_for_data_with_full_customer_index: str,
        input_turn_pair_index: Union[str, DataFrame],
        input_customer_index: Union[str, DataFrame],
        max_entity_similarity: float = 0.3,
        min_utterance_similarity: float = 0.7,
        timelag_threshold: int = 4,
        enable_customer_index_filter: bool = True,
        cpdr_config: CpdrConfig = None,
        save_requests: bool = True,
        save_history_index: bool = True,
        save_combined: bool = True,
        spark: SparkSession = None,
        input_format_turn_pair_index='json',
        input_format_customer_index='json'
):
    # region STEP0: load turn-pair aggregation and customer history index
    df_turn_pairs = sparku.cache__(
        input_turn_pair_index,
        spark=spark,
        name='df_turn_paris',
        input_format=input_format_turn_pair_index
    )

    # endregion

    # region STEP1: basic filtering
    if cpdr_config is None:
        cpdr_config = CpdrConfig()

    df_turn_paris_with_non_defect_turn1 = sparku.cache__(
        df_turn_pairs.where(
            # turn1 should be non-defect;
            # but we do not filter turn2 on defect score,
            # we will later filter it by customer history
            (F.col(cpdr_config.utterance_defect_first_colname) == 0) &
            # turn1/turn2 should have different utterance or hypothesis
            (F.col(c.KEY_REQUEST_FIRST) != F.col(c.KEY_REQUEST_SECOND)) &
            (F.col(c.KEY_HYPOTHESIS_FIRST) != F.col(c.KEY_HYPOTHESIS_SECOND)) &
            # turn1/turn2 should not contain each other
            (
                ~(  # noqa: E126
                        F.col(c.KEY_REQUEST_FIRST).contains(F.col(c.KEY_REQUEST_SECOND)) |
                        F.col(c.KEY_REQUEST_SECOND).contains(F.col(c.KEY_REQUEST_FIRST))
                )
            )
        ),
        name='df_turn_paris_with_non_defect_turn1',
        unpersist=df_turn_pairs
    )
    # endregion

    # region STEP2: filter by entity pair similarities
    KEY_MIN_ENTITY_SIMILAIRTY = 'min_entity_similarity'
    KEY_UTTERANCE_SIMILARITY = 'utterance_similarity'
    df_turn_paris_with_non_defect_turn1_and_entity_pairs = sparku.cache__(
        sparku.with_columns(
            df_turn_paris_with_non_defect_turn1.withColumn(
                c.KEY_ENTITY_PAIRS,
                extract_slot_pairs_from_hypothesis_udf(
                    source_hypothesis=c.KEY_HYPOTHESIS_FIRST,
                    target_hypothesis=c.KEY_HYPOTHESIS_SECOND,
                    source_utterance=c.KEY_REQUEST_FIRST,
                    target_utterance=c.KEY_REQUEST_SECOND,
                    slot_type_filter=NON_STOP_SLOT_TYPES,
                    source_slot_type_colname=c.KEY_ENTITY_TYPE_FIRST,
                    source_slot_value_colname=c.KEY_ENTITY_FIRST,
                    target_slot_type_colname=c.KEY_ENTITY_TYPE_SECOND,
                    target_slot_value_colname=c.KEY_ENTITY_SECOND,
                    slot_align_score_colname=c.KEY_ENTITY_ALIGNMENT_SCORE
                ),
            ),
            {
                KEY_MIN_ENTITY_SIMILAIRTY: _get_entity_pairs_similarity_score_udf(c.KEY_ENTITY_PAIRS),
                KEY_UTTERANCE_SIMILARITY: (
                        1 - (F.col(c.KEY_TURN_PAIR_TOKEN_SORTED_EDIT_DISTANCE)
                             / F.greatest(F.length(c.KEY_REQUEST_FIRST), F.length(c.KEY_REQUEST_SECOND)))
                )
            }
        ),
        name='df_turn_paris_with_non_defect_turn1_and_entity_pairs',
        unpersist=df_turn_paris_with_non_defect_turn1,
    )

    # endregion

    # region STEP3: filter by utterance level distance and time lag
    df_turn_paris_entity_risk = sparku.cache__(
        sparku.rename(
            df_turn_paris_with_non_defect_turn1_and_entity_pairs.where(
                (F.col(c.KEY_TURN_PAIR_TOKEN_SORTED_EDIT_DISTANCE) > 0) &
                (F.col(c.KEY_TIME_LAG) >= timelag_threshold) &
                (F.col(KEY_MIN_ENTITY_SIMILAIRTY) < max_entity_similarity) &
                (F.col(KEY_UTTERANCE_SIMILARITY) > min_utterance_similarity)
            ),
            {
                # TODO: online/offline NLU hypothesis column names are different
                c.KEY_HYPOTHESIS_FIRST: c.KEY_NLU_HYPOTHESIS_FIRST,
                c.KEY_HYPOTHESIS_SECOND: c.KEY_NLU_HYPOTHESIS_SECOND
            }
        ),
        name='df_turn_paris_entity_risk',
        unpersist=df_turn_paris_with_non_defect_turn1_and_entity_pairs,
    )
    # endregion

    # region STEP4: extracts history that overlap with the turn pairs

    df_customer_index = sparku.cache__(
        input_customer_index,
        spark=spark,
        name='df_customer_index',
        input_format=input_format_customer_index
    )

    df_customer_index_flat = sparku.cache__(
        sparku.explode_as_flat_columns(
            df=df_customer_index,
            col_to_explode=c.KEY_HISTORY_LIST_PROD
        ),
        name='df_customer_index_flat'
    )

    if enable_customer_index_filter:
        df_turn_paris_entity_risk_filtered_by_customer_index = sparku.cache__(
            get_turn_pairs_overlap_with_customer_history_index(
                df_turn_pairs=df_turn_paris_entity_risk,
                df_customer_history_index=df_customer_index_flat,
                turn1_in_hist_filter=True,
                turn2_in_hist_filter=True,
                history_index_col_name=c.KEY_HISTORY_LIST_PROD
            ),
            name='df_turn_paris_entity_risk_filtered_by_customer_index'
        )
    else:
        df_turn_paris_entity_risk_filtered_by_customer_index = df_turn_paris_entity_risk

    df_customer_index_overlap_with_entity_risk = sparku.cache__(
        get_customer_history_index_overlap_by_turn_pairs(
            df_turn_pairs=df_turn_paris_entity_risk,
            df_customer_history_index=df_customer_index_flat,
            history_index_col_name=c.KEY_HISTORY_LIST_PROD
        ),
        name='df_customer_history_index_overlap_with_turn_pairs_folded',
        unpersist=df_customer_index_flat
    )

    # endregion

    # region STEP5: save filtered turn pair data joint with customer history
    save_turn_pairs_with_customer_history(
        spark=spark,
        df_rephrase=df_turn_paris_entity_risk_filtered_by_customer_index,
        customer_history_index_dataframes_or_paths=[
            df_customer_index_overlap_with_entity_risk,
            df_customer_index
        ],
        output_paths=[
            output_path_for_data_with_selected_customer_index,
            output_path_for_data_with_full_customer_index
        ],
        save_requests=save_requests,
        save_history_index=save_history_index,
        save_combined=save_combined,
        extra_join_keys=[
            [c.KEY_NLU_HYPOTHESIS_SECOND],
            []
        ],
    )

    df_customer_index_flat.unpersist()
    df_turn_paris_entity_risk_filtered_by_customer_index.unpersist()
    df_customer_index_overlap_with_entity_risk.unpersist()
    # endregion


@F.udf(returnType=BooleanType())
def _asr_nbest_based_rephrase(occurrences, request_first, request_second):
    for occurrence in occurrences:
        nbest_first = occurrence[c.KEY_ASR_NBEST_FIRST]
        if nbest_first is not None and request_first != request_second:
            for asr_nbest in nbest_first:
                if asr_nbest[c.KEY_UTTERANCE] == request_second:
                    return True
    return False


def build_opportunity_data(
        output_path,
        input_turn_pair_index,
        input_customer_index,
        input_format_turn_pair_index_format=None,
        input_format_customer_index=None,
        cpdr_config: CpdrConfig = None,
        turn_pair_edit_dist_ranges=None,
        turn_pair_index_filter_config: TurnPairIndexFilterConfig = None,
        aggregate_occurrences=True,
        group_cols=c.GROUP_KEYS__CUSTOMER_REQ_HYP_DOMAIN_INTENT,
        non_occurrence_cols=c.KEYS_REPHRASE_TRAFFIC_NON_OCCURRENCE_COLS,
        occurrences_col=c.KEY_OCCURRENCES,
        save_requests=False,
        save_history_index=False,
        save_combined=True,
        turn2_in_hist_filter=True,
        edit_distance_filter_waiver=None,
        spark=None,
):
    if cpdr_config is None:
        cpdr_config = CpdrConfig()
    if non_occurrence_cols is None:
        non_occurrence_cols = ()
    if cpdr_config.session_defect_colname not in non_occurrence_cols:
        non_occurrence_cols = (cpdr_config.session_defect_colname, *non_occurrence_cols)

    # region STEP1: turn pair index filtering

    df_turn_pair_index = sparku.cache__(
        sparku.solve_input(input_turn_pair_index, spark, input_format_turn_pair_index_format),
        'df_turn_pair_index',
    )

    # region TODO: compatibility for legacy turn pair index
    if c.KEY_TIME_LAG not in df_turn_pair_index.columns:
        df_turn_pair_index = df_turn_pair_index.withColumn(
            c.KEY_TIME_LAG,
            get_time_lag_udf(c.KEY_TIMESTAMP_FIRST, c.KEY_TIMESTAMP_SECOND),
        )
    session_defect_colname = f'{cpdr_config.session_defect_colname}{c.NAME_SUFFIX_FIRST}'
    if f'{cpdr_config.session_defect_colname}{c.NAME_SUFFIX_FIRST}' in df_turn_pair_index.columns:
        df_turn_pair_index = df_turn_pair_index.withColumnRenamed(
            session_defect_colname, cpdr_config.session_defect_colname
        ).drop(f'{cpdr_config.session_defect_colname}{c.NAME_SUFFIX_SECOND}')
    # endregion

    if turn_pair_index_filter_config is not None:
        # For any change on the existing filter (e.g. thresholds),
        # update the configuration in `turn_pair_index_filter_config`;
        # To add additional filter logic,
        # we need to extend the `get_turn_pair_index_aggregation_filter` function.

        # region ! DEBUG USE !

        # When debugging, uncomment the following variables, copy them to ipython or notebook,
        # and then go into the `get_turn_pair_index_aggregation_filter` to execute line by line;
        # Make adjustment to the variable values if necessary.

        # filter_config = turn_pair_index_filter_config
        # cpdr_config = cpdr_config
        # turn1_utterance_col = c.KEY_REQUEST_FIRST
        # turn2_utterance_col = c.KEY_REQUEST_SECOND
        # turn1_dialog_id_col = c.KEY_DIALOG_ID_FIRST
        # turn2_dialog_id_col = c.KEY_DIALOG_ID_SECOND
        # turn_pair_edit_dist_col = c.KEY_TURN_PAIR_EDIT_DISTANCE
        # turn_pair_bow_edit_dist_col = c.KEY_TURN_PAIR_BOW_EDIT_DISTANCE
        # turn_pair_timelag_col = c.KEY_TIME_LAG

        # endregion
        # turn_pair_index_aggregation_filter = get_turn_pair_index_aggregation_filter(
        #     filter_config=turn_pair_index_filter_config,
        #     cpdr_config=cpdr_config,
        #     turn1_utterance_col=c.KEY_REQUEST_FIRST,
        #     turn2_utterance_col=c.KEY_REQUEST_SECOND,
        #     turn1_dialog_id_col=c.KEY_DIALOG_ID_FIRST,
        #     turn2_dialog_id_col=c.KEY_DIALOG_ID_SECOND,
        #     turn_pair_edit_dist_col=c.KEY_TURN_PAIR_EDIT_DISTANCE,
        #     turn_pair_token_sorted_edit_dist_col=c.KEY_TURN_PAIR_TOKEN_SORTED_EDIT_DISTANCE,
        #     turn_pair_timelag_col=c.KEY_TIME_LAG,
        # )
        # hprint_message(
        #     title='filter df_turn_pair_index', content=str(turn_pair_index_aggregation_filter)
        # )
        df_turn_pair_index = sparku.cache__(
            turn_pair_index_filter_config.filter(df_turn_pair_index, cpdr_config),
            name='df_turn_pair_index (filtered)',
            unpersist=df_turn_pair_index,
        )

    if aggregate_occurrences:
        _group_cols = [
            col_name
            for col_name in (c.KEY_TURN_PAIR_EDIT_DISTANCE, c.KEY_TURN_PAIR_TOKEN_SORTED_EDIT_DISTANCE)
            if col_name in df_turn_pair_index.columns
        ]
        for col_name in group_cols:
            if col_name in _group_cols:
                continue
            if (
                    col_name in non_occurrence_cols
                    or col_name.endswith(c.NAME_SUFFIX_FIRST)
                    or col_name.endswith(c.NAME_SUFFIX_SECOND)
            ):
                _group_cols.append(col_name)
            else:
                _group_cols.append(col_name + c.NAME_SUFFIX_FIRST)
                _group_cols.append(col_name + c.NAME_SUFFIX_SECOND)
        _collect_list_cols = [
            col_name
            for col_name in df_turn_pair_index.columns
            if (col_name not in _group_cols) and (col_name not in non_occurrence_cols)
        ]
        df_turn_pair_index = df_turn_pair_index.groupBy(*_group_cols).agg(
            F.collect_list(F.struct(*_collect_list_cols)).alias(occurrences_col)
        )

    df_turn_pair_index_filtered = sparku.cache__(
        df_or_path=sparku.rename(
            df_turn_pair_index,
            {
                # TODO: online/offline NLU hypothesis column names are different
                c.KEY_HYPOTHESIS_FIRST: c.KEY_NLU_HYPOTHESIS_FIRST,
                c.KEY_HYPOTHESIS_SECOND: c.KEY_NLU_HYPOTHESIS_SECOND,
            },
        ),
        name='df_turn_pair_index_filtered',
        unpersist=df_turn_pair_index,
    )  # 12465218

    # endregion

    # region STEP2: joins with history

    df_customer_history_index = sparku.cache__(
        input_customer_index,
        spark=spark,
        input_format=input_format_customer_index,
        name='df_customer_history_index'
    )

    def _save(_df_turn_pairs, _output_path):
        if turn2_in_hist_filter:
            _df_turn_pairs = get_turn_pairs_overlap_with_customer_history_index(
                df_turn_pairs=_df_turn_pairs,
                df_customer_history_index=df_customer_history_index,
                turn1_in_hist_filter=False,
                turn2_in_hist_filter=True
            )

        return save_turn_pairs_with_customer_history(
            spark=spark,
            df_rephrase=_df_turn_pairs,
            customer_history_index_dataframes_or_paths=df_customer_history_index,  # noqa: E501
            output_paths=output_path,
            save_requests=save_requests,
            save_history_index=save_history_index,
            save_combined=save_combined,
        )

    if turn_pair_edit_dist_ranges:
        if turn_pair_index_filter_config.includes_asrnbest_based_rephrase:
            edit_distance_filter_waiver = boba_python_utils.spark_utils.spark_functions.common_functions.or_(
                edit_distance_filter_waiver,
                _asr_nbest_based_rephrase(
                    c.KEY_OCCURRENCES, c.KEY_REQUEST_FIRST, c.KEY_REQUEST_SECOND
                ),
            )
        outs = []
        for i, (min_ed_dist, max_ed_dist) in enumerate(turn_pair_edit_dist_ranges):
            edit_dist_filter = (F.col(c.KEY_TURN_PAIR_EDIT_DISTANCE) <= max_ed_dist) & (
                    F.col(c.KEY_TURN_PAIR_EDIT_DISTANCE) >= min_ed_dist  # noqa: E126
            )
            edit_dist_filter_desc = (
                f'df_rephrases_with_stats (edit dist from {min_ed_dist} to {max_ed_dist})'
            )
            if i == 0 and edit_distance_filter_waiver is not None:
                edit_dist_filter = edit_distance_filter_waiver | edit_dist_filter
                edit_dist_filter_desc += '; with customized additional rephrases'
            _df_rephrases_with_stats = sparku.cache__(
                df_turn_pair_index_filtered.where(edit_dist_filter), edit_dist_filter_desc
            )
            _output_path = path.join(output_path, f'ed_{min_ed_dist}_{max_ed_dist}')
            outs.append(_save(_df_rephrases_with_stats, _output_path))
            _df_rephrases_with_stats.unpersist()
        df_turn_pair_index_filtered.unpersist()
        df_customer_history_index.unpersist()
        return outs
    else:
        return _save(df_turn_pair_index_filtered, output_path)

    # endregion
