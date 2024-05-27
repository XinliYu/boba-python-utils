import pyspark.sql.functions as F
from Levenshtein import ratio as edit_distance_similarity
from nltk import PorterStemmer
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    BooleanType,
)

from boba_python_utils.production_utils.nlu import (
    get_domain_intent_slots_from_hypothesis,
)

import boba_python_utils.spark_utils as sparku
from boba_python_utils.production_utils.pdfs.constants import (
    KEY_SUM_DEFECT_TERMINATION,
    KEY_SUM_DEFECT_BARGEIN,
    KEY_SUM_DEFECT_REPHRASE,
    KEY_SUM_DEFECT_SANDPAPER,
    KEY_SUM_DEFECT_UNHANDLED,
    KEY_PROD_NLU_MERGER_RESULT,
    KEY_NLU_MERGER_RESULT,
    KEY_SUM_DEFECT,
)

nltk_porter_stemmer = PorterStemmer()


def transform_legacy_daily_aggregation(
        df_daily_aggs: DataFrame, sum_defect_column_rename: str
) -> DataFrame:
    return sparku.rename(
        df_daily_aggs,
        {
            'sum_defective_bargein': KEY_SUM_DEFECT_BARGEIN,
            'sum_defective_termination': KEY_SUM_DEFECT_TERMINATION,
            'sum_rephrase': KEY_SUM_DEFECT_REPHRASE,
            'sum_sandpaper_friction': KEY_SUM_DEFECT_SANDPAPER,
            'sum_unhandled_request': KEY_SUM_DEFECT_UNHANDLED,
            KEY_PROD_NLU_MERGER_RESULT: KEY_NLU_MERGER_RESULT,
            KEY_SUM_DEFECT: sum_defect_column_rename,
        }
    )


def aus_rewrite_filter(request, replaced_request):
    """
    :param request:
    :param replaced_request:
    :return: boolean indicating whether filter should be applied
    """
    if not replaced_request:
        return False

    stop_words = {
        'alexa',
        'play',
        'show',
        'music',
        'movie',
        'video',
        'turn',
        'the',
        'in',
        'on',
        'to',
        'for',
        'by',
        'at',
        'a',
        'yes',
        'no',
    }

    # this is for the case like "play scots" -> "play scotts by travis scott",
    # "play you're replaceable" -> "play irreplaceable by beyonce"
    special_token = ' by '
    replaced_request_token_overlap = len(
        (set(map(nltk_porter_stemmer.stem, request.split())) - stop_words).intersection(
            set(map(nltk_porter_stemmer.stem, replaced_request.split())) - stop_words
        )
    )
    replaced_request_edit_distance_similarity = edit_distance_similarity(request, replaced_request)

    request_has_special_token = special_token in request
    replaced_request_has_special_token = special_token in replaced_request
    if request_has_special_token or replaced_request_has_special_token:
        filtered_request = (
            request[: request.index(special_token)] if request_has_special_token else request
        )
        filtered_replaced_request = (
            replaced_request[: replaced_request.index(special_token)]
            if replaced_request_has_special_token
            else replaced_request
        )

        replaced_request_edit_distance_similarity = max(
            replaced_request_edit_distance_similarity,
            edit_distance_similarity(filtered_request, filtered_replaced_request),
        )

    if replaced_request_token_overlap == 0 and replaced_request_edit_distance_similarity < 0.7:
        return True

    return False


aus_rewrite_filter_udf = F.udf(aus_rewrite_filter, returnType=BooleanType())


def _normalize_entity_value(raw_entity_value) -> str:
    """
    normalize the slot entity value
    Args:
        raw_entity_value: the raw slot entity value
    """
    return raw_entity_value.replace('.', '').replace(' ', '')


def callsign_stationname_slot_change_block(asr_hypothesis, hypothesis):
    """
    Whether to block the given rewrite
        based on asr hypothesis and rewrite hypothesis
        based on the JeffB escalation on 05/26/2021,
        decide to block the rewrite that satisfy the following condition:
            if the asr hypothesis intent is 'PlayStationIntent'
            and for slot entity type in ('CallSign', 'StationName')
            and the rewrite hypothesis didn't contain this entity type
            or the entity value does not match the value in asr hypothesis,
            then this rewrite will be blocked.
    For more details, please refer
    https://wiki.labcollab.net/confluence/display/Doppler/
        Multi-Source+Index+Risk+Rewrites+Mitigations

    Args:
        asr_hypothesis: the asr hypothesis with format
            '[domain]|[intent]|[entity_type1]:[entity_value1]|[entity_type2]:[entity_value2]...'
        hypothesis: the rewrite hypothesis with the same format of asr_hypothesis

    """
    if (not asr_hypothesis) or (not hypothesis):
        return False

    domain1, intent1, slots1 = get_domain_intent_slots_from_hypothesis(asr_hypothesis, slots_as_dict=True)
    domain2, _, slots2 = get_domain_intent_slots_from_hypothesis(hypothesis, slots_as_dict=True)

    if not slots1 or not slots2:
        return False

    if domain1 == 'Music' or intent1 == 'PlayStationIntent':
        for entity_type in ('CallSign', 'StationName'):
            if entity_type in slots1 and (
                    entity_type not in slots2
                    or _normalize_entity_value(slots2[entity_type])
                    != _normalize_entity_value(slots1[entity_type])
            ):
                return True
    return False


callsign_stationname_slot_change_block_udf = F.udf(
    callsign_stationname_slot_change_block, returnType=BooleanType()
)
