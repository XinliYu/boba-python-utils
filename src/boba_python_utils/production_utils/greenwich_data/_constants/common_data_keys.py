from boba_python_utils.general_utils.modeling_utility.feature_building.constants import KEY_LABEL
from boba_python_utils.general_utils.common_constants import *

NAME_PREFIX_BLOCKED = 'blocked_'
NAME_PREFIX_QUERY = 'query_'
NAME_PREFIX_REWRITE = 'rewrite_'
NAME_PREFIX_GLOBAL = 'global_'
NAME_PREFIX_GLOBAL_AVG = NAME_PREFIX_GLOBAL + NAME_PREFIX_AVG  # global_avg_
NAME_PREFIX_GLOBAL_SUM = NAME_PREFIX_GLOBAL + NAME_PREFIX_SUM  # global_sum_
NAME_PREFIX_CUSTOMER = 'customer_'
NAME_PREFIX_CUSTOMER_AVG = NAME_PREFIX_CUSTOMER + NAME_PREFIX_AVG  # customer_avg_
NAME_PREFIX_CUSTOMER_SUM = NAME_PREFIX_CUSTOMER + NAME_PREFIX_SUM  # customer_sum_

# region aggregation keys
KEY_LOCALE = 'locale'
KEY_SESSION_ID = 'session_id'
KEY_SESSION_TURNS = 'session_turns'
KEY_UTTERANCE_ID = 'utterance_id'
KEY_UTTERANCE_ID_GW_STYLE = 'utteranceId'
KEY_DIALOG_ID = 'dialog_id'
KEY_DEVICE_ID = 'device_id'
KEY_DEVICE_TYPE = 'device_type'
KEY_CLIENT_PROFILE = 'client_profile'
KEY_DEVICE_NO_SCREEN = 'device_no_screen'
KEY_DEVICE_IS_ECHO_SHOW = 'device_is_echo_show'
KEY_DEVICE_COMMON_NAME = 'device_common_name'
KEY_INDEX = 'index'
KEY_REQUEST = 'request'
KEY_UTTERANCE = 'utterance'
KEY_EMBEDDING = 'embed'
KEY_REPLACED_REQUEST = 'replaced_request'
KEY_AUS_REQUEST = 'aus_request'
KEY_HYPOTHESIS = 'hypothesis'
KEY_NLU_HYPOTHESIS = 'nlu_hypothesis'
KEY_ASR_HYPOTHESIS = 'asr_hypothesis'
KEY_AUS_HYPOTHESIS = 'aus_hypothesis'
KEY_DOMAIN = 'domain'
KEY_ASR_DOMAIN = 'asr_domain'
KEY_AUS_DOMAIN = 'aus_domain'
KEY_INTENT = 'intent'
KEY_ASR_INTENT = 'asr_intent'
KEY_AUS_INTENT = 'aus_intent'
KEY_RESOLVED_SLOTS = 'slots'
KEY_RESPONSE = 'response'
KEY_DFS_SHOULD_TRIGGER = 'shouldTrigger'
KEY_IS_HOLDOUT = 'is_holdout'
KEY_DFS_SOURCE = 'DFS_SOURCE'
KEY_DFS_SCORE = 'DFS_score'
KEY_DFS_SCORE_BIN = 'DFS_score_bin'
KEY_DFS_LATENCY = 'DFS_latency'
KEY_NLU_MERGER_RESULT = 'nluAusMergerResult'
KEY_PROD_NLU_MERGER_RESULT = 'nlu_aus_merger_result'
KEY_PROVIDER_NAME = 'providerName'
KEY_PROVIDER_NAME_PROD = 'provider_name'
KEY_ASR_INTERPRETATION = 'asrInterpretation'
KEY_AUS_INTERPRETATION = 'ausInterpretation'
KEY_AUS_CRITICAL = 'ausCritical'
KEY_NLU_MERGER_RULE = 'nluAusMergerRule'
KEY_NLU_MERGER_RULE_VERSION = 'nluAusMergerRuleVersion'
KEY_NLU_MERGER_RULE_C = 'nluAusMergerRule_C'
KEY_NLU_MERGER_RULE_T1 = 'nluAusMergerRule_T1'
KEY_NLU_MERGER_RULE_T2 = 'nluAusMergerRule_T2'
KEY_NLU_MERGER_RULE_DETAILS = 'nluAusMergerRuleDetails'
KEY_NLU_MERGER_RULE_NO_DOMAIN_BLOCK = 'nluAusMergerRule_no_domain_block'
KEY_MAX_TIMESTAMP = NAME_PREFIX_MAX + KEY_TIMESTAMP
KEY_TURNS = 'turns'
KEY_NUM_TURNS = 'num_turns'
KEY_NUM_SLOTS = 'num_slots'
KEY_NUM_ENTITIES = 'num_entities'
KEY_WEBLAB = 'weblab'
KEY_NLU_MERGER_DETAILED_RESULT = 'nluAusMergerDetailedResult'
KEY_MACAW_CUSTOMER_RESPONSE_TYPE = 'macaw_customer_response_type'
KEY_IS_LLM_TRAFFIC = 'is_llm_traffic'
KEY_LLM_TOKEN = 'llm_token'
KEY_LLM_SESSION_TOKEN = 'llm_session_token'
KEY_LLM_PROPERTIES = 'llm_properties'
KEY_ALEXA_STACK_TYPE = 'alexa_stack_type'
KEY_ALEXA_STACK_CONFIG = 'alexa_stack_config'
KEY_INFO_CATEGORY = 'info_category'
KEY_CHILD_DIRECT_REQUEST = 'child_directed_request'
KEY_MACAW_CAMPAIGN_ID = 'macaw_campaign_id'
KEY_DEVICE_VIDEO_STATE = 'device_video_state'
KEY_VIDEO_DEVICE_ACTIVE_SESSION = 'video_device_active_session'
KEY_VIDEO_DEVICE_ENABLED = 'video_device_enabled'

# region asr nbest keys
KEY_ASR_NBEST = 'n-best'
KEY_ASR_NBEST_FIRST = 'n-best' + NAME_SUFFIX_FIRST
KEY_ASR_NBEST_SECOND = 'n-best' + NAME_SUFFIX_SECOND
KEY_ASR_NBEST_FINAL = 'n-best_final'
KEY_NBEST_IN_HISTORY = 'n-best_in_hist'
# endregion

# region signals
KEY_POPULARITY = 'popularity'
KEY_GLOBAL_COUNT = NAME_PREFIX_GLOBAL + KEY_COUNT
KEY_CUSTOMER_COUNT = NAME_PREFIX_CUSTOMER + KEY_COUNT
KEY_QUERY_GLOBAL_COUNT = NAME_PREFIX_QUERY + NAME_PREFIX_GLOBAL + KEY_COUNT
KEY_QUERY_CUSTOMER_COUNT = NAME_PREFIX_QUERY + NAME_PREFIX_CUSTOMER + KEY_COUNT
KEY_REWRITE_GLOBAL_COUNT = NAME_PREFIX_REWRITE + NAME_PREFIX_GLOBAL + KEY_COUNT
KEY_REWRITE_CUSTOMER_COUNT = NAME_PREFIX_REWRITE + NAME_PREFIX_CUSTOMER + KEY_COUNT
KEY_CPDR3 = 'cpdr3'
KEY_CPDR6 = 'cpdr6'
KEY_CPDR7 = 'cpdr7'
KEY_CPDR7_1 = 'cpdr7_1'
KEY_DEFECT = 'defect'
KEY_CPD_SCORE = 'cpd_score'
KEY_CPD_VERSION = 'cpd_version'
KEY_REWRITE_DEFECT = 'rewrite_defect'
KEY_QUERY_DEFECT = 'query_defect'
KEY_DEFECT_DELTA = 'defect_delta'
KEY_SESSION_DEFECT6 = 'session_defect6'
KEY_SESSION_DEFECT7 = 'session_defect7'
KEY_SESSION_DEFECT7_1 = 'session_defect7_1'
KEY_SESSION_DEFECT = 'session_defect'
KEY_DEFECT_BARGEIN = 'defect_bargeIn'
KEY_DEFECT_REPHRASE = 'defect_rephrase'
KEY_DEFECT_TERMINATION = 'defect_termination'
KEY_DEFECT_SANDPAPER = 'defect_sandpaper'
KEY_DEFECT_UNHANDLED = 'defect_unhandled'
KEY_AVG_DEFECT = NAME_PREFIX_AVG + KEY_DEFECT  # avg_defect
KEY_GLOBAL_AVG_DEFECT = NAME_PREFIX_GLOBAL + KEY_AVG_DEFECT
KEY_CUSTOMER_AVG_DEFECT = NAME_PREFIX_CUSTOMER + KEY_AVG_DEFECT
KEY_QUERY_GLOBAL_AVG_DEFECT = NAME_PREFIX_QUERY + NAME_PREFIX_GLOBAL + KEY_AVG_DEFECT
KEY_QUERY_CUSTOMER_AVG_DEFECT = NAME_PREFIX_QUERY + NAME_PREFIX_CUSTOMER + KEY_AVG_DEFECT
KEY_REWRITE_GLOBAL_AVG_DEFECT = NAME_PREFIX_REWRITE + NAME_PREFIX_GLOBAL + KEY_AVG_DEFECT
KEY_REWRITE_CUSTOMER_AVG_DEFECT = NAME_PREFIX_REWRITE + NAME_PREFIX_CUSTOMER + KEY_AVG_DEFECT
# endregion

# region keys for pairwise utterances
KEY_SUFFIX_RATIO = '_ratio'
KEY_SUFFIX_NON_OVERLAP = '_non_overlap'
KEY_REQUEST_FIRST = KEY_REQUEST + NAME_SUFFIX_FIRST
KEY_REQUEST_SECOND = KEY_REQUEST + NAME_SUFFIX_SECOND
KEY_TURN_PAIR_NON_OVERLAP = 'turn_pair_non_overlap'
KEY_REPLACED_REQUEST_FIRST = KEY_REPLACED_REQUEST + NAME_SUFFIX_FIRST
KEY_REPLACED_REQUEST_SECOND = KEY_REPLACED_REQUEST + NAME_SUFFIX_SECOND
NAME_PREFIX_TURN_PAIR_GLOBAL = 'turn_pair_global_request_'
NAME_PREFIX_TURN_PAIR_CUSTOMER = 'turn_pair_customer_request_'
KEY_TURN_PAIR_GLOBAL_FIRST = NAME_PREFIX_TURN_PAIR_GLOBAL + NAME_SUFFIX_FIRST[1:]
KEY_TURN_PAIR_GLOBAL_SECOND = NAME_PREFIX_TURN_PAIR_GLOBAL + NAME_SUFFIX_SECOND[1:]
KEY_TURN_PAIR_EDIT_DISTANCE = 'turn_pair_edit_distance'
KEY_TURN_PAIR_SLOT_EDIT_DISTANCE = 'turn_pair_slot_edit_distance'
KEY_TURN_PAIR_NON_OVERLAP_EDIT_DISTANCE = 'turn_pair_non_overlap_edit_distance'
KEY_TURN_PAIR_EDIT_DISTANCE_RATIO = KEY_TURN_PAIR_EDIT_DISTANCE + KEY_SUFFIX_RATIO
KEY_TURN_PAIR_SLOT_EDIT_DISTANCE_RATIO = KEY_TURN_PAIR_SLOT_EDIT_DISTANCE + KEY_SUFFIX_RATIO
KEY_TURN_PAIR_SLOT_EDIT_DISTANCE_RATIO_NON_OVERLAP = KEY_TURN_PAIR_SLOT_EDIT_DISTANCE + KEY_SUFFIX_RATIO + KEY_SUFFIX_NON_OVERLAP
KEY_TURN_PAIR_TOKEN_SORTED_EDIT_DISTANCE = 'turn_pair_token_sorted_edit_distance'
KEY_TIME_LAG = 'time_lag'
KEY_REQUEST_SECOND_IN_ASRNBEST = f'{KEY_REQUEST_SECOND}_in_nbest'
KEY_GLOBAL_SUM_REQUEST_SECOND_IN_ASRNBEST = NAME_PREFIX_GLOBAL_SUM + KEY_REQUEST_SECOND_IN_ASRNBEST
KEY_CUSTOMER_SUM_REQUEST_SECOND_IN_ASRNBEST = NAME_PREFIX_CUSTOMER_SUM + KEY_REQUEST_SECOND_IN_ASRNBEST
KEY_SUM_REQUEST_SECOND_IN_ASRNBEST = NAME_PREFIX_SUM + f'{KEY_REQUEST_SECOND}_in_nbest'
KEY_INDEX_OFFSET = 'index_offset'
KEY_UTTERANCE_ID_FIRST = KEY_UTTERANCE_ID + NAME_SUFFIX_FIRST
KEY_UTTERANCE_ID_SECOND = KEY_UTTERANCE_ID + NAME_SUFFIX_SECOND
KEY_DIALOG_ID_FIRST = KEY_DIALOG_ID + NAME_SUFFIX_FIRST
KEY_DIALOG_ID_SECOND = KEY_DIALOG_ID + NAME_SUFFIX_SECOND
KEY_ASR_HYPOTHESIS_FIRST = KEY_ASR_HYPOTHESIS + NAME_SUFFIX_FIRST
KEY_ASR_HYPOTHESIS_SECOND = KEY_ASR_HYPOTHESIS + NAME_SUFFIX_SECOND
KEY_NLU_HYPOTHESIS_FIRST = KEY_NLU_HYPOTHESIS + NAME_SUFFIX_FIRST
KEY_NLU_HYPOTHESIS_SECOND = KEY_NLU_HYPOTHESIS + NAME_SUFFIX_SECOND
KEY_HYPOTHESIS_FIRST = KEY_HYPOTHESIS + NAME_SUFFIX_FIRST
KEY_HYPOTHESIS_SECOND = KEY_HYPOTHESIS + NAME_SUFFIX_SECOND
KEY_INDEX_FIRST = KEY_INDEX + NAME_SUFFIX_FIRST
KEY_INDEX_SECOND = KEY_INDEX + NAME_SUFFIX_SECOND
KEY_DOMAIN_FIRST = KEY_DOMAIN + NAME_SUFFIX_FIRST
KEY_DOMAIN_SECOND = KEY_DOMAIN + NAME_SUFFIX_SECOND
KEY_INTENT_FIRST = KEY_INTENT + NAME_SUFFIX_FIRST
KEY_INTENT_SECOND = KEY_INTENT + NAME_SUFFIX_SECOND
KEY_RESPONSE_FIRST = KEY_RESPONSE + NAME_SUFFIX_FIRST
KEY_RESPONSE_SECOND = KEY_RESPONSE + NAME_SUFFIX_SECOND
KEY_TIMESTAMP_FIRST = KEY_TIMESTAMP + NAME_SUFFIX_FIRST
KEY_TIMESTAMP_SECOND = KEY_TIMESTAMP + NAME_SUFFIX_SECOND
KEY_GLOBAL_AVG_DEFECT_FIRST = KEY_GLOBAL_AVG_DEFECT + NAME_SUFFIX_FIRST
KEY_GLOBAL_AVG_DEFECT_SECOND = KEY_GLOBAL_AVG_DEFECT + NAME_SUFFIX_SECOND
KEY_CUSTOMER_AVG_DEFECT_FIRST = KEY_CUSTOMER_AVG_DEFECT + NAME_SUFFIX_FIRST
KEY_CUSTOMER_AVG_DEFECT_SECOND = KEY_CUSTOMER_AVG_DEFECT + NAME_SUFFIX_SECOND
KEY_DEFECT_FIRST = KEY_DEFECT + NAME_SUFFIX_FIRST
KEY_DEFECT_SECOND = KEY_DEFECT + NAME_SUFFIX_SECOND
KEY_GLOBAL_COUNT_FIRST = KEY_GLOBAL_COUNT + NAME_SUFFIX_FIRST
KEY_GLOBAL_COUNT_SECOND = KEY_GLOBAL_COUNT + NAME_SUFFIX_SECOND
KEY_CUSTOMER_COUNT_FIRST = KEY_CUSTOMER_COUNT + NAME_SUFFIX_FIRST
KEY_CUSTOMER_COUNT_SECOND = KEY_CUSTOMER_COUNT + NAME_SUFFIX_SECOND
KEY_NLU_MERGER_RESULT_FIRST = KEY_NLU_MERGER_RESULT + NAME_SUFFIX_FIRST
KEY_NLU_MERGER_RESULT_SECOND = KEY_NLU_MERGER_RESULT + NAME_SUFFIX_SECOND
KEY_MAX_TIMESTAMP_FIRST = KEY_MAX_TIMESTAMP + NAME_SUFFIX_FIRST
KEY_MAX_TIMESTAMP_SECOND = KEY_MAX_TIMESTAMP + NAME_SUFFIX_SECOND
# endregion

# region keys for pairwise signals (DEPRECATED)
NAME_PREFIX_TURN_PAIR = 'turn_pair_'
NAME_PREFIX_GLOBAL_TURN_PAIR = NAME_PREFIX_GLOBAL + NAME_PREFIX_TURN_PAIR
NAME_PREFIX_CUSTOMER_TURN_PAIR = NAME_PREFIX_CUSTOMER + NAME_PREFIX_TURN_PAIR
NAME_PREFIX_GLOBAL_TURN_PAIR_AVG = NAME_PREFIX_GLOBAL_TURN_PAIR + NAME_PREFIX_AVG
NAME_PREFIX_CUSTOMER_TURN_PAIR_AVG = NAME_PREFIX_CUSTOMER_TURN_PAIR + NAME_PREFIX_AVG

KEY_GLOBAL_TURN_PAIR_AVG_DEFECT_FIRST = \
    NAME_PREFIX_GLOBAL_TURN_PAIR_AVG + KEY_DEFECT_FIRST
KEY_GLOBAL_TURN_PAIR_AVG_DEFECT_SECOND = \
    NAME_PREFIX_GLOBAL_TURN_PAIR_AVG + KEY_DEFECT_SECOND
KEY_CUSTOMER_TURN_PAIR_AVG_DEFECT_FIRST = \
    NAME_PREFIX_CUSTOMER_TURN_PAIR_AVG + KEY_DEFECT_FIRST
KEY_CUSTOMER_TURN_PAIR_AVG_DEFECT_SECOND = \
    NAME_PREFIX_CUSTOMER_TURN_PAIR_AVG + KEY_DEFECT_SECOND
KEY_GLOBAL_TURN_PAIR_AVG_TIME_LAG = NAME_PREFIX_GLOBAL_TURN_PAIR_AVG + KEY_TIME_LAG
KEY_CUSTOMER_TURN_PAIR_AVG_TIME_LAG = NAME_PREFIX_CUSTOMER_TURN_PAIR_AVG + KEY_TIME_LAG

KEY_GLOBAL_TURN_PAIR_COUNT = NAME_PREFIX_GLOBAL_TURN_PAIR + KEY_COUNT
KEY_CUSTOMER_TURN_PAIR_COUNT = NAME_PREFIX_CUSTOMER_TURN_PAIR + KEY_COUNT
KEY_TURN_PAIR_POPULARITY = NAME_PREFIX_TURN_PAIR + KEY_POPULARITY

KEY_GLOBAL_TURN_PAIR_SOURCE_COUNT = NAME_PREFIX_GLOBAL_TURN_PAIR + 'src_count'
KEY_CUSTOMER_TURN_PAIR_SOURCE_COUNT = NAME_PREFIX_CUSTOMER_TURN_PAIR + 'src_count'
KEY_GLOBAL_TURN_PAIR_RATIO = NAME_PREFIX_GLOBAL_TURN_PAIR + 'ratio'
KEY_CUSTOMER_TURN_PAIR_RATIO = NAME_PREFIX_CUSTOMER_TURN_PAIR + 'ratio'

# endregion

# region keys for pairwise signals
NAME_PREFIX_PAIR = 'pair_'
NAME_PREFIX_GLOBAL_PAIR = NAME_PREFIX_GLOBAL + NAME_PREFIX_PAIR
NAME_PREFIX_CUSTOMER_PAIR = NAME_PREFIX_CUSTOMER + NAME_PREFIX_PAIR
NAME_PREFIX_GLOBAL_PAIR_AVG = NAME_PREFIX_GLOBAL_PAIR + NAME_PREFIX_AVG
NAME_PREFIX_CUSTOMER_PAIR_AVG = NAME_PREFIX_CUSTOMER_PAIR + NAME_PREFIX_AVG

KEY_GLOBAL_PAIR_AVG_DEFECT_FIRST = \
    NAME_PREFIX_GLOBAL_PAIR_AVG + KEY_DEFECT_FIRST
KEY_GLOBAL_PAIR_AVG_DEFECT_SECOND = \
    NAME_PREFIX_GLOBAL_PAIR_AVG + KEY_DEFECT_SECOND
KEY_CUSTOMER_PAIR_AVG_DEFECT_FIRST = \
    NAME_PREFIX_CUSTOMER_PAIR_AVG + KEY_DEFECT_FIRST
KEY_CUSTOMER_PAIR_AVG_DEFECT_SECOND = \
    NAME_PREFIX_CUSTOMER_PAIR_AVG + KEY_DEFECT_SECOND
KEY_CUSTOMER_PAIR_MAX_TIMESTAMP_FIRST = \
    NAME_PREFIX_CUSTOMER_PAIR + KEY_MAX_TIMESTAMP_FIRST
KEY_CUSTOMER_PAIR_MAX_TIMESTAMP_SECOND = \
    NAME_PREFIX_CUSTOMER_PAIR + KEY_MAX_TIMESTAMP_SECOND
KEY_GLOBAL_PAIR_AVG_TIME_LAG = NAME_PREFIX_GLOBAL_PAIR_AVG + KEY_TIME_LAG
KEY_CUSTOMER_PAIR_AVG_TIME_LAG = NAME_PREFIX_CUSTOMER_PAIR_AVG + KEY_TIME_LAG

KEY_GLOBAL_PAIR_COUNT = NAME_PREFIX_GLOBAL_PAIR + KEY_COUNT
KEY_CUSTOMER_PAIR_COUNT = NAME_PREFIX_CUSTOMER_PAIR + KEY_COUNT
KEY_PAIR_POPULARITY = NAME_PREFIX_PAIR + KEY_POPULARITY

KEY_GLOBAL_TRANSITION_SOURCE_COUNT = NAME_PREFIX_GLOBAL + 'trans_src_count'
KEY_CUSTOMER_TRANSITION_SOURCE_COUNT = NAME_PREFIX_CUSTOMER + 'trans_src_count'
KEY_GLOBAL_TRANSITION_PROB = NAME_PREFIX_GLOBAL + 'trans_prob'
KEY_CUSTOMER_TRANSITION_PROB = NAME_PREFIX_CUSTOMER + 'trans_prob'
KEY_GLOBAL_TRANSITION_SOURCE_COUNT_EXCLUDE_SELF_LOOP = NAME_PREFIX_GLOBAL + 'trans_src_count_exclude_self_loop'
KEY_CUSTOMER_TRANSITION_SOURCE_COUNT_EXCLUDE_SELF_LOOP = NAME_PREFIX_CUSTOMER + 'trans_src_count_exclude_self_loop'
KEY_GLOBAL_TRANSITION_PROB_EXCLUDE_SELF_LOOP = NAME_PREFIX_GLOBAL + 'trans_prob_exclude_self_loop'
KEY_CUSTOMER_TRANSITION_PROB_EXCLUDE_SELF_LOOP = NAME_PREFIX_CUSTOMER + 'trans_prob_exclude_self_loop'

KEY_CUSTOMER_TRANSITION_COUNT = 'customer_trans_count'
KEY_GLOBAL_TRANSITION_COUNT = 'global_trans_count'

# endregion

# region key groups
KEYS_IMPRESSION_COUNTS = (KEY_GLOBAL_COUNT, KEY_CUSTOMER_COUNT)
KEYS_DEFECTIVE_SIGNALS = (
    KEY_DEFECT_BARGEIN,
    KEY_DEFECT_REPHRASE,
    KEY_DEFECT_TERMINATION,
    KEY_DEFECT_UNHANDLED,
    KEY_DEFECT_SANDPAPER,
)
KEYS_DETAILED_AUS_AND_MERGER_DATA = (KEY_AUS_CRITICAL, KEY_NLU_MERGER_RULE)
KEYS_ALL_CPDR_VERSIONS = (KEY_CPDR3, KEY_CPDR6, KEY_CPDR7)

KEYS_TURN_PAIR_AGG_SHARED_COLS = (
    KEY_CUSTOMER_ID,
    KEY_LOCALE,
    KEY_NUM_TURNS,
    KEY_DEVICE_ID,
    KEY_DEVICE_TYPE,
    KEY_SESSION_ID,
)

KEYS_REPHRASE_TRAFFIC_NON_OCCURRENCE_COLS = KEYS_TURN_PAIR_AGG_SHARED_COLS + (
    KEY_GLOBAL_COUNT,
    KEY_CUSTOMER_COUNT,
    KEY_TURN_PAIR_EDIT_DISTANCE,
    KEY_TURN_PAIR_TOKEN_SORTED_EDIT_DISTANCE,
    KEY_TURN_PAIR_GLOBAL_FIRST,
    KEY_TURN_PAIR_GLOBAL_SECOND,
)
# endregion

# key affinity
KEY_CUSTOMER_ID_LIST = 'customer_id_list'
KEY_NUM_CUSTOMERS = 'num_customers'
KEY_CUSTOMER_AFFINITY = 'customer_affinity'
# endregion

# endregion


# region entity traffic keys
KEY_ENTITY_TYPE = 'entity_type'
KEY_ENTITY_TYPES = 'entity_types'
KEY_ENTITY_INDEX = 'entity_index'
KEY_SONG_NAME = 'song_name'
KEY_ARTIST_NAME = 'artist_name'
KEY_ENTITY = 'entity'
KEY_ENTITIES = 'entities'
KEY_HAS_TARGET_ENTITY = 'has_compare_target_slot_type'
KEY_UTTERANCE_TEMPLATE = 'utterance_template'
KEY_ENTITY_TYPE_FIRST = KEY_ENTITY_TYPE + NAME_SUFFIX_FIRST
KEY_ENTITY_TYPE_SECOND = KEY_ENTITY_TYPE + NAME_SUFFIX_SECOND
KEY_ENTITY_FIRST = KEY_ENTITY + NAME_SUFFIX_FIRST
KEY_ENTITY_SECOND = KEY_ENTITY + NAME_SUFFIX_SECOND
KEY_ENTITY_INDEX_FIRST = KEY_ENTITY_INDEX + NAME_SUFFIX_FIRST
KEY_ENTITY_INDEX_SECOND = KEY_ENTITY_INDEX + NAME_SUFFIX_SECOND
KEY_RELATION = 'relation'
KEY_REVERSED_RELATION = 'reversed_relation'
KEY_STATS_ENTITY_TYPE_CUSTOMER_DOMAIN_INTENT = 'type_customer_domain_intent_stats'
KEY_STATS_ENTITY_CUSTOMER_DOMAIN_INTENT = 'customer_domain_intent_stats'
KEY_STATS_ENTITY_TYPE_CUSTOMER_DOMAIN = 'type_customer_domain_stats'
KEY_STATS_ENTITY_CUSTOMER_DOMAIN = 'customer_domain_stats'
KEY_STATS_ENTITY_TYPE_CUSTOMER = 'type_customer_stats'
KEY_STATS_ENTITY_CUSTOMER = 'customer_stats'
KEY_ENTITY_ALIGNMENT_SCORE = 'entity_alignment_score'
# endregion

# region label & tag keys
KEY_LABEL_FIRST = KEY_LABEL + NAME_SUFFIX_FIRST
KEY_LABEL_SECOND = KEY_LABEL + NAME_SUFFIX_SECOND
# endregion

# region dataframe grouping keys
GROUP_KEYS__REQ_HYP = (KEY_REQUEST, KEY_HYPOTHESIS)
GROUP_KEYS__REQ_HYP_DOMAIN = (KEY_REQUEST, KEY_HYPOTHESIS, KEY_DOMAIN)
GROUP_KEYS__CUSTOMER_REQ_HYP_DOMAIN = (KEY_CUSTOMER_ID, KEY_REQUEST, KEY_HYPOTHESIS, KEY_DOMAIN)
GROUP_KEYS__CUSTOMER_REQ_HYP_DOMAIN_INTENT = (
    KEY_CUSTOMER_ID,
    KEY_REQUEST,
    KEY_HYPOTHESIS,
    KEY_DOMAIN,
    KEY_INTENT,
)
GROUP_KEYS__CUSTOMER_REQ_HYP = (KEY_CUSTOMER_ID, KEY_REQUEST, KEY_HYPOTHESIS)
GROUP_KEYS__CUSTOMER_REQ_HYP_DOMAIN_PROVIDER = (
    KEY_CUSTOMER_ID,
    KEY_REQUEST,
    KEY_HYPOTHESIS,
    KEY_DOMAIN,
    KEY_INTENT,
    KEY_REPLACED_REQUEST,
    KEY_NLU_MERGER_RESULT,
    KEY_PROVIDER_NAME,
    KEY_ASR_HYPOTHESIS,
)
GROUP_KEYS__ENTITY_TYPE_PAIR_CUSTOMER_DOMAIN_INTENT = [
    KEY_ENTITY_TYPE,
    KEY_ENTITY,
    KEY_CUSTOMER_ID,
    KEY_DOMAIN,
    KEY_INTENT,
]
GROUP_KEYS__ENTITY_CUSTOMER_DOMAIN_INTENT = [
    KEY_ENTITY,
    KEY_CUSTOMER_ID,
    KEY_DOMAIN,
    KEY_INTENT,
]
GROUP_KEYS__ENTITY_TYPE_PAIR_CUSTOMER_DOMAIN = [
    KEY_ENTITY_TYPE,
    KEY_ENTITY,
    KEY_CUSTOMER_ID,
    KEY_DOMAIN,
]
GROUP_KEYS__ENTITY_CUSTOMER_DOMAIN = [KEY_ENTITY, KEY_CUSTOMER_ID, KEY_DOMAIN]
GROUP_KEYS__ENTITY_TYPE_PAIR_CUSTOMER = [KEY_ENTITY_TYPE, KEY_ENTITY, KEY_CUSTOMER_ID]
GROUP_KEYS__ENTITY_CUSTOMER = [KEY_ENTITY, KEY_CUSTOMER_ID]
GROUP_KEYS__ENTITY_TYPE_PAIR = [KEY_ENTITY_TYPE, KEY_ENTITY]
# endregion

# region dataframe struct keys
STRUCT_FIELDS__OCCURRENCES = (
    KEY_UTTERANCE_ID,
    KEY_TIMESTAMP,
    KEY_RESPONSE,
    KEY_LOCALE,
    KEY_DFS_SHOULD_TRIGGER,
    KEY_DFS_SOURCE,
    KEY_RESOLVED_SLOTS,
    KEY_ASR_NBEST,
    KEY_AUS_REQUEST,
    KEY_AUS_HYPOTHESIS,
)
# endregion

# region provider names
PROVIDER_NAME__PDFS = 'FlareDFSPersonalized'
PROVIDER_NAME__PDFS_PRECOMPUTE = 'FlareDFSPersonalizedPrecompute'
PROVIDER_NAME__YANGTZE = 'Yangtze'
PROVIDER_NAME__GCRD = 'FlareDFSContextualRephrase'
PROVIDER_NAME__GDFS = 'FlareDFSGlobal'
PROVIDER_NAME__GLOBAL_GRAPH = 'GlobalGraph'
PROVIDER_NAME__FLARE_INFORMATION = 'FlareInformation'
# endregion