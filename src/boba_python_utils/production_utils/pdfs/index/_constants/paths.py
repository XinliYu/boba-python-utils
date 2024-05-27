from boba_python_utils.production_utils.common.path import (
    get_path_prefix_dict
)
from boba_python_utils.production_utils.common.constants import PreDefinedBuckets, SupportedOwners, SupportedRegions
from enum import Enum

PRODUCT_NAME_PDFS = 'pDFS'

DIRNAME_COMBINED_DATA = 'combined'


class PreDefinedDailyAggregationTypes(str, Enum):
    P0ProductionAggregation = 'prod'
    TrafficAggregation = 'traffic'
    TurnPairsAggregation = 'turn_pairs'
    TurnPairsAggregationFiltered = 'turn_pairs_filtered'
    UtteranceIds = 'uid'


class PreDefinedIndexDataTypes(str, Enum):
    CUSTOMER_HISTORY_INDEX = 'customer_history'
    UTTERANCE_INDEX = 'utterance_index'
    ENTITY_INDEX = 'entity_index'
    REWRITE_INDEX = 'rewrite_index'
    GLOBAL_AFFINITY_INDEX = 'global_affinity_index'
    CUSTOMER_AFFINITY_INDEX = 'customer_affinity_index'
    CUSTOMER_HISTORY_AFFINITY_INDEX = 'customer_history_affinity_index'
    REPHRASE_TRAFFIC = 'rephrase_traffic'
    ENTITY_TRAFFIC = 'entity_traffic'
    ENTITY_REPHRASE = 'entity_rephrase'
    TURN_PAIR_STATS = 'turn_pair_stats'
    TURN_PAIR_INDEX = 'turn_pair_index'
    # region testsets
    AGGREGATED_TRAFFIC = 'aggregated_traffic'
    REPHRASE_TESTSET = 'rephrase_testset'
    REPHRASE_TESTSET_NLU = 'rephrase_testset_nlu'
    PERSONALIZED_TESTSET = 'personalized_testset'
    PERSONALIZED_TESTSET_NLU = 'personalized_testset_nlu'
    GUARDRAIL_TESTSET_ENTITY_SWAP = 'guardrail_testset_entity_swap'
    GUARDRAIL_TESTSET_ENTITY_SWAP_WITH_FULL_HISTORY = 'guardrail_testset_entity_swap_with_full_history'
    LIVETRAFFIC_EVALUATION_DATA = 'livetraffic_evaluation_data'
    CONTEXT_DATA = 'context_data'
    # endregion


PDFS_AESLU_AGGREGATION_S3_PREFIX_NA = 's3://hoverboard-shared-aeslu-datasets-us-east-1'

SLAB_AGGREGATION_S3_PREFIXS = get_path_prefix_dict(
    {
        'bucket': PreDefinedBuckets.AbmlWorkspaces,
        'owner': SupportedOwners.AESLU,
        'region': SupportedRegions.NA,
        'prefix': PDFS_AESLU_AGGREGATION_S3_PREFIX_NA
    }
)

PDFS_AGGREGATION_S3_PREFIX_EN_US_LEGACY = 's3abml://abml-workspaces-na/gw_traffic/data'
AGGREGATION_ROOT_PATH_PATTERN_SCIENTIST_LEGACY = '{prefix}/{version}/{data}/{locale}'
AGGREGATION_ROOT_PATH_PATTERN_SCIENTIST_LEGACY_NO_VERSION = '{prefix}/{data}/{locale}'
