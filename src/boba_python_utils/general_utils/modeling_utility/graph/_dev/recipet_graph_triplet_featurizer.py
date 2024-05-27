from typing import Union, List

from attr import attrs, attrib
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, FloatType
import boba_python_utils.spark_utils as sparku
import boba_python_utils.spark_utils.spark_functions as F
import boba_python_utils.production_utils.pdfs.constants as c
import boba_python_utils.production_utils.nlu as nlu
from boba_python_utils.general_utils.modeling_utility.feature_building.featurizer import SparkIndexedDataFeaturizer

MAJOR_MULTI_SOURCE_PROVIDERS = ['GlobalGraph', 'FlareDFSGlobal', 'FlareDFS', 'FlareDFSPersonalized']


def get_provider_label(provider_name):
    if not provider_name:
        return 0

    if provider_name in MAJOR_MULTI_SOURCE_PROVIDERS:
        return MAJOR_MULTI_SOURCE_PROVIDERS.index(provider_name) + 1
    else:
        return len(MAJOR_MULTI_SOURCE_PROVIDERS) + 1


def cal_token_diff(query, utt):
    return len(set(utt.split()).difference(set(query.split())))


def get_slot_value_exact_match(query, hyp):
    _, _, slots = nlu.get_domain_intent_slots_from_hypothesis(hyp, slots_as_types_and_values=True)
    if not slots:
        return 0.0

    slot_value_exact_match = 0
    slot_values = slots[1]
    for slot_value in slot_values:
        if slot_value in query:
            slot_value_exact_match += 1
    slot_value_exact_match = slot_value_exact_match / float(len(slot_values))
    return slot_value_exact_match


def get_slot_value_overlap(query, hyp):
    # online code: https://code.amazon.com/packages/AlexaDeepFeedbackSearchRanker/blobs/1.1_pdfs_p1_l2_ranker/--/src/alexa_deep_feedback_search/featurizer/slot_match_featurizer.py
    # TODO: slight difference
    _, _, slots = nlu.get_domain_intent_slots_from_hypothesis(hyp, slots_as_types_and_values=True)
    if not slots:
        return 0.0
    slot_value_tokens = [token for slot_value in slots[1] for token in slot_value.split()]
    # ? in the raw script, it is `len(slot_value_tokens)`
    #     as the denominator; but feels it should be `len(set(slot_value_tokens))`
    slot_value_overlap = len(set(slot_value_tokens).intersection(set(query.split()))) / float(
        len(slot_value_tokens)
    )
    return slot_value_overlap


def get_slot_char_value_overlap(query, hyp):
    _, _, slots = nlu.get_domain_intent_slots_from_hypothesis(hyp, slots_as_types_and_values=True)
    if not slots:
        return 0.0

    slot_value = ''.join(''.join(slot_value.split()) for slot_value in slots[1])
    if not slot_value:
        return 0.0

    slot_value = set(slot_value)
    return len(slot_value.intersection(set(query))) / float(len(slot_value))


@attrs(slots=False)
class PdfsP1L2MiscFeaturizer(SparkIndexedDataFeaturizer):
    """
    Featurizer for misc features that are already applied in P0 L2 ranker,
        like global/customer impression and defect signals, embedding similarities;
    also includes some additional features like 'char_length_change', 'rewrite_slot_number'.
    """
    embedding_model_names = attrib(type=Union[str, List[str]], default=None)
    embedding_similarity_feature_name_patterns = attrib(type=Union[str, List[str]], default=None)

    def __attrs_post_init__(self):
        super(PdfsP1L2MiscFeaturizer, self).__attrs_post_init__()
        self.requires_flat_data = True
        if isinstance(self.embedding_model_names, str):
            self.embedding_model_names = [self.embedding_model_names]

        if self.embedding_similarity_feature_name_patterns is None:
            self.embedding_similarity_feature_name_patterns = self.embedding_model_names
        elif isinstance(self.embedding_similarity_feature_name_patterns, str):
            self.embedding_similarity_feature_name_patterns = [
                self.embedding_similarity_feature_name_patterns
            ]

    def _get_features(
            self,
            data: DataFrame,
            feature_id_keys: List[str],
            data_id_field_name: str,
            index_item_id_field_name: str = None,
            index_list_field_name: str = None,
            is_index_only_data=False
    ):
        pass
