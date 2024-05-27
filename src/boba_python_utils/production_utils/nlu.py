from pyspark.sql.column import Column

from boba_python_utils.spark_utils.spark_functions.common import col_
from boba_python_utils.production_utils._nlu.hypothesis_parsing import *  # noqa: F401,F403,E501
from boba_python_utils.production_utils._nlu.non_overlap import *  # noqa: F401,F403,E501
from boba_python_utils.production_utils._nlu.entity_pairing import *  # noqa: F401,F403,E501
from boba_python_utils.production_utils._nlu.target_slots import *  # noqa: F401,F403,E501
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, BooleanType
from boba_python_utils.production_utils.greenwich_data.constants import (
    KEY_ENTITY_TYPE,
    KEY_ENTITY,
)
from typing import Union, Dict

DOMAIN_PREFIX_3P = 'amzn1.'
DOMAIN_NAME_ALL_3P = '3p-skill'


# region domain
def is_third_party_skill_domain(domain_or_hyp: str):
    return domain_or_hyp is not None and domain_or_hyp.startswith(DOMAIN_PREFIX_3P)


def is_third_party_skill_domain_udf(domain_or_hyp: NameOrColumn):
    domain_or_hyp = col_(domain_or_hyp)
    return domain_or_hyp.isNotNull() & domain_or_hyp.startswith(DOMAIN_PREFIX_3P)


def get_domain(domain, third_party_skill_as_one_domain=True):
    if third_party_skill_as_one_domain and is_third_party_skill_domain(domain):
        return DOMAIN_NAME_ALL_3P
    else:
        return domain


def get_domain_udf(domain: NameOrColumn, third_party_skill_as_one_domain: bool = True):
    domain = col_(domain)
    if third_party_skill_as_one_domain:
        return F.when(
            is_third_party_skill_domain_udf(domain),
            F.lit(DOMAIN_NAME_ALL_3P)
        ).otherwise(domain)
    else:
        return domain


# endregion

# region extraction from hypothesis
def _get_slots_dict(hyp_splits):
    slots = {}
    for slotstr in sorted(hyp_splits[2:]):
        slot_splits = slotstr.split(':', 1)
        if len(slot_splits) == 1:
            slots[slot_splits[0].strip()] = ''
        else:
            slots[slot_splits[0].strip()] = slot_splits[1].strip()
    return slots


def _get_slots_types_values(hyp_splits):
    slot_types, slot_values = [], []
    for slotstr in sorted(hyp_splits[2:]):
        slot_splits = slotstr.split(':', 1)
        if len(slot_splits) == 1:
            slot_types.append(slot_splits[0].strip())
            slot_values.append(None)
        else:
            slot_types.append(slot_splits[0].strip())
            slot_values.append(slot_splits[1].strip())
    return slot_types, slot_values


def get_domain_intent_from_splits(hyp_splits, third_party_skill_as_one_domain=False):
    domain = solve_domain(
        hyp_splits[0], third_party_skill_as_one_domain=third_party_skill_as_one_domain
    )
    intent = hyp_splits[1] if len(hyp_splits) > 1 else None
    return domain, intent


def get_domain_intent_slots_from_splits(
        hyp_splits,
        slots_as_dict=False,
        slots_as_types_and_values=False,
        third_party_skill_as_one_domain=False,
        return_empty_dict_if_no_slots=False,
):
    domain = solve_domain(
        hyp_splits[0], third_party_skill_as_one_domain=third_party_skill_as_one_domain
    )
    if len(hyp_splits) == 2:
        return domain, hyp_splits[1], ({} if return_empty_dict_if_no_slots else None)
    else:
        if slots_as_dict:
            return domain, hyp_splits[1], _get_slots_dict(hyp_splits)
        elif slots_as_types_and_values:
            return domain, hyp_splits[1], _get_slots_types_values(hyp_splits)
        else:
            return domain, hyp_splits[1], hyp_splits[2:]


def get_domain_intent_slots(
        hyp,
        slots_as_dict=False,
        slots_as_types_and_values=False,
        third_party_skill_as_one_domain=False,
        return_empty_dict_if_no_slots=False,
):
    if not hyp:
        return None, None, ({} if return_empty_dict_if_no_slots else None)
    hyp_splits = hyp.split('|')
    return get_domain_intent_slots_from_hypothesis_splits(
        hypothesis_splits=hyp_splits,
        slots_as_dict=slots_as_dict,
        slots_as_types_and_values=slots_as_types_and_values,
        third_party_skill_as_one_domain=third_party_skill_as_one_domain,
    )


def get_slots(hyp: str, return_empty_dict_if_no_slots=False) -> Union[Dict[str, str], None]:
    if hyp:
        hyp_splits = hyp.split('|')
        if return_empty_dict_if_no_slots or len(hyp_splits) > 2:
            return _get_slots_dict(hyp_splits)
        else:
            return None
    else:
        return {} if return_empty_dict_if_no_slots else None


# endregion

# region entities


def concat_entity_type_and_value(entity_type: str, entity_value: str) -> str:
    return f'{entity_type}:{entity_value}'


concat_entity_type_and_value_udf = F.udf(concat_entity_type_and_value, returnType=StringType())


# endregion


def sort_hypothesis_slots(hyp):
    if hyp:
        hyp = hyp.strip()
        if hyp[-1] == '|':
            hyp = hyp[:-1]
        if hyp:
            splits = hyp.split('|')
            return '|'.join(splits[:2] + sorted(splits[2:]))

    return None


def has_slots(hyp: str) -> bool:
    return hyp is not None and len(hyp.split('|')) > 2


@F.udf(returnType=BooleanType())
def has_slots_udf(hyp):
    return F.col(hyp).isNotNull() & (F.size(F.split(hyp, r'\|')) > 2)
