import json
from collections import defaultdict
from datetime import datetime
from functools import partial

from pyspark.sql.types import DoubleType, IntegerType

import boba_python_utils.spark_utils.spark_functions as F
from boba_python_utils.production_utils.greenwich_data.constants import (
    TIMESTAMP_FORMAT,
    TIMESTAMP_FORMAT_ALTERNATIVE,
)


def bool2str(val):
    if val is True:
        return 'true'
    elif val is False:
        return 'false'
    elif val == 'true' or val == 'false':
        return val


def get_device_id_from_utterance_id(utterance_id):
    splits = utterance_id.split('/')
    if len(splits) > 5:
        return splits[5]


get_device_id_from_utterance_id_udf = F.udf(get_device_id_from_utterance_id)


def get_device_type_from_utterance_id(utterance_id):
    return utterance_id.split('/')[0].split(':')[0]


get_device_type_from_utterance_id_udf = F.udf(get_device_type_from_utterance_id)


def get_provider(aus_merger_results):
    if aus_merger_results is None or len(aus_merger_results) == 0:
        return None
    for i in aus_merger_results:
        if i is None or len(i) == 0:
            continue
        else:
            return i[0]['metadata']['providerName']
    return None


get_provider_udf = F.udf(get_provider)


def get_dfs_source(aus_merger_results):
    if aus_merger_results is None or len(aus_merger_results) == 0:
        return None
    for i in aus_merger_results:
        if i is None or len(i) == 0:
            continue
        else:
            dfs_source = i[0]['metadata']['features']['DFS_SOURCE']
            if dfs_source is not None:
                return dfs_source
    return None


get_dfs_source_udf = F.udf(get_dfs_source)


def generate_hypothesis(domain, intent, token_label_text):
    if not token_label_text:
        if not intent:
            return domain
        else:
            return domain + '|' + intent
    tokens_by_label = defaultdict(list)
    tokens = token_label_text.split(' ')
    for token in tokens:
        text_and_label = token.split('|')
        if len(text_and_label) != 2:
            continue
        text, label = text_and_label
        if label == 'Other':
            continue
        tokens_by_label[label].append(text)
    return '|'.join(
        [domain, intent]
        + ["{}:{}".format(x[0], ' '.join(x[1])) for x in sorted(tokens_by_label.items())]
    )


generate_hypothesis_udf = F.udf(generate_hypothesis)


def extract_resolved_slots(slots):
    slots_txt = ""
    if slots is not None and len(slots) > 0:
        for slot in slots:
            slot_name = slot['slotName']
            slot_value = slot['slotValue']
            tokens = slot['tokens']
            if ('Date' in slot_name or 'MISSING' in slot_value) and tokens:
                slots_txt += slot_name + ":" + " ".join(tokens) + "|"
            else:
                slots_txt += slot_name + ":" + slot_value + "|"
    return slots_txt.strip("|")


extract_resolved_slots_udf = F.udf(extract_resolved_slots)


def extract_merger_rule(nlu_merger_rules):
    if not nlu_merger_rules:
        return None
    for item in nlu_merger_rules:
        if item['ruleGroup'] == 'MergerDecision':
            return item['ruleName']
    return None


def extract_merger_rule_version(nlu_merger_rules):
    if not nlu_merger_rules:
        return None
    for item in nlu_merger_rules:
        if item['ruleName'].startswith('LogVersion_'):
            return item['ruleName'][11:]
    return None


extract_merger_rule_udf = F.udf(extract_merger_rule)
extract_merger_rule_version_udf = F.udf(extract_merger_rule_version)


def extract_hypothesis_from_asr_aus_interpretation(asr_or_aus_interpretation):
    if not asr_or_aus_interpretation:
        return None
    if isinstance(asr_or_aus_interpretation, str):
        try:
            asr_or_aus_interpretation = json.loads(asr_or_aus_interpretation)
            if 'topInterpretation' not in asr_or_aus_interpretation:
                return None
        except:  # noqa: E722
            return None

    asr_or_aus_interpretation = asr_or_aus_interpretation['topInterpretation']
    if asr_or_aus_interpretation:
        domain = asr_or_aus_interpretation['scoredIntentLabel']['domainName']
        intent = asr_or_aus_interpretation['scoredIntentLabel']['intentName']
        token_label_text = asr_or_aus_interpretation['scoredSegmentation']['tokenLabelText']
        return generate_hypothesis(domain, intent, token_label_text)


extract_hypothesis_from_asr_aus_interpretation_udf = F.udf(
    extract_hypothesis_from_asr_aus_interpretation
)


def extract_utterance_from_asr_aus_interpretation(asr_or_aus_interpretation):
    if not asr_or_aus_interpretation:
        return None
    if isinstance(asr_or_aus_interpretation, str):
        try:
            asr_or_aus_interpretation = json.loads(asr_or_aus_interpretation)
            if 'topInterpretation' not in asr_or_aus_interpretation:
                return None
        except:  # noqa: E722
            return None

    asr_or_aus_interpretation = asr_or_aus_interpretation['topInterpretation']
    if asr_or_aus_interpretation:
        return asr_or_aus_interpretation['scoredUtterance']['utterance']


extract_utterance_from_asr_aus_interpretation_udf = F.udf(
    extract_utterance_from_asr_aus_interpretation
)


def extract_defect_from_nextgen(signals, version):
    if signals:
        for item in signals:
            if item['version'] == version:
                return float(item['value'])
    return None


def extract_defect_from_nextgen__(signals, version):
    return F.udf(partial(extract_defect_from_nextgen, version=version), returnType=DoubleType())(
        signals
    )


def extract_session_defect_for_next_gen(session_signals, version):
    if session_signals:
        for item in session_signals:
            if item['cpd_version'] == version:
                return item['total_signals']['total_defect']
    return None


def extract_session_defect_for_next_gen_udf(session_signals, version):
    return F.udf(
        partial(extract_session_defect_for_next_gen, version=version), returnType=DoubleType()
    )(session_signals)


def _get_time_from_str(s):
    if '.' in s:
        return datetime.strptime(s, TIMESTAMP_FORMAT)
    else:
        return datetime.strptime(s, TIMESTAMP_FORMAT_ALTERNATIVE)


def get_time_lag(t1, t2) -> int:
    if t1 is not None and t2 is not None:
        return (_get_time_from_str(t2) - _get_time_from_str(t1)).seconds  # noqa: E501


get_time_lag_udf = F.udf(get_time_lag, returnType=IntegerType())