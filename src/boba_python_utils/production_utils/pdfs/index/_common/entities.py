from boba_python_utils.production_utils._nlu.target_slots import is_for_comparison_slot_type


def get_utterance_template(utterance, entities):
    for entity_type, entity_value in entities:
        if is_for_comparison_slot_type(entity_type):
            utterance = utterance.replace(entity_value, entity_type)
    return utterance
