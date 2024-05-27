from typing import Iterable

from pyspark.sql.column import Column

from boba_python_utils.spark_utils.spark_functions.common import or_
from boba_python_utils.production_utils.pdfs import constants as c
from boba_python_utils.production_utils.pdfs.filters.configs import (
    CUSTOMER_ENTITY_DEFECT_GUARDRAIL_CONFIGS, EntityDefectGuardrailConfig
)


def customer_entity_defect_guardrail_filter(
        entity_type_colname: str = c.KEY_ENTITY_TYPE,
        customer_defect_colname: str = c.KEY_CUSTOMER_AVG_DEFECT,
        customer_count_colname: str = c.KEY_CUSTOMER_COUNT,
        configs: Iterable[EntityDefectGuardrailConfig] = CUSTOMER_ENTITY_DEFECT_GUARDRAIL_CONFIGS

) -> Column:
    return or_(
        config.get_filter_cond(
            entity_type_colname=entity_type_colname,
            defect_colname=customer_defect_colname,
            count_colname=customer_count_colname
        )
        for config in configs
    )
