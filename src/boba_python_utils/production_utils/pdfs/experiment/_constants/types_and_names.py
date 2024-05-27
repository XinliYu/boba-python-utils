from enum import Enum


class PreDefinedExperimentTypes(str, Enum):
    LiveTraffic = 'livetraffic'
    Opportunity = 'opportunity'
    Guardrail = 'guardrail'
    Train = 'run'


class PreDefinedExperimentNames(str, Enum):
    EN_GB_LOCALE_EXPANSION = 'en_gb_locale_expansion'
    EN_US_P1_L2 = 'en_us_p1_l2'
