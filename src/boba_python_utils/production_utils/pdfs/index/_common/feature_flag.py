from enum import Enum

from boba_python_utils.general_utils.general import make_list_
from boba_python_utils.production_utils.greenwich_data._constants.paths import GwVersion
from boba_python_utils.production_utils.greenwich_data.common import (
    CPDR7_1_CONFIG,
    CPDR_UNIFIED_CONFIG,
    CPDR7_CONFIG,
    CPDR6_CONFIG,
    CpdrConfig,
)


class SupportedCPDRVersions(str, Enum):
    # ! DEPRECATED
    # Only Unified Greenwich will be supported

    V3 = '3'
    V6 = '6'
    V7 = '7',
    V7_1 = '7_1'
    UNIFIED = 'unified'


class FeatureFlag:
    # ! DEPRECATED
    def __init__(
            self,
            cpdr_version: str = SupportedCPDRVersions.UNIFIED,
            multi_source_index: bool = True,
            add_signals: bool = True,
            secondary_version: str = ''
    ):
        self.cpdr_version = cpdr_version
        self.multi_source_index = multi_source_index
        self.add_signals = add_signals
        self.cpdr_config = get_cpdr_config_by_version(cpdr_version)
        self.secondary_version = secondary_version

    def no_secondary_version(self):
        return FeatureFlag(
            cpdr_version=self.cpdr_version,
            multi_source_index=self.multi_source_index,
            add_signals=self.add_signals
        )

    def replace_cpdr_version(self, new_cpdr_version):
        return FeatureFlag(
            cpdr_version=new_cpdr_version or self.cpdr_version,
            multi_source_index=self.multi_source_index,
            add_signals=self.add_signals,
            secondary_version=self.secondary_version
        )


def get_cpdr_config_by_version(cpdr_version) -> CpdrConfig:
    if cpdr_version == SupportedCPDRVersions.UNIFIED:
        return CPDR_UNIFIED_CONFIG
    if cpdr_version == SupportedCPDRVersions.V7:
        return CPDR7_CONFIG
    if cpdr_version == SupportedCPDRVersions.V7_1:
        return CPDR7_1_CONFIG
    elif cpdr_version == SupportedCPDRVersions.V6:
        return CPDR6_CONFIG
    else:
        raise ValueError(f"the CPDR version '{cpdr_version}' is not supported")


def get_greenwich_version_by_cpdr_version(cpdr_version) -> GwVersion:
    if cpdr_version == SupportedCPDRVersions.UNIFIED:
        return GwVersion.GREENWICH_UNIFIED
    else:
        return GwVersion.GREENWICH3
