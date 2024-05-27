import os
from argparse import ArgumentParser
from typing import Union

from boba_python_utils.general_utils.experiment_utility.simple_experiment import ModelingExperiment
from boba_python_utils.production_utils.common._constants.path import (
    SupportedLocales, SupportedRegions, PreDefinedWorkspaces, PreDefinedBuckets
)
from boba_python_utils.production_utils.slab.common.paths import (
    get_slab_experiment_s3_root_path, get_slab_experiment_local_root_path
)

ENV_NAME_SLAB_CODEBASE = 'SLAB_CODEBASE'


class SlabExperiment(ModelingExperiment):
    def __init__(
            self,
            *arg_info_objs,
            expname: str,
            region: Union[str, SupportedRegions] = SupportedRegions.NA,
            locale: Union[str, SupportedLocales] = SupportedLocales.EN_US,
            bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces,
            workspace: Union[str, PreDefinedWorkspaces] = PreDefinedWorkspaces.SCIENCE,
            local_codebase_root: str = ENV_NAME_SLAB_CODEBASE,
            model_type: str = '',
            model_family: str = '',
            model_name: str = '',
            model_version: str = '',
            arg_parser: ArgumentParser = None,
            **kwargs
    ):
        super().__init__(
            *arg_info_objs,
            expname=expname,
            region=region,
            locale=locale,
            workspace=workspace,
            bucket=bucket,
            local_codebase_root=local_codebase_root,
            model_type=model_type,
            model_family=model_family,
            model_name=model_name,
            model_version=model_version,
            get_default_workspace_root_path=get_slab_experiment_s3_root_path,
            get_default_local_workspace_root_path=get_slab_experiment_local_root_path,
            arg_parser=arg_parser,
            **kwargs
        )
