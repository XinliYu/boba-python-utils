import uuid
from argparse import ArgumentParser
from typing import Union, Callable, Iterable

from boba_python_utils.general_utils.arg_utility.arg_parse import ARG_INFO, ArgInfo
from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.general_utils.experiment_utility.experiment_base import ExperimentBase
from os import path, environ

from boba_python_utils.string_utils.common import join_

DIRNAME_RESOURCES = 'resources'
DIRNAME_MODELS = 'models'
DIRNAME_CONFIGS = 'configs'
RESULT_TYPE_EVALUATION = 'evaluation'
RESULT_TYPE_MODEL = 'model'
RESULT_TYPE_MODEL_CACHE = 'model_cache'


class SimpleExperiment(ExperimentBase):
    def __init__(
            self, *arg_info_objs: ARG_INFO,
            artifact_dir_name='artifacts',
            source_data_dir_name='source_data',
            datasets_dir_name='datasets',
            metadata_dir_name='meta_data',
            analysis_dir_name='analysis',
            results_dir_name='results',
            default_arg_preset_root: str = None,
            arg_preset: Union[str, dict] = None,
            default_workspace_root='.',
            workspace_override_args=True,
            verbose=False,
            arg_parser: ArgumentParser = None,
            **kwargs
    ):
        super().__init__(
            *arg_info_objs,
            simple_experiment=True,
            artifact_dir_name=artifact_dir_name,
            dirs=(source_data_dir_name, datasets_dir_name, metadata_dir_name, results_dir_name, analysis_dir_name),
            no_version_dirs=(source_data_dir_name, metadata_dir_name),
            preset_root=default_arg_preset_root,
            preset=arg_preset,
            default_workspace_root=default_workspace_root,
            general_args=False,
            workspace_override_args=workspace_override_args,
            deep_learning_args=False,
            nlp_args=False,
            verbose=verbose,
            arg_parser=arg_parser,
            **kwargs
        )


class ModelingExperiment(SimpleExperiment):
    def __init__(
            self,
            *arg_info_objs: ARG_INFO,
            region: str = 'NA',
            locale: str = 'en_US',
            workspace='science',
            bucket='',
            local_codebase_root='',
            model_type: str = '',
            model_family: str = '',
            model_name: str = '',
            model_version: str = '',
            get_default_workspace_root_path: Callable = None,
            get_default_local_workspace_root_path: Callable = None,
            arg_parser: ArgumentParser = None,
            **kwargs
    ):
        super().__init__(
            *arg_info_objs,
            default_workspace_root='',
            default_local_workspace_root='',
            default_region=region,
            default_locale=locale,
            default_workspace=workspace,
            default_bucket=bucket,
            default_local_codebase_root=environ.get(local_codebase_root, local_codebase_root),
            default_model_type=model_type,
            default_model_family=model_family,
            default_model_name=model_name,
            default_model_version=model_version,
            arg_parser=arg_parser,
            **kwargs
        )
        args = self.args
        region, locale, workspace = args.region, args.locale, args.workspace
        if get_default_workspace_root_path is not None:
            self.args.workspace_root = (
                    self.args.workspace_root or
                    get_default_workspace_root_path(
                        region=region,
                        locale=locale,
                        bucket=bucket,
                        workspace=workspace
                    )
            )
        if get_default_local_workspace_root_path is not None:
            self.args.local_workspace_root = (
                    self.args.local_workspace_root or
                    get_default_local_workspace_root_path(
                        locale=locale,
                        workspace=workspace
                    )
            )

        self.local_codebase_root = args.local_codebase_root

    def resources_dir_path(
            self,
            resource_type,
            resource_family=None,
            resource_name=None,
            version=None
    ):
        return self.artifacts_path(
            artifact_type=path.join(DIRNAME_RESOURCES, resource_type),
            artifact_family=resource_family or getattr(self.args, 'resource_family', None),
            artifact_name=resource_name or getattr(self.args, 'resource_name', None),
            version=version or getattr(self.args, 'resource_version', None)
        )

    def model_dir_path(
            self,
            model_type: str = None,
            model_family: str = None,
            model_name: str = None,
            model_version: str = None
    ):
        return self.artifacts_path(
            artifact_type=path.join(
                DIRNAME_MODELS,
                (model_type or getattr(self.args, 'model_type', None))
            ),
            artifact_family=model_family or getattr(self.args, 'model_family', None),
            artifact_name=model_name or getattr(self.args, 'model_name', None),
            version=model_version or getattr(self.args, 'model_version', None)
        )

    def local_resources_dir_path(
            self,
            resource_type,
            resource_family=None,
            resource_name=None,
            version=None
    ):
        return self.local_artifacts_path(
            artifact_type=path.join(DIRNAME_RESOURCES, resource_type),
            artifact_family=resource_family or getattr(self.args, 'resource_family', None),
            artifact_name=resource_name or getattr(self.args, 'resource_name', None),
            version=version or getattr(self.args, 'resource_version', None)
        )

    def local_model_dir_path(
            self,
            model_type: str = None,
            model_family: str = None,
            model_name: str = None,
            model_version: str = None
    ):
        return self.local_artifacts_path(
            artifact_type=path.join(
                DIRNAME_MODELS,
                (model_type or getattr(self.args, 'model_type', None))
            ),
            artifact_family=model_family or getattr(self.args, 'model_family', None),
            artifact_name=model_name or getattr(self.args, 'model_name', None),
            version=model_version or getattr(self.args, 'model_version', None)
        )

    def local_model_results_dir_path(
            self,
            result_type: Union[str, Iterable[str]] = (
                    RESULT_TYPE_MODEL,
                    RESULT_TYPE_MODEL_CACHE,
                    RESULT_TYPE_EVALUATION
            ),
            model_type: str = None,
            model_family: str = None,
            model_name: str = None,
            model_version: str = None,
            result_name: str = None
    ):
        target = self.model_fullname(
            model_type=model_type,
            model_family=model_family,
            model_name=model_name,
            model_version=model_version
        )
        if result_name is None:
            result_name = str(uuid.uuid4())

        target = path.join(target, result_name)

        out = []
        for _result_type in iter__(result_type):
            _target = path.join(target, _result_type)
            out.append(self.local_output_path_to_results(target=_target))

        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)

    def get_result_name_from_result_dir_path(self, result_dir_path: str):
        return path.basename(path.dirname(result_dir_path))

    def configs_dir_path(self):
        return path.join(self.local_codebase_root, DIRNAME_CONFIGS, self.name)

    def configs_dir_path_in_artifacts(self):
        return path.join(self.local_artifact_dir_path(), DIRNAME_CONFIGS)

    def model_fullname(
            self,
            model_type: str = None,
            model_family: str = None,
            model_name: str = None,
            model_version: str = None
    ):
        return join_(
            model_type or getattr(self.args, 'model_type', None),
            model_family or getattr(self.args, 'model_family', None),
            model_name or getattr(self.args, 'model_name', None),
            model_version or getattr(self.args, 'model_version', None),
            sep='-'
        )
