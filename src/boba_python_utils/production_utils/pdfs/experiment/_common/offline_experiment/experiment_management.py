import uuid
from os import path

from slab_pdfs_offline.l1.common.constants import PDFS_DATASET_DATESTAMP_FORMAT
from boba_python_utils.general_utils.experiment_utility.simple_experiment import SimpleExperiment
from boba_python_utils.general_utils.modeling_utility.feature_building.common import read_feature_list
from boba_python_utils.string_utils.common import join_
from boba_python_utils.string_utils.datetime import reformat_datetime_str
from boba_python_utils.production_utils.pdfs.experiment._common.paths import get_pdfs_experiment_s3_root_path, get_pdfs_experiment_local_root_path
from boba_python_utils.production_utils.pdfs.experiment._constants.paths import (
    DIRNAME_RESOURCES,
    DIRNAME_MODELS,
    DIRNAME_CONFIGS,
    DIRNAME_L2_FEATURE_LIST,
)

RESOURCE_TYPE_EMBEDDINGS = 'embeddings'
RESOURCE_TYPE_BLOCKLIST = 'blocklist'

BLOCKLIST_NAME_ONLINE_LEARNING = 'online_learning'

METADATA_TYPE_L2_FEATURES = 'l2_features'
METADATA_TYPE_L2_FEATURES_NUMPY = 'l2_features_numpy'

RESULT_TYPE_L2 = 'l2'


def get_dataset_name(
        source_data_type: str,
        source_data_num_days: int,
        source_data_end_date: str
) -> str:
    return join_(
        source_data_type,
        f'{source_data_num_days}days',
        reformat_datetime_str(source_data_end_date, dst_format=PDFS_DATASET_DATESTAMP_FORMAT),
        sep='-'
    )


class PdfsExperiment(SimpleExperiment):
    def __init__(
            self,
            *arg_info_objs,
            expname,
            region,
            locale,
            bucket='',
            workspace='',
            local_codebase_root='',
            **kwargs
    ):
        super().__init__(
            *arg_info_objs,
            expname=expname,
            default_workspace_root='',
            default_local_workspace_root='',
            default_region=region,
            default_locale=locale,
            default_workspace=workspace,
            default_bucket=bucket,
            default_local_codebase_root=local_codebase_root,
            **kwargs
        )
        args = self.args
        region, locale, workspace = args.region, args.locale, args.workspace
        self.args.workspace_root = (
                self.args.workspace_root or
                get_pdfs_experiment_s3_root_path(
                    region=region,
                    locale=locale,
                    bucket=bucket,
                    workspace=workspace
                )
        )
        self.args.local_workspace_root = (
                self.args.local_workspace_root or
                get_pdfs_experiment_local_root_path(
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
            model_type,
            model_family=None,
            model_name=None,
            version=None
    ):
        return self.artifacts_path(
            artifact_type=path.join(DIRNAME_MODELS, model_type),
            artifact_family=model_family or getattr(self.args, 'model_family', None),
            artifact_name=model_name or getattr(self.args, 'model_name', None),
            version=version or getattr(self.args, 'model_version', None)
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
            model_type,
            model_family=None,
            model_name=None,
            version=None
    ):
        return self.local_artifacts_path(
            artifact_type=path.join(DIRNAME_MODELS, model_type),
            artifact_family=model_family or getattr(self.args, 'model_family', None),
            artifact_name=model_name or getattr(self.args, 'model_name', None),
            version=version or getattr(self.args, 'model_version', None)
        )

    def configs_dir_path(self):
        return path.join(self.local_codebase_root, DIRNAME_CONFIGS, self.name)

    def configs_dir_path_in_artifacts(self):
        return path.join(self.local_artifact_dir_path(), DIRNAME_CONFIGS)


class PdfsL1Experiment(PdfsExperiment):
    def __init__(
            self,
            *arg_info_objs,
            **kwargs
    ):
        super().__init__(
            *arg_info_objs,
            **kwargs
        )
        args = self.args
        if not args.dataset:
            args.dataset = get_dataset_name(
                source_data_type=args.source_data_type,
                source_data_num_days=args.source_data_num_days,
                source_data_end_date=args.source_data_end_date
            )

    def l1_model_fullname(self):
        return join_(
            getattr(self.args, 'model_type', None),
            getattr(self.args, 'model_family', None),
            getattr(self.args, 'model_name', None),
            getattr(self.args, 'model_version', None),
            sep='-'
        )

    def output_path_l1_embeddings_dump(self):
        return self.output_path_to_results(
            target=path.join(RESOURCE_TYPE_EMBEDDINGS, self.l1_model_fullname())
        )

    def embeddings_index_path_root(self):
        return self.resources_dir_path(
            resource_type=RESOURCE_TYPE_EMBEDDINGS,
            resource_family=self.args.model_family,
            resource_name=self.args.model_name,
            version=self.args.model_version
        )

    def generate_local_embeddings_index_path(self):
        embed_dump_id = str(uuid.uuid4())
        return path.join(self.local_resources_dir_path(
            resource_type=RESOURCE_TYPE_EMBEDDINGS,
            resource_family=self.args.model_family,
            resource_name=self.args.model_name,
            version=self.args.model_version
        ), embed_dump_id)


class PdfsL2Experiment(PdfsExperiment):
    def __init__(
            self,
            *arg_info_objs,
            expname,
            region,
            locale,
            default_l2_feature_list_name='l2_feats.txt',
            default_l2_feature_version='dev',
            default_local_codebase_root='',
            s3_resources_path=None,
            **kwargs
    ):
        super().__init__(
            *arg_info_objs,
            expname=expname,
            region=region,
            locale=locale,
            bucket=s3_resources_path,
            default_l2_feature_list_name=default_l2_feature_list_name,
            default_l2_feature_version=default_l2_feature_version,
            local_codebase_root=default_local_codebase_root,
            **kwargs
        )

    def l2_feat_list_dir_path(self):
        return path.join(self.configs_dir_path(), DIRNAME_L2_FEATURE_LIST)

    # def l2_feat_list_dir_path_in_artifacts(self):
    #     return path.join(self.configs_dir_path_in_artifacts(), c.DIRNAME_L2_FEATURE_LIST)

    def get_l2_feature_list(
            self,
            l2_feature_list_name=None,
            l2_feature_version=None,
            copy_to_artifacts=False,
            check_feature_index=False
    ):
        l2_feature_list_name = l2_feature_list_name or self.args.l2_feature_list_name
        l2_feature_version = l2_feature_version or self.args.l2_feature_version
        feature_list_path = path.join(self.l2_feat_list_dir_path(), l2_feature_version, l2_feature_list_name)
        # if copy_to_artifacts:
        #     feature_list_path_in_artifact_dir = path.join(self.l2_feat_list_dir_path_in_artifacts(), l2_feature_version, l2_feature_list_name)
        #     ensure_parent_dir_existence(feature_list_path_in_artifact_dir)
        #     if path.exists(feature_list_path_in_artifact_dir):
        #         os.remove(feature_list_path_in_artifact_dir)
        #     shutil.copyfile(feature_list_path, feature_list_path_in_artifact_dir)
        return read_feature_list(feature_list_path, check_feature_index=check_feature_index)

    # region model and results
    @staticmethod
    def _result_name(model_type, model_name, feature_name):
        return f'{model_type}-{model_name}.bin' if feature_name is None else f'{model_type}-{model_name}-{feature_name}'

    # endregion

    def local_output_path_l2_features_numpy(self, feature_name):
        return self.local_output_path_to_results(
            target=path.join(
                METADATA_TYPE_L2_FEATURES_NUMPY,
                self.args.l2_feature_version,
                feature_name
            )
        )

    def local_input_path_l2_features_numpy(self, feature_name):
        return self.local_input_path_from_results(
            target=path.join(
                METADATA_TYPE_L2_FEATURES_NUMPY,
                self.args.l2_feature_version,
                feature_name
            )
        )

    def output_path_l2_features_numpy(self, feature_name):
        return self.output_path_to_results(
            target=path.join(
                METADATA_TYPE_L2_FEATURES_NUMPY,
                self.args.l2_feature_version,
                feature_name
            )
        )

    def input_path_l2_features_numpy(self, feature_name):
        return self.input_path_from_results(
            target=path.join(
                METADATA_TYPE_L2_FEATURES_NUMPY,
                self.args.l2_feature_version,
                feature_name
            )
        )

    def output_path_l2_features(self, featurizer_name):
        return self.input_path_from_results(
            target=path.join(
                METADATA_TYPE_L2_FEATURES,
                self.args.l2_feature_version,
                featurizer_name
            )
        )

    def local_output_path_l2_results(self, feature_name):
        return self.local_output_path_to_results(
            target=path.join(RESULT_TYPE_L2, self.args.l2_feature_version, feature_name)
        )

    def local_input_path_l2_results(self, feature_name):
        return self.local_input_path_from_results(
            target=path.join(RESULT_TYPE_L2, self.args.l2_feature_version, feature_name)
        )

    def output_path_l2_results(self, feature_name):
        return self.output_path_to_results(
            target=path.join(RESULT_TYPE_L2, self.args.l2_feature_version, feature_name)
        )

    def input_path_l2_results(self, feature_name):
        return self.input_path_from_results(
            target=path.join(RESULT_TYPE_L2, self.args.l2_feature_version, feature_name)
        )

    def get_l2_full_feature_list(self, copy_to_artifacts=True, check_feature_index=False):
        return self.get_l2_feature_list(c.FILENAME_L2_FULL_FEATURE_LIST, copy_to_artifacts=copy_to_artifacts, check_feature_index=check_feature_index)

    def l2_model_fullname(self):
        return join_(
            getattr(self.args, 'model_type', None),
            getattr(self.args, 'model_family', None),
            getattr(self.args, 'model_name', None),
            getattr(self.args, 'model_version', None),
            sep='-'
        )
