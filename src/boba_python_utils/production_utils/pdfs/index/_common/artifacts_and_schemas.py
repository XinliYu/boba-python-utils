from os import path
from typing import Union

from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
from pyspark.sql.types import StructType

from abi_python_commons.schemas.datamart_schemas import alexa_brain_data_schema
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.spark_utils.schema import insert_schema, get_child_schema_by_path, get_field_schema_from_struct
from boba_python_utils.spark_utils.schema import save_schema
from boba_python_utils.string_utils.prefix_suffix import add_suffix
from boba_python_utils.production_utils.alexabrain_data.constants import AB_KEY_SIMPLIFIED_REQUEST, KEY_SITUATED_CONTEXT
from boba_python_utils.production_utils.common._constants.path import PreDefinedWorkspaces, SupportedLocales, SupportedRegions, PreDefinedBuckets, SupportedOwners
from boba_python_utils.production_utils.pdfs.index._common.data_paths import get_slab_aggregation_artifact_path, get_slab_aggregation_schema_artifact_path
from boba_python_utils.production_utils.pdfs.index._common.feature_flag import FeatureFlag
from boba_python_utils.production_utils.pdfs.index._constants.paths import PRODUCT_NAME_PDFS
from boba_python_utils.production_utils.s3.download import load_pickle_file
from boba_python_utils.production_utils.s3.upload import put_pickle_file


def pickle_save_object_as_artifact(
        obj,
        artifact_type: str,
        artifact_name: str,
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        workspace: Union[str, PreDefinedWorkspaces],
        feature_flag: Union[FeatureFlag, str],
        owner: Union[str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces
):
    output_path = get_slab_aggregation_artifact_path(
        artifact_type=artifact_type,
        artifact_name=artifact_name,
        region=region,
        locale=locale,
        workspace=workspace,
        feature_flag=feature_flag,
        product_name=PRODUCT_NAME_PDFS,
        owner=owner,
        bucket=bucket
    )

    put_pickle_file(
        obj=obj,
        dst_path=add_suffix(output_path, 'pkl', sep='.')
    )


def pickle_load_artifact_object(
        artifact_type: str,
        artifact_name: str,
        file_name: str,
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        workspace: Union[str, PreDefinedWorkspaces],
        feature_flag: Union[FeatureFlag, str],
        owner: Union[str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces
):
    output_path = get_slab_aggregation_artifact_path(
        artifact_type=artifact_type,
        artifact_name=artifact_name,
        region=region,
        locale=locale,
        workspace=workspace,
        feature_flag=feature_flag,
        product_name=PRODUCT_NAME_PDFS,
        owner=owner,
        bucket=bucket
    )

    hprint_message(
        'artifact_type', artifact_type,
        'artifact_name', artifact_name,
        'output_path', output_path,
        'file_name', file_name,
        title='pickle_load_artifact_object'
    )

    return load_pickle_file(
        add_suffix(path.join(output_path, file_name), 'pkl', sep='.', avoid_repeat=True)
    )


def pickle_load_schema(
        schema_name: str,
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        workspace: Union[str, PreDefinedWorkspaces],
        feature_flag: Union[FeatureFlag, str],
        owner: Union[str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces,
        file_name: str = 'schema.pkl'
):
    schema_path = get_slab_aggregation_schema_artifact_path(
        artifact_name=schema_name,
        region=region,
        locale=locale,
        workspace=workspace,
        feature_flag=feature_flag,
        product_name=PRODUCT_NAME_PDFS,
        owner=owner,
        bucket=bucket
    )

    hprint_message(
        'schema_name', schema_name,
        'schema_path', schema_path,
        'file_name', file_name,
        title='pickle_load_schema'
    )

    return load_pickle_file(
        add_suffix(path.join(schema_path, file_name), 'pkl', sep='.', avoid_repeat=True)
    )


def pickle_save_schema_as_artifact(
        dataframe_or_schema: Union[DataFrame, StructType],
        schema_name: str,
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        workspace: Union[str, PreDefinedWorkspaces],
        feature_flag: Union[FeatureFlag, str],
        owner: Union[str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces,
        file_name: str = 'schema.pkl'
):
    schema_path = get_slab_aggregation_schema_artifact_path(
        artifact_name=schema_name,
        region=region,
        locale=locale,
        workspace=workspace,
        feature_flag=feature_flag,
        product_name=PRODUCT_NAME_PDFS,
        owner=owner,
        bucket=bucket
    )

    hprint_message(
        'schema_name', schema_name,
        'schema_path', schema_path,
        'file_name', file_name,
        title='pickle_save_schema_as_artifact'
    )

    save_schema(
        dataframe_or_schema,
        output_path=add_suffix(
            path.join(schema_path, file_name), 'pkl', sep='.', avoid_repeat=True
        ),
        write_method=put_pickle_file
    )


ARTIFACT_NAME_SITUATED_CONTEXT = 'situated_context'


def get_alexa_brain_data_schema_with_situated_context(
        region: Union[str, SupportedRegions],
        locale: Union[str, SupportedLocales],
        workspace: Union[str, PreDefinedWorkspaces],
        feature_flag: Union[FeatureFlag, str] = None,
        owner: Union[str, SupportedOwners] = SupportedOwners.IN_HOUSE,
        bucket: Union[str, PreDefinedBuckets] = PreDefinedBuckets.AbmlWorkspaces,
        file_name: str = 'schema.pkl'
):
    situated_context_schema = StringType()
    has_parsed_situated_context_schema = False

    try:
        _situated_context_schema = pickle_load_schema(
            schema_name=ARTIFACT_NAME_SITUATED_CONTEXT,
            region=region,
            locale=locale,
            workspace=workspace,
            feature_flag=feature_flag,
            owner=owner,
            bucket=bucket,
            file_name=file_name
        )
        if _situated_context_schema is not None:
            situated_context_schema = _situated_context_schema
            has_parsed_situated_context_schema = True
    except:
        pass

    alexa_brain_data_schema_with_situated_context = insert_schema(
        schema_path=AB_KEY_SIMPLIFIED_REQUEST,
        schema=alexa_brain_data_schema,
        insertion_field_name_and_schema={
            KEY_SITUATED_CONTEXT: situated_context_schema
        },
        overwrite=True
    )

    hprint_message(
        'situated_context_schema', situated_context_schema,
        'inserted situated_context_schema', get_field_schema_from_struct(get_child_schema_by_path(
            schema_path=AB_KEY_SIMPLIFIED_REQUEST,
            schema=alexa_brain_data_schema_with_situated_context,
            return_element_schema=True
        ), field_name=KEY_SITUATED_CONTEXT)
    )

    return (
        alexa_brain_data_schema_with_situated_context,
        has_parsed_situated_context_schema
    )
