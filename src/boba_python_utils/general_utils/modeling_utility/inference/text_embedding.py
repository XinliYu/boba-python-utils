import shutil
import uuid
from os import path
from typing import Callable, List, Union, Optional

from pyspark.sql import DataFrame, Column, SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType

import boba_python_utils.spark_utils as sparku
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.io_utils.json_io import write_json
from boba_python_utils.spark_utils.spark_functions.common import col_


def dump_text_embeddings(
        df_data: DataFrame,
        spark: SparkSession,
        encoder: Callable,
        cols_to_encode: List[Union[str, Column]],
        output_text_colname: str,
        output_embed_colname: str,
        model_path: str,
        model_workspace_path: str = None,
        embeddings_index_root_path: Optional[str] = None,
        local_embeddings_index_path: Optional[str] = None,
        partitions: int = 600,
        exists_path_method: Callable = path.exists,
        path_copy_method: Callable = shutil.copytree,
        **model_args
):
    """
    Dumps text embeddings for text columns in a Spark dataframe.

    Args:
        df_data: the dataframe containing the text columns to encode.
        spark: the spark session object.
        encoder: a callable (e.g. a type) that returns an encoder object with a method named
            `encode_texts`, and this `encode_texts` method can convert texts to their numeric
            representations; the object can be neural models compatible with Spark. This `encoder`
            callable will take `model_path`, `model_workspace_path` and `model_args` as its
            arguments to construct the encoder object.
        cols_to_encode: the text columns in `df_data` to encode.
        output_text_colname: the name of the column in the output dataframe to store all unique
            texts from the `cols_to_encode` of `df_data`.
        output_embed_colname: the name of the column in the output dataframe to store corresponding
            numeric embeddings of the texts in the text column.
        model_path: the path to the model files.
        model_workspace_path: the path to the model's workspace directory;
            usually this directory saves certain metadata like vocabulary, and
            intermediate results per epoch like checkpoints and intermediate evaluations;
            for example, AllenNLP's 'serialization' directory.
        embeddings_index_root_path: optionally provide the path to the root directory
            where all embeddings of the provided model are stored; there might be previously
            dumped embeddings in this directory and they will be reused; this can be a remote
            path like a s3 path; if this path is not provided, then embeddings will only be
            saved in `local_embeddings_dump_path`.
        local_embeddings_index_path: optionally provide the local path to the directory
            where the embeddings for the currently provided dataframe will be dumped;
            this must be a path recognized by the local computer.
        partitions: the number of partitions Spark can re-partition the dataframe `df_data` into
            before the computing the embeddings; each partition will be sent to one worker to
            compute the text embeddings.
        exists_path_method: the method to check if a possibly non-local path exists,
            e.g. the `embeddings_dump_root_path`; this method must be able to take
            `embeddings_dump_root_path` as its argument.
        path_copy_method: the method to copy dumped embeddings from `local_embeddings_dump_path` to
            the possibly non-local path `embeddings_dump_root_path`; the method must be able to
            take `local_embeddings_dump_path` and `embeddings_dump_root_path` as two arguments.
        model_args: provides named arguments for `encoder`.

    Returns: a dataframe with two columns, one column of name provided by `output_text_colname`
        storing the texts, and the other column of name provided by `output_embed_colname` storing
        the corresponding embeddings of the texts.

    """

    # region STEP1: argument sanitization


    if not local_embeddings_index_path:
        if not model_workspace_path:
            raise ValueError("one of 'model_workspace_path' or 'local_embeddings_dump_path' "
                             "must be provided so that we have a local path to store "
                             "the dumped embeddings")
        local_embeddings_index_path = path.join(model_workspace_path, str(uuid.uuid4()))

    embed_index_dirname = path.basename(local_embeddings_index_path)
    if not embed_index_dirname:
        embed_index_dirname = path.basename(path.dirname(local_embeddings_index_path))
    if not embed_index_dirname:
        raise ValueError(f"invalid 'local_embedding_index_path'; got {local_embeddings_index_path}")
    # endregion

    # region STEP2: compute texts to encode
    # will reuse existing embeddings if `embeddings_dump_root_path` is provided
    df_data_all_texts, count_df_data_all_texts = sparku.cache__(
        sparku.union(
            *(
                df_data.where(col_(x).isNotNull()).select(col_(x).alias(output_text_colname))
                for x in cols_to_encode
            )
        ).distinct(),
        name='df_data_all_texts_to_encode',
        return_count=True
    )

    embedding_index_schema = StructType(
        fields=[
            StructField(name=output_text_colname, dataType=StringType()),
            StructField(name=output_embed_colname, dataType=ArrayType(elementType=DoubleType()))
        ]
    )
    if embeddings_index_root_path:
        _embedding_index_root_path_exists = exists_path_method(embeddings_index_root_path)
        hprint_message(
            'embeddings_index_root_path', embeddings_index_root_path,
            'embedding_index_root_path_exists', _embedding_index_root_path_exists
        )
        if _embedding_index_root_path_exists:
            df_data_all_texts_new, count_df_data_all_texts_new = sparku.cache__(
                sparku.exclude_by_anti_join_on_columns(
                    df_data_all_texts,
                    sparku.solve_input(
                        path.join(embeddings_index_root_path, '*'),
                        schema=embedding_index_schema,
                        select=[output_text_colname],
                        spark=spark
                    ),
                    broadcast_join=True
                ),
                name='df_data_all_texts_to_encode_new',
                return_count=True
            )
        else:
            df_data_all_texts_new = df_data_all_texts
            count_df_data_all_texts_new = count_df_data_all_texts
    else:
        df_data_all_texts_new = df_data_all_texts
        count_df_data_all_texts_new = count_df_data_all_texts
    # endregion

    # region STEP3: compute embeddings
    if count_df_data_all_texts_new != 0:
        def compute_request_embeddings(partition, meta_data_compute=None):
            texts = []
            for row in partition:
                texts.append(row[output_text_colname])
            model = encoder(
                model_path,
                (
                    path.join(model_workspace_path, str(uuid.uuid4()))
                    if model_workspace_path
                    else None
                ),
                **model_args
            )

            if hasattr(model, 'encode_texts') and callable(model.encode_texts):
                encode_texts = model.encode_texts
            elif callable(model):
                encode_texts = model
            else:
                raise ValueError(
                    "the text encoding model must either has a method 'encode_texts' "
                    f"or be a callable itself; got {model}"
                )

            embeds = encode_texts(texts)
            return (
                {output_text_colname: text, output_embed_colname: embed.tolist()}
                for text, embed in zip(texts, embeds)
            )

        sparku.parallel_compute(
            df=df_data_all_texts_new,
            partition_transform_func=compute_request_embeddings,
            combine_partition_transform_func=None,
            repartition=partitions,
            file_based_combine=True,
            output_result_to_files=True,
            output_path=local_embeddings_index_path,
            output_overwrite=True,
            output_write_func=write_json,
            output_file_pattern='{}.json'
        )

        if embeddings_index_root_path:
            df_data_all_texts_new.unpersist()
            path_copy_method(
                local_embeddings_index_path,
                path.join(embeddings_index_root_path, embed_index_dirname)
            )

    _embedding_dump_path = embeddings_index_root_path or local_embeddings_index_path
    df_text_embeds, count_df_text_embeds = sparku.cache__(
        sparku.filter_by_inner_join_on_columns(
            sparku.solve_input(
                path.join(_embedding_dump_path, '*'),
                schema=embedding_index_schema,
                spark=spark
            ),
            df_data_all_texts,
            broadcast_join=True
        ),
        name='df_text_embeds',
        return_count=True
    )

    if count_df_text_embeds != count_df_data_all_texts:
        raise ValueError(f"missing {count_df_data_all_texts - count_df_text_embeds} embeddings")

    if count_df_text_embeds != df_text_embeds.select(output_text_colname).distinct().count():
        raise ValueError("duplicate embeddings found")

    df_data_all_texts.unpersist()
    # endregion

    return df_text_embeds

