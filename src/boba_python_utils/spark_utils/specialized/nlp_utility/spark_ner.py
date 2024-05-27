import re
import uuid
from enum import Enum
from functools import partial
from os import path
from typing import Union, Callable, Mapping, List, Optional

from attr import attrs, attrib
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import ArrayType, StringType

from boba_python_utils.general_utils import spark_utils as sparku
from boba_python_utils.general_utils.common_objects.debuggable import Debuggable
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.general_utils.nlp_utility.common import Languages, PreDefinedNlpToolNames
from boba_python_utils.general_utils.nlp_utility.part_of_speech.pos_based_entity_extraction import ner_by_pos_pattern_spark_batch
from boba_python_utils.string_utils.regex import RegexFactory, regex_extract_all_groups
from boba_python_utils.time_utils.tictoc import tic, toc


class PreDefinedErMethods(str, Enum):
    PreDefinedPatterns = 'patterns'
    NLTK_POS = 'nltk_pos'
    FLAIR_POS = 'flair_pos'


@attrs(slots=False)
class SparkER(Debuggable):
    _spark = attrib(type=SparkSession)
    _er_method = attrib(
        type=Union[PreDefinedErMethods, Callable],
        default=PreDefinedErMethods.NLTK_POS
    )
    _er_method_fast = attrib(type=bool, default=False)
    _er_text_colname = attrib(type=str, default='text')
    _er_result_colname = attrib(type=str, default='entities')
    _er_enable_text_sanitization = attrib(type=bool, default=False)
    _er_args = attrib(type=Mapping, default=None)
    _er_name = attrib(type=str, default=None)
    _er_local_tmp_path = attrib(type=str, default=None)
    _er_parallel = attrib(type=int, default=None)
    _er_language = attrib(type=Languages, default=Languages.English)
    _er_cache = attrib(type=Union[str, List[str], DataFrame], default=None)
    _er_cache_num_files = attrib(type=int, default=1000)

    def __attrs_post_init__(self):
        if self._er_cache == 'default':
            self._er_cache = self._get_default_er_cache_path()

        if self._er_local_tmp_path:
            self._er_local_tmp_path = path.join(
                path.join(self._er_local_tmp_path, self._er_name),
                str(uuid.uuid4())
            )

        if self._spark is not None and not self._er_parallel:
            self._er_parallel = sparku.num_shuffle_partitions(self._spark)

        if isinstance(self._er_method, str) and not self._er_name:
            self._er_name = f'{self._er_method}'

        if not self._er_name:
            raise ValueError(f"'ner_name' must be specified")

        if isinstance(self._er_method, str):
            if self._er_method == PreDefinedErMethods.NLTK_POS:
                self._er_method = partial(
                    ner_by_pos_pattern_spark_batch,
                    text_field_name=self._er_text_colname,
                    ner_result_field_name=self._er_result_colname,
                    output_path=self._er_local_tmp_path,
                    repartition=self._er_parallel,
                    language=self._er_language,
                    tool=PreDefinedNlpToolNames.NLTK,
                    **(self._er_args or {})
                )
            elif self._er_method == PreDefinedErMethods.FLAIR_POS:
                self._er_method = partial(
                    ner_by_pos_pattern_spark_batch,
                    text_field_name=self._er_text_colname,
                    ner_result_field_name=self._er_result_colname,
                    output_path=self._er_local_tmp_path,
                    repartition=self._er_parallel,
                    language=self._er_language,
                    tool=PreDefinedNlpToolNames.FLAIR,
                    fast=self._er_method_fast,
                    **(self._er_args or {})
                )
            elif self._er_method == PreDefinedErMethods.PreDefinedPatterns:
                if not self._er_args:
                    raise ValueError("regex patterns must be provided for entity extraction")
                if isinstance(self._er_args, (list, tuple)):
                    self._er_args = RegexFactory(
                        patterns=self._er_args
                    )
                elif not isinstance(self._er_args, (str, RegexFactory)):
                    raise ValueError(f"not supported pattern object {self._er_args}")

                _er_pattern = str(self._er_args)
                hprint_message('er_pattern', _er_pattern)
                self._er_args = (self._er_args, re.compile(_er_pattern))
                self._er_method = self._ner_by_predefined_patterns
            else:
                raise ValueError(f"'{self._er_method}' is not a supported ner method")

    def _ner_by_predefined_patterns(self, df_text: DataFrame):
        return df_text.withColumn(
            self._er_result_colname,
            F.udf(
                partial(regex_extract_all_groups, pattern=self._er_args[1]),
                returnType=ArrayType(StringType())
            )(self._er_text_colname)
        )

    def _get_default_er_cache_path(self) -> Optional[str]:
        return None

    def _er_text_sanitization(self, text):
        return text

    def get_entities(self, df_text):
        KEY_TEXT = self._er_text_colname

        if self._er_enable_text_sanitization:
            KEY_TEXT_SANITIZED = f'{KEY_TEXT}_sanitized'

            df_text = sparku.cache__(
                df_text.withColumn(
                    KEY_TEXT_SANITIZED,
                    self._er_text_sanitization(KEY_TEXT)
                ),
                name='df_text (with sanitization)',
                unpersist=df_text
            )

            if self._debug_mode:
                has_sanitization_cond = (F.col(KEY_TEXT) != F.col(KEY_TEXT_SANITIZED)).alias(
                    'has_sanitization'
                )
                sparku.show_counts(df_text, has_sanitization_cond)
                df_text.where(has_sanitization_cond).show(20, False)

            df_er = sparku.cache__(
                df_text.select(KEY_TEXT_SANITIZED).withColumnRenamed(
                    KEY_TEXT_SANITIZED, KEY_TEXT
                ).distinct(),
                name='df_er'
            )
        else:
            df_er = sparku.cache__(
                df_text.select(KEY_TEXT).distinct(),
                name='df_er'
            )

        tic("running NER")
        df_er = sparku.cache__(
            self._er_method(df_er),
            spark=self._spark,
            name='df_er (with results)',
            unpersist=df_er
        )
        toc()

        if self._er_enable_text_sanitization:
            df_er = sparku.cache__(
                sparku.join_on_columns(
                    df_text,
                    df_er,
                    [KEY_TEXT_SANITIZED],
                    [KEY_TEXT]
                ).drop(
                    KEY_TEXT_SANITIZED
                ),
                unpersist=(
                    df_text,
                    df_er
                ),
                name='df_er (original request with results)'
            )
        else:
            df_er = sparku.cache__(
                sparku.join_on_columns(
                    df_text,
                    df_er,
                    [KEY_TEXT]
                ),
                unpersist=(
                    df_text,
                    df_er
                ),
                name='df_er (original request with results)'
            )

        return df_er
