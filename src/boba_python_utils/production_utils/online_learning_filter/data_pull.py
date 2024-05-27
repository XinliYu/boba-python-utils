from datetime import datetime, timedelta
from typing import Union

from pyspark.sql import SparkSession, DataFrame

from boba_python_utils.common_utils.map_helper import get_by_key_path
from boba_python_utils.path_utility.path_string_operations import get_shortest_prefix
from boba_python_utils.general_utils.pathex import (
    get_path_with_year_month_day, )
from boba_python_utils.path_utils.path_with_date_time_info import next_available_path_with_year_month_day
from boba_python_utils.spark_utils.data_loading import cache__, solve_input
from boba_python_utils.spark_utils.common import CacheOptions
from boba_python_utils.spark_utils.data_writing import write_df
from boba_python_utils.spark_utils.join_and_filter import where
from boba_python_utils.production_utils.online_learning_filter.constants import *
from boba_python_utils.production_utils.pdfs.testsets.constants import KEY_QUERY, KEY_REWRITE
from boba_python_utils.production_utils.s3.existence import (
    execute_spark_aggregation_if_not_exist_spark_success, exists_path
)
import pyspark.sql.functions as F

DEFAULT_ONLINE_LEARNING_FILTER_DATE_DELTA = -2


def get_online_learning_filter_path(
        date: Union[str, datetime] = None,
        region: Union[str, SupportedRegions] = SupportedRegions.NA,
        locale: Union[str, SupportedLocales] = SupportedLocales.EN_US,
        use_first_available_date: bool = True,
        return_first_available_date: bool = False
):
    if not date:
        date = datetime.today() + timedelta(days=DEFAULT_ONLINE_LEARNING_FILTER_DATE_DELTA)

    online_learning_filter_path_pattern = get_by_key_path(
        ONLINE_LEARNING_FILTER_PATH_PATTERNS,
        [locale, region],
        raise_key_error=True,
        error_messages="locale or region {key} is not supported"
    )

    if use_first_available_date:
        _, date = next_available_path_with_year_month_day(
            path_pattern=get_shortest_prefix(online_learning_filter_path_pattern, '* {day}'),
            start_date=date,
            days_delta=-100,
            path_exists=exists_path,
            return_hit_date=return_first_available_date,
            verbose=True
        )

    _path = get_path_with_year_month_day(
        path_pattern=online_learning_filter_path_pattern,
        date=date,
        hour='*',
        locale=locale
    )
    if return_first_available_date:
        return _path, date
    else:
        return _path


def load_online_learning_filter(
        input_online_learning_filter: str = None,
        date: Union[str, datetime] = None,
        region: Union[str, SupportedRegions] = SupportedRegions.NA,
        locale: Union[str, SupportedLocales] = SupportedLocales.EN_US,
        output_path: str = None,
        spark: SparkSession = None,
        cache_option=CacheOptions.IMMEDIATE,
        provider_name=None,
        force: bool = False
):
    if not input_online_learning_filter:
        input_online_learning_filter, date = get_online_learning_filter_path(
            date=date,
            region=region,
            locale=locale,
            return_first_available_date=True
        )
        output_path = get_path_with_year_month_day(
            path_pattern=output_path,
            date=date
        )

    def spark_aggregation_method() -> DataFrame:
        df_online_learning_filter = cache__(
            input_online_learning_filter,
            name='df_online_learning_filter',
            cache_option=cache_option,
            spark=spark
        )

        latest_version = df_online_learning_filter.select(
            KEY_OLB_VERSION
        ).distinct().orderBy(
            KEY_OLB_VERSION
        ).collect()[-1][0]

        select_cond = {KEY_OLB_VERSION: latest_version}
        if KEY_OLB_PROVIDER_NAME in df_online_learning_filter.columns:
            all_supported_providers = set(
                map(
                    lambda x: x[0],
                    df_online_learning_filter.select(
                        KEY_OLB_PROVIDER_NAME
                    ).distinct().collect()
                )
            )

            select_cond[KEY_OLB_PROVIDER_NAME] = (
                provider_name
                if provider_name is not None and provider_name in all_supported_providers
                else 'Others'
            )

        if latest_version > '4.0':
            select_cond[KEY_OLB_IS_FINALLY_BLOCKED] = True
            select_cond[KEY_OLB_LOCALE] = locale

        if latest_version <= '4.0':
            key_rewrite = KEY_OLB_REWRITE_LEGACY
        else:
            key_rewrite = KEY_OLB_REWRITE

        df_online_learning_filter = cache__(
            where(
                df_online_learning_filter,
                select_cond
            ).select(
                F.col(KEY_OLB_QUERY).alias(KEY_QUERY), F.col(key_rewrite).alias(KEY_REWRITE)
            ).distinct(),
            name='df_online_learning_filter (distinct)',
            cache_option=cache_option,
            unittest=df_online_learning_filter
        )

        return df_online_learning_filter

    def spark_read_method(s3path: str) -> DataFrame:
        return solve_input(s3path, spark=spark)

    def spark_write_method(df, s3path):
        write_df(
            df,
            output_path=s3path,
            num_files=1200,
            format='json',
            repartition=True,
            compress=True,
            show_counts=False,
            unpersist=False
        )

    return execute_spark_aggregation_if_not_exist_spark_success(
        s3path=output_path,
        spark_aggregation_method=spark_aggregation_method,
        spark_read_method=spark_read_method,
        spark_write_method=spark_write_method,
        force=force,
        save_aggregation=True,
        return_dataframe=True,
    )

# date = '09/27/2022'
# region = 'na'
# locale = 'pt_BR'
# output_path = 's3://abml-workspaces-na/zgchen/tmp/test'
# input_online_learning_filter = get_online_learning_filter_path(date, region, locale)
# cache_option = CacheOptions.IMMEDIATE
# force: bool = False
# show_counts(
#     where(
#         df_online_learning_filter,
#         select_cond
#     ), ['version', 'simplifiedRewriteProviderName', 'isFinallyBlocked', 'locale']
# )
