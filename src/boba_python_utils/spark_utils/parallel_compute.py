import shutil
import tempfile
from functools import partial
from multiprocessing.pool import ThreadPool
from os import path
import uuid
from typing import Union, List, Any, Optional, Callable

from pyspark.sql import DataFrame, SparkSession
from tqdm import tqdm
from pathlib import Path

from boba_python_utils.common_utils.iter_helper import iter_
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.general_utils.multiprocess_utility import parallel_process_by_pool
from boba_python_utils.io_utils.pickle_io import pickle_save, pickle_load
from boba_python_utils.path_utils.common import ensure_dir_existence
from boba_python_utils.spark_utils.aggregation import union
from boba_python_utils.spark_utils.data_loading import solve_input, create_one_column_dataframe_from_iterable


def _parallel_compute_partition_transform(
        partition_index,
        partition,
        partition_transform_func,
        output_result_to_files,
        output_write_func,
        output_path,
        file_based_combine,
        output_file_pattern
):
    result = partition_transform_func(partition)
    if file_based_combine or output_result_to_files:
        # occasionally, there could be error
        # Spark does not have write permission for a remote folder (e.g. efs-storage),
        # but another tool like shutil has the permission;
        # in this case we first write the results to a temporary folder where Spark has permission,
        # and then copy the result to the actual output path
        output_path = path.join(output_path, output_file_pattern.format(partition_index))
        output_write_func(result, output_path)
        yield output_path if file_based_combine else result
    else:
        yield result


def _combine_lists(result, combined_result):
    if combined_result is None:
        combined_result = result
    else:
        for x, y in zip(result, combined_result):
            y.extend(x)
    return combined_result


def _combine_list(result, combined_result):
    if combined_result is None:
        combined_result = result
    else:
        combined_result.extend(result)
    return combined_result


def _partition_transform_func_mt_wrap(partition, partition_transform_func: Callable, num_p: int):
    return parallel_process_by_pool(
        num_p=num_p,
        data_iter=partition,
        target=partition_transform_func,
        merge_output=True,
        vertical_merge=False,
        pool_object=ThreadPool
    )


def parallel_compute(
        df: Union[DataFrame, List[DataFrame], List[Any]],
        partition_transform_func,
        combine_partition_transform_func=None,
        output_result_to_files=False,
        output_write_func=None,
        output_path=None,
        output_overwrite=True,
        output_file_pattern='{}.bin',
        file_based_combine=False,
        output_read_func=None,
        repartition=None,
        output_tmp_path: Optional[str] = None,
        multi_threading: Optional[int] = None,
        spark: Optional[SparkSession] = None
):
    """
    Runs parallel computation over a dataframe

    File-Based Combine
    ------------------
    First output results to files, and then reload the results from files
    and combine them as a single dataframe.

    """

    # region STEP1: preparation
    if output_path and path.exists(output_path) and not output_overwrite:
        raise ValueError(f"path '{output_path}' already exists; "
                         f"remove the existing path or set 'output_overwrite' as True")
    if isinstance(df, (list, tuple)):
        if all(isinstance(_df, DataFrame) for _df in df):
            df = union(*df)
        if all((not isinstance(_df, DataFrame)) for _df in df):
            df = create_one_column_dataframe_from_iterable(df, spark=spark)
        else:
            raise ValueError(f"input data is a mix of dataframe and non-dataframe objects; "
                             f"got {df}")
    repartition_args = repartition
    if repartition_args:
        from boba_python_utils.spark_utils.common import repartition
        if repartition_args is True:
            df = repartition(df, spark=spark)
        else:
            df = repartition(df, *iter_(repartition_args), spark=spark)

    if multi_threading is not None:
        partition_transform_func = partial(
            _partition_transform_func_mt_wrap,
            partition_transform_func=partition_transform_func,
            num_p=multi_threading
        )

    # endregion

    # region STEP2: solving output paths
    if file_based_combine or output_result_to_files:
        # either we want to enable file-based result combine,
        # or we simply want to output the results of each partition to files
        output_write_func = output_write_func or pickle_save
        if not output_path:
            if output_result_to_files:
                raise ValueError("'output_result_to_files' is set True, "
                                 "but 'output_path' is not specified")
            else:
                output_path = tempfile.mkdtemp()
                if output_tmp_path is not None and path.exists(output_path):
                    shutil.rmtree(output_path)
        else:
            if output_tmp_path is None:
                ensure_dir_existence(output_path, clear_dir=True)
            else:
                if path.exists(output_path):
                    shutil.rmtree(output_path)

    if output_tmp_path is not None:
        if output_tmp_path is True:
            output_tmp_path = str(Path.home())
        output_tmp_path = path.join(
            output_tmp_path,
            '_tmp',
            'spark_parallel_compute',
            str(uuid.uuid4())
        )
        hprint_message('use tmp path', output_tmp_path)
        ensure_dir_existence(output_tmp_path, clear_dir=True)
    # endregion

    # region STEP3: parallel compute
    results = df.rdd.mapPartitionsWithIndex(partial(
        _parallel_compute_partition_transform,
        partition_transform_func=partition_transform_func,
        output_result_to_files=output_result_to_files,
        output_write_func=output_write_func,
        output_path=output_tmp_path or output_path,
        file_based_combine=file_based_combine,
        output_file_pattern=output_file_pattern
    ))

    if output_tmp_path is not None:
        hprint_message('move dumped data to', output_path)
        shutil.move(output_tmp_path, output_path)
    # endregion

    if combine_partition_transform_func:
        if combine_partition_transform_func == 'dataframe':
            return spark.createDataFrame(results)

        results = results.collect()
        if combine_partition_transform_func == 'list':
            combine_partition_transform_func = _combine_list
        elif combine_partition_transform_func == 'lists':
            combine_partition_transform_func = _combine_lists

        if file_based_combine:
            if isinstance(combine_partition_transform_func, str) and combine_partition_transform_func.startswith('reload_'):
                return solve_input(results, spark=spark, input_format=combine_partition_transform_func[7:])

            output_read_func = output_read_func or pickle_load
            combined_result = None
            for result_path in tqdm(results, desc=f'combining results from files at {output_path}'):
                result = output_read_func(result_path)
                combined_result = combine_partition_transform_func(result, combined_result)
        else:
            combined_result = None
            for result in tqdm(results, desc='combining results'):
                combined_result = combine_partition_transform_func(result, combined_result)

        if (not output_result_to_files) and file_based_combine and path.exists(output_path):
            try:
                shutil.rmtree(output_path)
            except:
                pass
        return combined_result
    else:
        return results.collect()
