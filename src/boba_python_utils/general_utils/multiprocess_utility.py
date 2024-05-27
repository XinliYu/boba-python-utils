import multiprocessing
import uuid
import warnings
from collections import defaultdict, Counter
from itertools import islice
from multiprocessing import get_context, cpu_count
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
from os import path
from time import sleep
from typing import Tuple, Callable, List, Union, Iterator, Iterable, Any, Set

from attr import attrs, attrib
from tqdm import tqdm

from boba_python_utils.common_utils.iter_helper import split_iter
from boba_python_utils.common_utils.map_helper import (
    merge_counter_valued_mappings,
    merge_mappings,
    merge_list_valued_mappings,
    merge_set_valued_mappings,
    sum_dicts
)
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.io_utils.text_io import iter_all_lines_from_all_files
from boba_python_utils.io_utils.pickle_io import pickle_save, pickle_load
from boba_python_utils.path_utils.common import ensure_dir_existence
from boba_python_utils.path_utils.path_string_operations import append_timestamp
from boba_python_utils.time_utils.tictoc import tic, toc


# region basics

def get_suggested_num_workers(num_p=None):
    """
    Gets a suggested number of workers considering both the specified number by `num_p`,
    and the number of available CPUs. If `num_p` is `None` or 0 or a negative number,
    the number of CPUs minus 1 or 2 will be returned.
    Otherwise, the smaller number between `num_p` and the number of CPUs will be returned.

    Args:
        num_p: specify a desired number of workers.

    Returns: the suggested number of workers considering both the specified `num_p`
        and the number of available CPUs.

    """
    if num_p is None or num_p <= 0:
        num_workers = cpu_count()
        if num_workers <= 8:
            return num_workers - 1
        else:
            return num_workers - 2
    else:
        return min(num_p, cpu_count())


class MPResultTuple(tuple):
    # Use a dummy data type `MPResultTuple` (just a tuple) to indicate
    # a multi-processing output contains results from multiple data items.
    pass


@attrs(slots=True)
class MPTarget:
    """
    Wraps a multi-processing target callable and provides rich configuration
    for multi-processing operations.

    Attributes:
        target: the multi-processing target callable which processes the assigned data and returns
            the corresponding results.
        pass_pid: True to indicate the process id should be passed to the target function.
        result_dump_path: provides a path to dump the output of multi-processing target.
        result_dump_file_pattern: provides a file pattern for the dumped output files;
            uuid will be used as file names if this argument is not provided.
        result_dump_method: provides a customized function to save the output; by default
            we use pickle to serialize and save the output.
        name: a name for this multi-processing target.
        use_queue: True to indicate the input for this multi-processing target is a queue,
            and the output will also use a queue. The input queue consists of task assignments,
            and each process takes an assignment from the input queue, and puts the result
            in the output queue. The queue must implement its `get`, `put` and `empty` methods.
            The queue can be a streaming queue with indefinite number of assignments in the queue,
            and use a flag to indicate whether the processing should stop.
        queue_wait_time: sets a time length in seconds so that we wait this amount time before the
            processing the next data item from the queue when `use_queue` is set True.
        pass_each_data_item: True to pass each item from the input
            (e.g. one file, or one single data item) to the `target` function,
            rather than passing a list of items to the `target`.
            In multi-processing, the `target` function is expected to have logic for processing
            a list of assigned items; when this attribute is set True, we apply the build-in
            logic to help iterate through the assigned items and pass each item to `target`, so that
            `target` can be an existing function that already processes a single file or
            a single data item.
        data_item_iter: True or assign a customized data iterator to read data items from provided
            input (e.g. file paths), and pass the data items to the `target` function rather than
            passing the file paths. By default we assume the input are file paths of text files and
            each line of a text data file is treated as one data item. We can also assign a
            customized function to provide customized logic to iterate through data items.
        unpack_single_result: True to take the result out of the containing tuple
            if a process's assignment contains a single input; effective only if
            `data_from_files` is set False.
        remove_none_from_output: True to remove None from the output.
        return_output: True to always return the output of the target function; otherwise returns
            the output dumps when `result_dump_path` is set.
        common_func: a convenience parameter; True to indicate `pass_pid` is False,
            `pass_each_data_item` is True and `unpack_single_result` is True.
        is_target_iter: True to indicate the target function return an iterator.

    """
    _target = attrib(type=Callable)
    pass_pid = attrib(type=bool, default=True)
    result_dump_path = attrib(type=str, default=None)
    result_dump_file_pattern = attrib(type=str, default=None)
    result_dump_method = attrib(type=Callable, default=pickle_save)
    name = attrib(type=str, default=None)
    use_queue = attrib(type=bool, default=False)
    queue_wait_time = attrib(type=float, default=0.5)
    pass_each_data_item = attrib(type=bool, default=True)
    data_item_iter = attrib(type=Union[bool, Callable[[Any], Iterator]], default=False)
    unpack_single_result = attrib(type=bool, default=False)
    remove_none_from_output = attrib(type=bool, default=False)
    return_output = attrib(type=bool, default=False)
    common_func = attrib(type=bool, default=False)
    is_target_iter = attrib(type=bool, default=False)

    def __attrs_post_init__(self):
        if self.result_dump_path:
            ensure_dir_existence(self.result_dump_path)
        if self.common_func:
            self.pass_pid = False
            self.pass_each_data_item = True
            self.unpack_single_result = True

    def target(self, pid, data, *args):
        if self._target is not None:
            if self.pass_pid:
                rst = self._target(pid, data, *args)
            else:
                rst = self._target(data, *args)
            return list(rst) if self.is_target_iter else rst
        else:
            raise NotImplementedError

    def _get_name(self):
        return self.name or str(self._target)

    def _has_data_iter(self):
        return (
                self.data_item_iter is not None and
                self.data_item_iter is not False
        )

    def _process_input_data(self, data):
        return (
            iter_all_lines_from_all_files(input_paths=data, use_tqdm=True)
            if self.data_item_iter is True
            else tqdm(self.data_item_iter(data))
        )

    def _process_result_no_unpack(self, result):
        return MPResultTuple(
            (x for x in result if x is not None)
            if self.remove_none_from_output
            else result
        )

    def _process_result(self, result):
        return result[0] if (
                self.unpack_single_result and
                (
                        isinstance(result, (list, tuple)) or
                        (
                                hasattr(result, '__len__') and
                                hasattr(result, '__getitem__')
                        )
                ) and len(result) == 1
        ) else self._process_result_no_unpack(result)

    def __call__(self, pid, data, *args):
        hprint_message('initialized', f'{self.name}{pid}')
        no_job_cnt = 0
        if self.pass_each_data_item:
            if (
                    not self.result_dump_path
                    and self.use_queue
            ):
                iq: Queue = data
                oq: Queue = args[0]
                flags = args[1]
                while True:
                    #  keeps taking assignment from the queue until the flag is False
                    while not iq.empty():
                        data = iq.get()
                        if self._has_data_iter():
                            oq.put(
                                self._process_result_no_unpack((
                                    self.target(pid, dataitem, *args[2:])
                                    for dataitem in self._process_input_data(data)
                                ))
                            )
                        else:
                            oq.put(self._process_result(tuple(
                                self.target(pid, dataitem, *args[2:]) for dataitem in data
                            )))

                    if not flags or flags[0]:
                        return
                    no_job_cnt += 1
                    if no_job_cnt % 10 == 0:
                        hprint_message(
                            'job', f'{self._get_name()}: {pid}',
                            'wait for', self.queue_wait_time
                        )
                    sleep(self.queue_wait_time)
            else:
                if self._has_data_iter():
                    output = self._process_result_no_unpack((
                        self.target(pid, dataitem, *args)
                        for dataitem in self._process_input_data(data)
                    ))
                else:
                    data = tqdm(data, desc=f'{self._get_name()}: {pid}')
                    output = self._process_result(tuple(
                        (self.target(pid, dataitem, *args) for dataitem in data)
                    ))
        elif not self.result_dump_path and self.use_queue:
            #  keeps taking assignment from the queue until the flag is False
            iq: Queue = data
            oq: Queue = args[0]
            flags = args[1]
            while True:
                while not iq.empty():
                    data = iq.get()
                    if self._has_data_iter():
                        data = self._process_input_data(data)
                    result = self.target(pid, data, *args[2:])
                    oq.put(self._process_result(result))
                if not flags or flags[0]:
                    return
                no_job_cnt += 1
                if no_job_cnt % 10 == 0:
                    hprint_message(
                        'job', f'{self._get_name()}: {pid}',
                        'wait for', self.queue_wait_time
                    )
                sleep(self.queue_wait_time)
        else:
            if self._has_data_iter():
                data = self._process_input_data(data)
            output = self._process_result(self.target(pid, data, *args))

        if self.result_dump_path:
            dump_path = path.join(
                self.result_dump_path,
                (
                    f'{pid:05}-{append_timestamp(str(uuid.uuid4()))}.mpb'
                    if self.result_dump_file_pattern is None
                    else self.result_dump_file_pattern.format(pid))
            )

            hprint_message('dumping results to', dump_path)
            self.result_dump_method(output, dump_path)
            if self.return_output:
                return output
            else:
                del output
                return dump_path
        else:
            return output


# class MPWriteOutput:
#     def __init__(self, target, output_path):
#         self._target = target
#         self._output_path = output_path
#
#     def __call__(self, *args):
#         output = self._target(*args)


def _default_result_merge(results):
    if isinstance(results[0], list):
        if all((isinstance(result, list) for result in results[1:])):
            results = tqdm(results)
            results.set_description('merging lists')
            return sum(results, [])
    elif isinstance(results[0], tuple):
        if all((isinstance(result, tuple) for result in results[1:])):
            results = tqdm(results)
            results.set_description('merging tuples')
            return sum(results, ())
    elif isinstance(results[0], dict):
        if all((isinstance(result, dict) for result in results[1:])):
            output = results[0]
            results = tqdm(results[1:])
            results.set_description('merging dicts')
            for d in results:
                output.update(d)
            return output
    return results


# endregion


def dispatch_data(
        num_p: int,
        data_iter: Union[Iterator, Iterable, List],
        args: Tuple,
        verbose: bool = __debug__
) -> List[Tuple]:
    if num_p <= 0:
        raise ValueError(f"The number of processes specified in `nump_p` must be positive; "
                         f"got {num_p}.")

    tic("Splitting task", verbose=verbose)
    splits = split_iter(
        it=data_iter,
        num_splits=num_p,
        use_tqdm=verbose
    )
    toc(verbose=verbose)

    num_p = len(splits)
    if num_p == 0:
        raise ValueError(f"The number of data splits is zero. "
                         f"Possibly no data was read from the provided iterator.")
    else:
        job_args = [None] * num_p
        for pidx in range(num_p):
            if verbose:
                hprint_message(
                    'pid', pidx,
                    'workload', len(splits[pidx])
                )
            job_args[pidx] = (pidx, splits[pidx], *args)
        return job_args


def start_jobs(jobs: [Union[List, Tuple]], interval: float = 0.01):
    for p in jobs:
        p.start()
        if interval != 0:
            sleep(interval)


def start_and_wait_jobs(jobs: [Union[List, Tuple]], interval: float = 0.01):
    start_jobs(jobs, interval=interval)
    for p in jobs:
        p.join()


def parallel_process(
        num_p,
        data_iter: Union[Iterator, Iterable, List],
        target: Callable, args: Tuple,
        ctx: BaseContext = None,
        verbose=__debug__
):
    if isinstance(target, MPTarget):
        target.use_queue = False
    if ctx is None:
        ctx = get_context('spawn')
    job_args = dispatch_data(
        num_p=num_p,
        data_iter=data_iter,
        args=args,
        verbose=verbose
    )
    jobs = [None] * num_p
    for i in range(num_p):
        jobs[i] = ctx.Process(target=target, args=job_args[i])
    start_and_wait_jobs(jobs)


def parallel_process_by_pool(
        num_p: int,
        data_iter: Union[Iterator, Iterable],
        target: Union[MPTarget, Callable],
        args: Tuple = (),
        verbose: bool = __debug__,
        merge_output: bool = False,
        mergers: Union[List, Tuple] = None,
        vertical_merge: bool = True,
        debug: bool = False,
        return_job_splits: bool = False,
        load_dumped_results: bool = False,
        result_dump_load_method: Callable[[str], Any] = pickle_load,
        wait_for_pool_close: bool = True,
        start_method: str = None,
        multiprocessing_module=multiprocessing,
        pool_object=None,
):
    """
    Parallelizes a given function or callable using a multiprocessing pool.

    Args:
        num_p: The number of processes to initiate.
        data_iter: An iterable providing the data source for the multi-processing.
        target: The function or callable to be parallelized.
        args: Additional arguments to be passed to the target function. Defaults to ().
        verbose: If True, prints progress and debug information.
        merge_output: If True, merges the output from all processes.
        mergers: Custom merge functions for merging the results.
        vertical_merge: If True, merges results vertically.
        debug: If True or 1, runs in debug mode (no parallelism).
            If set to an integer > 1, runs a limited number of iterations.
        return_job_splits: If True, returns the job splits along with the results.
        load_dumped_results: If True, loads results from dumped files.
        result_dump_load_method: Method to load dumped results.
        wait_for_pool_close: If True, waits for the pool to close before returning results. Defaults to True.
        start_method: Specifies the start method for the multiprocessing module.
            Typically values include 'fork', 'spawn'. Setting this to None will skip setting
            the "start method" for the multiprocessing module, and it will use its current setting.
        multiprocessing_module: The multiprocessing module to use.
            Defaults to the built-in multiprocessing module.
        pool_object: The pool object to use for multiprocessing.
            If this is None, we use the `Pool` object of `multiprocessing_module`.

    """
    if debug == 1 and merge_output:
        raise ValueError('debug is set True or 1, '
                         'in this case the result merge will not work; '
                         'change debug to an integer higher than 2')

    if num_p is None or num_p <= 0:
        num_p = get_suggested_num_workers(num_p)

    if num_p == 1:
        rst = target(0, data_iter, *args)
        if load_dumped_results:
            if isinstance(rst, str) and path.isfile(rst):
                rst = result_dump_load_method(rst)
            else:
                warnings.warn(f'Expected to load results from dumped files; '
                              f'in this case the returned result from each process '
                              f'must be a file path; '
                              f'got {type(rst)}')
        return rst

    if multiprocessing_module is None:
        multiprocessing_module = multiprocessing

    curr_start_method = multiprocessing_module.get_start_method(allow_none=True)
    if start_method is not None and start_method != curr_start_method:
        if start_method == 'spawn':
            multiprocessing_module.freeze_support()
        multiprocessing_module.set_start_method(start_method, force=True)
    elif curr_start_method == 'spawn':
        multiprocessing_module.freeze_support()

    if pool_object is None:
        pool_object = multiprocessing_module.Pool

    if isinstance(target, MPTarget):
        target.use_queue = False

    if isinstance(debug, int) and debug > 1:
        num_p = debug
        data_iter = islice(data_iter, 1000)

    job_args = dispatch_data(
        num_p=num_p,
        data_iter=data_iter,
        args=args,
        verbose=verbose
    )
    if debug is True or debug == 1:
        rst = target(*job_args)
    else:
        pool = pool_object(processes=num_p)

        try:
            rst = pool.starmap(target, job_args)
        except Exception as err:
            pool.close()
            if wait_for_pool_close:
                warnings.warn(f"waiting for multi-process pool of size {num_p} to close "
                              f"due to error '{err}'")
                pool.join()
            raise err

        pool.close()
        if wait_for_pool_close:
            hprint_message(f"waiting for multi-process pool of size {num_p} to close")
            pool.join()

    if load_dumped_results:
        if isinstance(rst[0], str) and path.isfile(rst[0]):
            rst = [
                result_dump_load_method(file_path)
                for file_path in tqdm(rst, desc='loading dumped files')
            ]
        else:
            warnings.warn(f'Expected to load results from dumped files; '
                          f'in this case the returned result from each process must be a file path; '
                          f'got {type(rst[0])}')

    if merge_output:
        rst = list(zip(*rst)) if vertical_merge else [rst]

        hprint_message(
            'mergers', mergers,
            'result types', [type(x) for x in rst],
            'result size', [(len(x) if hasattr(x, '__len__') else 'n/a') for x in rst]
        )
        rst = merge_results(result_collection=rst, mergers=mergers)
        if not vertical_merge:
            rst = rst[0]

    return (rst, (job_arg[1] for job_arg in job_args)) if return_job_splits else rst


def merge_results(result_collection, mergers: Union[List, Tuple] = None):
    def _default_merger_1(results, merge_method: str):
        if merge_method == 'list':
            return sum(results, [])
        elif merge_method == 'dict':
            return merge_mappings(results, in_place=True)
        elif merge_method == 'list_dict':
            return merge_list_valued_mappings(results, in_place=True)
        elif merge_method == 'set_dict':
            return merge_set_valued_mappings(results, in_place=True)
        elif merge_method == 'counter_dict':
            return merge_counter_valued_mappings(results, in_place=True)
        elif merge_method == 'sum':
            return sum(results)
        raise ValueError(
            f"The provided results does not support the default merge method {merge_method}."
        )

    def _default_merger_2(results):
        size_results = len(results)
        if size_results == 0:
            return results
        err = None
        rst_type = type(results[0])
        if rst_type in (int, float, bool):
            return sum(results)
        elif rst_type is list:
            return sum(results, [])
        elif rst_type is tuple:
            return sum(results, ())
        elif rst_type in (dict, defaultdict):
            peek_value = None
            for i in range(size_results):
                if len(results[i]) != 0:
                    peek_value = next(iter(results[i].values()))
                    break
            if isinstance(peek_value, List):
                return merge_list_valued_mappings(results[i:], in_place=True)
            elif isinstance(peek_value, Set):
                return merge_set_valued_mappings(results[i:], in_place=True)
            elif isinstance(peek_value, Counter):
                return merge_counter_valued_mappings(results[i:], in_place=True)
            else:
                try:
                    return sum_dicts(results[i:], in_place=True)
                except Exception as err:
                    pass
        raise ValueError(
            f"The provided results does not support the default merge. "
            f"Error: {err or f'type {rst_type} not supported'}."
        )

    return (
        tuple(
            (
                merger(results)
                if callable(merger)
                else _default_merger_1(results, merger)
            )
            for results, merger in zip(result_collection, mergers)
        )
        if mergers
        else tuple(_default_merger_2(results) for results in tqdm(result_collection))
    )
