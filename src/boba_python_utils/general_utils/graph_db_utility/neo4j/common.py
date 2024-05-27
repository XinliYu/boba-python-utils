import datetime
from enum import Enum
from typing import Mapping
from datetime import datetime

from neo4j import Driver, AsyncGraphDatabase
from os import path

from boba_python_utils.common_utils.iter_helper import tqdm_wrap
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.general_utils.graph_db_utility.neo4j.constants import VERBOSE, DEFAULT_PATH_NEO4J, DEFAULT_DIRNAME_NEO4J_IMPORT
from boba_python_utils.io_utils.json_io import write_json
from boba_python_utils.path_utility.path_string_operations import add_ending_path_sep, get_main_name

import asyncio


class ResultConsumption(str, Enum):
    Lazy = 'lazy'
    Single = 'single'
    All = 'all'
    ReturnAll = 'return_all'


def mapping_to_params(d: Mapping):
    out = []
    for k, v in d.items():
        if isinstance(v, bool):
            v = 'True' if v else 'False'
        elif isinstance(v, str):
            v = f"'{v}'"

        out.append(f"{k}: {v}")

    return '{' + ', '.join(out) + '}'


def run_query(
        driver: Driver,
        query: str,
        result_consumption: ResultConsumption = ResultConsumption.Lazy,
        verbose=VERBOSE,
        **kwargs
):
    with driver.session() as session:
        if verbose:
            hprint_message(
                'query', query,
                'result_consumption', result_consumption,
                title='executing Neo4j query'
            )
        result = session.run(query, **kwargs)
        records = None
        if result_consumption == ResultConsumption.All:
            records = result.consume()
        elif result_consumption == ResultConsumption.ReturnAll:
            records = result.to_eager_result().records
        elif result_consumption == ResultConsumption.Single:
            records = result.single()
        return result, records


def get_num_nodes(driver: Driver):
    _, rcd = run_query(
        driver,
        'MATCH (n) RETURN count(n) as count',
        ResultConsumption.Single
    )
    return rcd[0]


def export_to_csv_query(output_path, query, **kwargs):
    return f'CALL apoc.export.csv.query("{query}", "{output_path}", {mapping_to_params(kwargs)})'


def export_to_csv(driver, output_path, query, **kwargs):
    return run_query(
        driver,
        export_to_csv_query(output_path=output_path, query=query, **kwargs),
        ResultConsumption.Single
    )


def import_csv_query(input_path, query, batch_size=0, header=False, delimiter='\t', neo4j_import_path=None):
    if neo4j_import_path is None:
        neo4j_import_path = add_ending_path_sep(path.expanduser(
            path.join(DEFAULT_PATH_NEO4J, DEFAULT_DIRNAME_NEO4J_IMPORT)
        ))
    input_path = path.expanduser(input_path)
    if input_path.startswith(neo4j_import_path):
        input_path = input_path[len(neo4j_import_path):]

    delimiter = repr(delimiter)[1:-1]
    if delimiter[0] == '\\' and delimiter[1] == '\\':
        delimiter = delimiter[1:]

    query_pieces = []

    query_pieces.append('LOAD CSV')
    if header:
        query_pieces.append('WITH HEADERS')
    query_pieces.append(f"FROM 'file:///{input_path}' AS line")
    if delimiter:
        query_pieces.append(f"FIELDTERMINATOR '{delimiter}'")
    if batch_size:
        query_pieces.append('CALL { ')
        query_pieces.append('WITH line')
    query_pieces.append(query)
    if batch_size:
        query_pieces.append(f" }} IN TRANSACTIONS OF {batch_size} ROWS")

    return ' '.join(query_pieces)


def _import_csv_simple(pid, input_path, *args):
    header = False
    delimiter = '\t'
    query = "MATCH (c:Customer {id: line[0]}), (e:Entity {entity: line[1]}) MERGE (c)-[:INTERACT_WITH]->(e)"
    batch_size = 100000
    # n4j = Neo4j(server='bolt://10.0.34.241:7687')
    # driver = n4j._driver
    AUTH = ("neo4j", "SLAB1234")

    global log_dir
    max_retry = 5

    count_success = count_fail = 0

    async def main():
        input_iter = tqdm_wrap(input_path, use_tqdm=True, tqdm_msg=f'prog ({pid})')
        for _input_path in input_iter:
            failed = False
            err = None

            async def do_job(driver):
                # result = await tx.run("MATCH (a:Person) RETURN a.name AS name")
                # records = await result.values()
                # return records

                retry = 0
                while retry <= max_retry:
                    failed = False
                    records = err = rlt = None

                    try:
                        _query = import_csv_query(
                            input_path=_input_path,
                            query=query,
                            batch_size=batch_size,
                            header=header,
                            delimiter=delimiter
                        )
                        rlt = await driver.run(_query)
                        status = 'succeeded'
                        records = await rlt.values()
                        print(records)
                        # count_success += 1
                    except Exception as err:
                        failed = True
                        status = 'failed'
                        # count_fail += 1

                    if log_dir:
                        log_obj = {
                            'status': status,
                            'attempt': retry,
                            'file': path.basename(_input_path),
                            'time': str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')),
                            'response': str(records)
                        }

                        if failed and ('err' in locals() or 'err' in globals()):
                            log_obj['error'] = str(err)

                        write_json(
                            log_obj,
                            path.join(log_dir, f'{get_main_name(_input_path)}-{status}.json')
                        )

                    if failed:
                        retry += 1
                    else:
                        break

            async with AsyncGraphDatabase.driver('bolt://10.0.34.241:7687', auth=AUTH) as driver:
                async with driver.session() as session:
                    records = await session.read_transaction(do_job)
                    print(records)

    asyncio.run(main())

# def import_csv(driver, input_path, query, batch_size, split=None, parallel=64):
#     pass
#
#
# @attrs(slots=True)
# class Neo4j:
#     _server = attrib(type=str, default=DEFAULT_SERVER)
#     _path_neo4j = attrib(type=str, default=DEFAULT_PATH_NEO4J)
#     _path_neo4j_import = attrib(type=str, default=None)
#     _user = attrib(type=str, default=DEFAULT_USER)
#     _password = attrib(type=str, default=DEFAULT_PASSWORD)
#     _driver = attrib(type=Driver, default=None)
#
#     def __attrs_post_init__(self):
#         if self._driver is None:
#             self._driver = AsyncGraphDatabase.driver(self._server, auth=(self._user, self._password))
#
#         if not self._path_neo4j_import:
#             self._path_neo4j_import = path.join(self._path_neo4j, DEFAULT_DIRNAME_NEO4J_IMPORT)
#
#         try:
#             num_nodes = self.num_nodes()
#         except Exception as err:
#             print("unable to connect to neo4j server")
#             print(err)
#             return
#
#         hprint_message(
#             'num nodes', num_nodes,
#             title='successfully connected to neo4j'
#         )
#
#     def num_nodes(self):
#         return get_num_nodes(self._driver)
#
#
# n4j = Neo4j(server='bolt://10.0.34.241:7687')
#
# # export_to_csv_query(output_path='results.csv', query="MATCH (e:Entity) RETURN e.entity LIMIT 10", quotes=False)
# # export_to_csv(n4j._driver, 'results2.csv', "MATCH (e:Customer) RETURN e.id LIMIT 10", quotes=False)
#
# # import_csv_query('/home/ec2-user/neo4j/import/interactions/part-00001-85e11ef8-62f5-424c-a2ee-b5ca3a45f64d-c000.csv', query="MATCH (c:Customer {id: line[0]}), (e:Entity {entity: line[1]}) MERGE (c)-[:INTERACT_WITH]->(e)", batch_size=50000)
# # #
# # tic()
# # _import_csv_simple('/home/ec2-user/neo4j/import/interactions/part-00001-85e11ef8-62f5-424c-a2ee-b5ca3a45f64d-c000.csv', driver=n4j._driver, query="MATCH (c:Customer {id: line[0]}), (e:Entity {entity: line[1]}) MERGE (c)-[:INTERACT_WITH]->(e)", batch_size=100000)
# # toc()
# input_path = 'interactions'
#
# if not path.exists(input_path):
#     input_path = path.join(DEFAULT_PATH_NEO4J, DEFAULT_DIRNAME_NEO4J_IMPORT, input_path)
# prev_log_dir = get_paths_by_pattern(input_path, '*-succeeded.json', sort_use_basename=True)
# log_dir = path.join(input_path, f'improt_log-{str(int(time.time() * 1000000))}')
# if path.exists(input_path):
#     if path.isdir(input_path):
#         input_path = get_paths_by_pattern(
#             input_path,
#             pattern='*.csv',
#             full_path=True,
#             sort=True,
#             sort_use_basename=True
#         )
#
# mp_process = MPTarget(
#     name='entity linking',
#     target=_import_csv_simple,
#     result_dump_path=None,
#     pass_each_data_item=False,
#     result_dump_method=None
# )
#
# parallel_process_by_pool(
#     num_p=5,
#     data_iter=input_path,
#     target=mp_process
# )
#
# # ===============
# import datetime
# import time
# from enum import Enum
# from functools import partial
# from typing import Mapping
# from datetime import datetime
#
# import tqdm
# from attr import attrs, attrib
# from neo4j import GraphDatabase, Driver, Query, AsyncGraphDatabase
# from os import path
#
# from boba_python_utils.common_utils.iter_helper import tqdm_wrap
# from boba_python_utils.general_utils.console_util import hprint_message
# from boba_python_utils.general_utils.graph_db_utility.neo4j.constants import DEFAULT_SERVER, DEFAULT_USER, DEFAULT_PASSWORD, VERBOSE, DEFAULT_PATH_NEO4J, DEFAULT_DIRNAME_NEO4J_IMPORT
# from boba_python_utils.general_utils.ioex import write_json
# from boba_python_utils.general_utils.multiprocess_utility import MPTarget, parallel_process_by_pool
# from boba_python_utils.general_utils.path_utility.path_string_operations import add_ending_path_sep, get_main_name
# from boba_python_utils.general_utils.pathex import get_paths_by_pattern
# from boba_python_utils.general_utils.time_utility import tic, toc
#
# import asyncio
#
# def import_csv_query(input_path, query, batch_size=0, header=False, delimiter='\t', neo4j_import_path=None):
#     if neo4j_import_path is None:
#         neo4j_import_path = add_ending_path_sep(path.expanduser(
#             path.join(DEFAULT_PATH_NEO4J, DEFAULT_DIRNAME_NEO4J_IMPORT)
#         ))
#     input_path = path.expanduser(input_path)
#     if input_path.startswith(neo4j_import_path):
#         input_path = input_path[len(neo4j_import_path):]
#
#     delimiter = repr(delimiter)[1:-1]
#     if delimiter[0] == '\\' and delimiter[1] == '\\':
#         delimiter = delimiter[1:]
#
#     query_pieces = []
#
#     query_pieces.append('LOAD CSV')
#     if header:
#         query_pieces.append('WITH HEADERS')
#     query_pieces.append(f"FROM 'file:///{input_path}' AS line")
#     if delimiter:
#         query_pieces.append(f"FIELDTERMINATOR '{delimiter}'")
#     if batch_size:
#         query_pieces.append('CALL { ')
#         query_pieces.append('WITH line')
#     query_pieces.append(query)
#     if batch_size:
#         query_pieces.append(f" }} IN TRANSACTIONS OF {batch_size} ROWS")
#
#     return ' '.join(query_pieces)


# input_path = 'interactions'
#
# if not path.exists(input_path):
#     input_path = path.join(DEFAULT_PATH_NEO4J, DEFAULT_DIRNAME_NEO4J_IMPORT, input_path)
# prev_log_dir = get_paths_by_pattern(input_path, '*-succeeded.json', sort_use_basename=True)
# log_dir = path.join(input_path, f'improt_log-{str(int(time.time() * 1000000))}')
# if path.exists(input_path):
#     if path.isdir(input_path):
#         input_path = get_paths_by_pattern(
#             input_path,
#             pattern='*.csv',
#             full_path=True,
#             sort=True,
#             sort_use_basename=True
#         )
#
#
# def get_query(file_path):
#     header = False
#     delimiter = '\t'
#     query = "MATCH (c:Customer {id: line[0]}), (e:Entity {entity: line[1]}) CREATE (c)-[:INTERACT_WITH]->(e)"
#     batch_size = 50000
#     _query = import_csv_query(
#         input_path=file_path,
#         query=query,
#         batch_size=batch_size,
#         header=header,
#         delimiter=delimiter
#     )
#     return _query
# rlt = await driver.run(_query)
# records = await rlt.values()
# print(records)


# async def do_job(file_path):
#     print('job for ' + file_path)
#     AUTH = ("neo4j", "SLAB1234")
#     async with AsyncGraphDatabase.driver('bolt://10.0.34.241:7687', auth=AUTH) as driver:
#         async with driver.session() as session:
#             result = await session.run(get_query(file_path))
#             records = await result.values()
#             print(records)
#             semaphore.get_nowait()

# from time import sleep
# async def main(loop):
#     for file_path in tqdm_wrap(input_path[980:], use_tqdm=True, tqdm_msg='proc'):
#         await semaphore.put(file_path)  # It does'n matter what we put in the queue. We use it as semaphore.
#         sleep(0.1)
#         loop.create_task(do_job(file_path))
# all the tasks are scheduled at the moment but not all done


# asyncio.run(main())
# tic()
# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)
# semaphore = asyncio.Queue(maxsize=250)  # Max 3 processes
# loop.run_until_complete(main(loop))
# loop.run_until_complete(asyncio.gather(*asyncio.Task.all_tasks()))  # Wait for all tasks in the loop.
# loop.close()
# toc()


# 10269.552491664886
