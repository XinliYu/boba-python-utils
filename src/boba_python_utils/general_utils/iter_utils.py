import uuid
from itertools import chain, islice
from typing import Iterator, Iterable, Union, List


def with_uuid(it, prefix='', suffix=''):
    yield from ((prefix + str(uuid.uuid4()) + suffix, x) for x in it)


def with_names(it, name_format: str = None, name_prefix='', name_suffix=''):
    if name_format is None or name_format == 'uuid':
        return with_uuid(it=it, prefix=name_prefix, suffix=name_suffix)
    else:
        for i, x in enumerate(it):
            yield name_prefix + name_format.format(i) + name_suffix, x


def chunk_iter(it: Union[Iterator, Iterable], chunk_size: int, as_list=False) -> Union[Iterator[Iterator], Iterator[List]]:
    """
    Returns an iterator that iterates through chunks of the provided iterator; each chunk is also represented by an iterator that can iterate through the elements in it.
    :param it: the iterator to chunk.
    :param chunk_size: the size of each chunk.
    :return: an iterator that iterates through chunks of the provided iterator.
    """
    it = iter(it)
    if as_list:
        while True:
            cur = list(islice(it, chunk_size))
            if not cur:
                break
            yield cur
    else:
        while True:
            cur_it = islice(it, chunk_size)
            try:
                first = next(cur_it)
            except StopIteration:
                break
            yield chain((first,), cur_it)
