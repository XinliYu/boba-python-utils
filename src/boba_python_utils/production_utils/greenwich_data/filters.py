from functools import partial
from pyspark.sql.catalog import Column
import pyspark.sql.functions as F
from pyspark.sql.types import BooleanType


@F.udf(returnType=BooleanType())
def _callable_filter_wrap(x, filter_func, waiver_func):
    return waiver_func(x) or filter_func(x)


def _df_filter(df, cond):
    return df.where(cond)


def solve_filter(_filter, ignore_filter_cond=None):
    if callable(_filter):
        if ignore_filter_cond is not None:
            if callable(ignore_filter_cond):
                return partial(
                    _callable_filter_wrap, filter_func=_filter, waiver_func=ignore_filter_cond
                )
            else:
                raise ValueError("the 'waiver' argument must also be a callable")
        else:
            return F.udf(_filter, returnType=BooleanType())
    elif isinstance(_filter, Column):
        if ignore_filter_cond is None:
            return partial(_df_filter, cond=_filter)
        else:
            if isinstance(ignore_filter_cond, Column):
                return partial(_df_filter, cond=(_filter | ignore_filter_cond))
            else:
                raise ValueError("the 'waiver' argument must also be a get_spark column")
    return _filter
