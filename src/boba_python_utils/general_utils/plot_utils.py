from enum import Enum
from typing import Union, Callable

from pyspark.sql import DataFrame
from pyspark.sql.column import Column

from boba_python_utils.common_utils.function_helper import get_relevant_named_args


def _post_proc_plot(
        plt,
        title=None,
        title_fontsize=None,
        xlabel=None,
        ylabel=None,
        xlabel_fontsize=None,
        ylabel_fontsize=None,
        show=False,
        output_path=None,
        output_dpi=600,
        output_format='png'
):
    if xlabel:
        plt.xlabel(xlabel, fontsize=xlabel_fontsize or 16)
    if ylabel:
        plt.ylabel(ylabel, fontsize=ylabel_fontsize or 16)
    if title:
        plt.title(title, fontsize=title_fontsize or 16)
    plt.tight_layout()
    if show:
        plt.show()
    if output_path:
        plt.savefig(output_path, dpi=output_dpi, format=output_format)


class BucketCoding(str, Enum):
    INDEX = 'index'
    LeftBoundary = 'left_boundary'
    RightBoundary = 'right_boundary'
    Mean = 'mean'


def plot2d(
        data_obj: DataFrame,
        x_col: Union[str, Column],
        y_col: Union[str, Column],
        data_proc_method: Callable,
        plot_method: Union[str, Callable] = 'scatterplot',
        sample_method: Callable = None,
        sample_ratio_or_size: Union[float, int] = 200000,
        sample_seed: int = 0,
        bucketize_method: Callable = None,
        bucketize_x=None,
        bucket_coding_x: BucketCoding = BucketCoding.Mean,
        to_pandas_method: Callable = None,
        x_colname: str = None,
        y_colname: str = None,
        other_cols=None,
        clear_before_plot=True,
        **kwargs
):
    import matplotlib.pyplot as plt

    data_obj, x_col, y_col, x_colname, y_colname = data_proc_method(
        data_obj, x_col, y_col, x_colname, y_colname, other_cols
    )

    if 'xlabel' not in kwargs:
        kwargs['xlabel'] = x_colname
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = y_colname

    if sample_method:
        data_obj = sample_method(
            data_obj,
            sample_ratio_or_size,
            sample_seed
        )
    if bucketize_method:
        data_obj = bucketize_method(
            data_obj,
            x_colname,
            bucketize_x,
            bucket_coding_x
        )
    if to_pandas_method:
        pdf = to_pandas_method(data_obj)
    elif hasattr(data_obj, 'toPandas'):
        pdf = data_obj.toPandas()
    else:
        raise ValueError()

    if clear_before_plot:
        plt.clf()
    post_proc_kwargs = get_relevant_named_args(_post_proc_plot, **kwargs)
    if isinstance(plot_method, str):
        import seaborn as sns
        plot_method = getattr(sns, plot_method)
    plot_method(
        data=pdf,
        x=x_colname,
        y=y_colname,
        **{k: v for k, v in kwargs.items() if k not in post_proc_kwargs}
    )
    _post_proc_plot(plt=plt, **post_proc_kwargs)
