from typing import Mapping, Iterable, Any, Tuple, List, Optional
from typing import Union

from attr import attrib, attrs
from pyspark.sql import DataFrame, Column

import boba_python_utils.spark_utils.spark_functions as F
import boba_python_utils.spark_utils as sparku
from boba_python_utils.common_utils.array_helper import first_half, second_half
from boba_python_utils.common_utils.iter_helper import iter__, zip__
from boba_python_utils.common_utils.map_helper import join_mapping_by_values
from boba_python_utils.common_utils.typing_helper import make_list, make_tuple
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.spark_utils import VERBOSE
from boba_python_utils.spark_utils.analysis import agg_trans_prob
from boba_python_utils.spark_utils.common import has_col, solve_name_and_column, solve_names_and_columns, get_colname
from boba_python_utils.spark_utils.join_and_filter import join_on_columns
from boba_python_utils.spark_utils.typing import NameOrColumn, AliasAndColumn, resolve_name_or_column, resolve_names_or_columns, ColumnOrAliasedColumn, NamesOrColumns
from boba_python_utils.string_utils.prefix_suffix import add_suffix


@attrs(slots=True)
class LeveledStatsAggregationArgs:
    avg_colnames = attrib(type=Mapping[str, str], default=None)
    extra_key_avg_colnames = attrib(type=Mapping[str, str], default=None)
    main_stat_colname = attrib(type=str, default=None)
    extra_key_main_stat_colname = attrib(type=str, default=None)
    collect_list_colnames = attrib(type=Mapping[str, Tuple[str, ...]], default=None)
    collect_set_colnames = attrib(type=Mapping[str, Tuple[str, ...]], default=None)
    collect_list_max_size = attrib(type=int, default=None)
    collect_set_max_size = attrib(type=int, default=None)
    extra_key_collect_list_colnames = attrib(type=Mapping[str, Tuple[str, ...]], default=None)
    extra_key_collect_set_colnames = attrib(type=Mapping[str, Tuple[str, ...]], default=None)
    extra_key_collect_list_max_size = attrib(type=int, default=None)
    extra_key_collect_set_max_size = attrib(type=int, default=None)
    other_agg_args = attrib(type=Mapping[str, Any], default=None)
    extra_key_other_agg_args = attrib(type=Mapping[str, Any], default=None)
    other_agg_input_colnames = attrib(type=Iterable[str], default=None)
    extra_key_other_agg_input_colnames = attrib(type=Iterable[str], default=None)

    def __attrs_post_init__(self):
        if self.collect_list_colnames is not None:
            self.collect_list_colnames = make_tuple(self.collect_list_colnames)
        if self.extra_key_collect_list_colnames is not None:
            self.extra_key_collect_list_colnames = make_tuple(self.extra_key_collect_list_colnames)
        if self.collect_set_colnames is not None:
            self.collect_set_colnames = make_tuple(self.collect_set_colnames)
        if self.extra_key_collect_set_colnames is not None:
            self.extra_key_collect_set_colnames = make_tuple(self.extra_key_collect_set_colnames)


def _add_trans_prob(
        df_leveled_agg: DataFrame,
        aggregate_transition_prob: bool,
        aggregate_pair_ratio: bool,
        transition_source_group_keys,
        transition_target_group_keys,
        extra_group_keys,
        count_colname: str,
        transition_source_count_colname: str,
        transition_prob_colname: str,
        pair_source_count_colname: str,
        pair_ratio_colname: str,
        keep_transition_count_cols,
        verbose: bool = VERBOSE
):
    transition_source_group_keys_with_extra_group_keys = [
        *(extra_group_keys or ()), *transition_source_group_keys
    ]

    if verbose:
        hprint_message(
            'df_leveled_agg', df_leveled_agg,
            'aggregate_transition_prob', aggregate_transition_prob,
            'aggregate_pair_ratio', aggregate_pair_ratio,
            'transition_source_group_keys', transition_source_group_keys,
            'transition_target_group_keys', transition_target_group_keys,
            'extra_group_keys', extra_group_keys,
            'count_colname', count_colname,
            'transition_source_count_colname', transition_source_count_colname,
            'transition_prob_colname', transition_prob_colname,
            'pair_source_count_colname', pair_source_count_colname,
            'pair_ratio_colname', pair_ratio_colname,
            'keep_transition_count_cols', keep_transition_count_cols
        )

    if aggregate_transition_prob or aggregate_pair_ratio:
        if not transition_source_group_keys or not transition_target_group_keys:
            raise ValueError(
                "'transition_source_group_keys' and 'transition_target_group_keys' "
                "must be specified when 'aggregate_transition_prob' or 'aggregate_pair_ratio' "
                "is True"
            )
        self_transition_cond = F.and_(
            (F.col_(x) == F.col_(y))
            for x, y in zip(transition_source_group_keys, transition_target_group_keys)
        ).alias('is_self_transition')
        if verbose:
            hprint_message('self_transition_cond', self_transition_cond)

    if aggregate_transition_prob:
        if not transition_source_count_colname:
            raise ValueError("'transition_source_count_colname' must be specified "
                             "when 'aggregate_transition_prob' is set True")
        if not transition_prob_colname:
            raise ValueError("'transition_prob_colname' must be specified "
                             "when 'aggregate_transition_prob' is set True")
        df_leveled_agg_joint_trans_prob = agg_trans_prob(
            df_data=df_leveled_agg,
            transition_source_cols=transition_source_group_keys_with_extra_group_keys,
            count_col=count_colname,
            transition_count_colname=None,
            output_transition_source_count_colname=transition_source_count_colname,
            output_transition_prob_colname=transition_prob_colname,
            keep_transition_count_cols=keep_transition_count_cols,
            join_with_input=True
        )
        if verbose:
            sparku.show_counts(df_leveled_agg_joint_trans_prob, self_transition_cond)
            sparku.show_counts(
                df_leveled_agg_joint_trans_prob,
                F.col(transition_prob_colname).isNull()
            )
    else:
        df_leveled_agg_joint_trans_prob = df_leveled_agg

    if aggregate_pair_ratio:
        if not pair_source_count_colname:
            raise ValueError("'pair_source_count_colname' must be specified "
                             "when 'aggregate_pair_ratio' is set True")
        if not pair_ratio_colname:
            raise ValueError("'pair_ratio_colname' must be specified "
                             "when 'aggregate_pair_ratio' is set True")

        df_leveled_agg_joint_trans_prob = join_on_columns(
            df_leveled_agg_joint_trans_prob,
            agg_trans_prob(
                df_data=df_leveled_agg_joint_trans_prob.where(~self_transition_cond),
                transition_source_cols=transition_source_group_keys_with_extra_group_keys,
                transition_target_cols=transition_target_group_keys,
                count_col=count_colname,
                transition_count_colname=None,
                output_transition_source_count_colname=pair_source_count_colname,
                output_transition_prob_colname=pair_ratio_colname,
                keep_transition_count_cols=keep_transition_count_cols,
                join_with_input=False
            ),
            [*transition_source_group_keys_with_extra_group_keys, *transition_target_group_keys],
            how='left'
        )
        if verbose:
            sparku.show_counts(df_leveled_agg_joint_trans_prob, self_transition_cond)
            sparku.show_counts(
                df_leveled_agg_joint_trans_prob,
                F.col(pair_ratio_colname).isNull()
            )
    if df_leveled_agg_joint_trans_prob is not df_leveled_agg:
        return sparku.cache__(
            df_leveled_agg_joint_trans_prob,
            name='df_leveled_agg_joint_trans_prob',
            unpersist=df_leveled_agg
        )
    else:
        return df_leveled_agg


def _add_group_keys_suffix(
        group_keys,
        pairwise_group_key_suffix_first,
        pairwise_group_key_suffix_second
):
    return [
        *(add_suffix(x, suffix=pairwise_group_key_suffix_first) for x in group_keys),
        *(add_suffix(x, suffix=pairwise_group_key_suffix_second) for x in group_keys)
    ]


def _split_collect_list_or_set_cols(
        df: DataFrame,
        group_keys,
        collect_list_or_set_cols,
        concat_list_or_set_cols: List
):
    if collect_list_or_set_cols is not None:
        if isinstance(collect_list_or_set_cols, Mapping):
            _collect_list_or_set_cols = {}
            for _colname, _col in collect_list_or_set_cols.items():
                if _colname in group_keys:
                    _collect_list_or_set_cols[_colname] = _col
                else:
                    concat_list_or_set_cols.append(_colname)

            return _collect_list_or_set_cols
        else:
            _collect_list_or_set_cols = []
            for _col in iter__(collect_list_or_set_cols):
                _colname = get_colname(df, _col)
                if _colname in group_keys:
                    _collect_list_or_set_cols.append(_col)
                else:
                    concat_list_or_set_cols.append(_col)

            return _collect_list_or_set_cols
    return collect_list_or_set_cols


def iter_leveled_aggregations(
        df_data: DataFrame,
        group_keys_ladder: Iterable[Iterable[Iterable[str]]],
        avg_cols,
        count_col: Union[str, Mapping[str, str]],
        weighted_aggregation_by_count_col: bool,
        extra_group_keys: Iterable[Union[str, Column]] = None,
        collect_list_cols: Union[str, Mapping[str, Iterable[str]]] = None,
        collect_list_max_size: int = None,
        collect_set_cols: Union[str, Mapping[str, Iterable[str]]] = None,
        collect_set_max_size: int = None,
        unpersist_input_data: bool = True,
        aggregation_label: str = None,
        other_agg_kwargs: Mapping = None,
        # region supports pairwise aggregations & transition prob
        pairwise_group_key_suffix_first: str = None,
        pairwise_group_key_suffix_second: str = None,
        aggregate_transition_prob: bool = False,
        aggregate_pair_ratio: bool = False,
        transition_source_count_colname: str = 'trans_src_count',
        transition_prob_colname: str = 'trans_prob',
        pair_source_count_colname: str = 'pair_src_count',
        pair_ratio_colname: str = 'pair_ratio',
        keep_transition_count_cols: bool = True,
        # endregion,
        verbose: bool = VERBOSE
):
    has_count_col = has_col(df_data, solve_name_and_column(count_col)[1])
    avg_col_names = tuple(avg_cols)
    count_colname = next(iter__(count_col))
    extra_group_keys = make_list(extra_group_keys, atom_types=(str, Column))
    is_pairwise_aggregation = (
            bool(pairwise_group_key_suffix_first) and bool(pairwise_group_key_suffix_second)
    )

    if verbose:
        hprint_message(
            'df_data', df_data,
            'group_keys_ladder', group_keys_ladder,
            'extra_group_keys', extra_group_keys,
            'count_col', count_col,
            'count_colname', count_colname,
            'has_count_col', has_count_col,
            'avg_cols', avg_cols,
            'avg_col_names', avg_col_names,
            'collect_list_cols', collect_list_cols,
            'collect_set_cols', collect_set_cols,
            'other_agg_kwargs', other_agg_kwargs,
            'pairwise_group_key_suffix_first', pairwise_group_key_suffix_first,
            'pairwise_group_key_suffix_second', pairwise_group_key_suffix_second,
            'is_pairwise_aggregation', is_pairwise_aggregation,
            'aggregate_transition_prob', aggregate_transition_prob,
            'transition_source_count_colname', transition_source_count_colname,
            'transition_prob_colname', transition_prob_colname,
            'keep_transition_count_cols', keep_transition_count_cols,
            title=aggregation_label
        )

    def __add_trans_prob():
        return _add_trans_prob(
            df_leveled_agg=df_leveled_agg,
            aggregate_transition_prob=aggregate_transition_prob,
            aggregate_pair_ratio=aggregate_pair_ratio,
            transition_source_group_keys=first_half(_group_keys),
            transition_target_group_keys=second_half(_group_keys),
            extra_group_keys=extra_group_keys,
            count_colname=count_colname,
            transition_source_count_colname=transition_source_count_colname,
            transition_prob_colname=transition_prob_colname,
            pair_source_count_colname=pair_source_count_colname,
            pair_ratio_colname=pair_ratio_colname,
            keep_transition_count_cols=keep_transition_count_cols
        )

    df_aggs_prev_level = None

    def _print_df_leveled_agg():
        if verbose:
            hprint_message(
                'level', i,
                'group_keys', _group_keys,
                'df_leveled_agg', df_leveled_agg,
                'extra_group_keys', extra_group_keys,
                title=f'{aggregation_label} ({group_keys_with_extra_group_keys})'
            )

    concat_list_cols, concat_set_cols = [], []
    for i, group_keys in enumerate(group_keys_ladder):
        df_aggs_curr_level = []
        if i == 0:
            # ! We only allow one top level aggregation;
            # ! this is the finest aggregation whose group keys
            # ! is a superset of all other aggregations.
            _group_keys = tuple(next(iter(group_keys)))
            if is_pairwise_aggregation:
                _group_keys = _add_group_keys_suffix(
                    group_keys=_group_keys,
                    pairwise_group_key_suffix_first=pairwise_group_key_suffix_first,
                    pairwise_group_key_suffix_second=pairwise_group_key_suffix_second
                )
            group_keys_with_extra_group_keys = [*(extra_group_keys or ()), *_group_keys]
            df_leveled_agg = sparku.cache__(
                sparku.aggregate(
                    df=df_data,
                    group_cols=group_keys_with_extra_group_keys,
                    # the first aggregation may be a name mapping
                    # from an aggregated signal column name to its source signal column name,
                    # because the aggregated signal column names
                    # can be different from the source signal column names
                    avg_cols=avg_cols,
                    count_col=count_col,
                    avg_by_count_col=has_count_col,
                    weighted_avg_by_count_col=weighted_aggregation_by_count_col and has_count_col,
                    collect_list_cols=collect_list_cols,
                    collect_list_max_size=collect_list_max_size,
                    collect_set_cols=collect_set_cols,
                    collect_set_max_size=collect_set_max_size,
                    ignore_agg_cols_of_conflict_names=True,
                    **(other_agg_kwargs or {})
                ),
                name=f'{aggregation_label or "leveled aggregation"} ({_group_keys})',
                unpersist=df_data if unpersist_input_data else None
            )

            df_leveled_agg = __add_trans_prob()
            if verbose:
                hprint_message(
                    f'df_leveled_agg columns ({_group_keys})',
                    df_leveled_agg.columns
                )
            df_aggs_curr_level.append(df_leveled_agg)
            _print_df_leveled_agg()
        else:
            for _group_keys in group_keys:
                if is_pairwise_aggregation:
                    _group_keys = _add_group_keys_suffix(
                        group_keys=_group_keys,
                        pairwise_group_key_suffix_first=pairwise_group_key_suffix_first,
                        pairwise_group_key_suffix_second=pairwise_group_key_suffix_second
                    )
                group_keys_with_extra_group_keys = [*(extra_group_keys or ()), *_group_keys]
                df_leveled_agg = sparku.cache__(
                    sparku.aggregate(
                        # we assume the last dataframe in
                        # `df_customer_level_stats_collection`
                        # can always be used for the next level of aggregation
                        df=df_aggs_prev_level[-1],
                        group_cols=group_keys_with_extra_group_keys,
                        # further aggregations are performed on the aggregated signal columns,
                        # these subsequent aggregations keep the aggregated signal column names,
                        # so it only needs the aggregated signal column names
                        # (mapping keys if `avg_cols` is a name mapping).
                        avg_cols=avg_col_names,
                        count_col=count_colname,
                        avg_by_count_col=True,
                        weighted_avg_by_count_col=True,
                        collect_list_cols=collect_list_cols,
                        collect_list_max_size=collect_list_max_size,
                        collect_set_cols=collect_set_cols,
                        collect_set_max_size=collect_set_max_size,
                        concat_list_cols=concat_list_cols,
                        concat_list_max_size=collect_list_max_size,
                        concat_set_cols=concat_set_cols,
                        concat_set_max_size=collect_set_max_size,
                        ignore_agg_cols_of_conflict_names=True,
                        **{
                            k: (tuple(v) if isinstance(v, Mapping) else v)
                            for k, v in other_agg_kwargs.items()
                        } if other_agg_kwargs else {}
                    ),
                    name=f'{aggregation_label or "leveled aggregation"} ({_group_keys})',
                )
                df_leveled_agg = __add_trans_prob()
                if verbose:
                    hprint_message(
                        f'df_leveled_agg columns ({_group_keys})',
                        df_leveled_agg.columns
                    )
                df_aggs_curr_level.append(df_leveled_agg)
                _print_df_leveled_agg()

        yield group_keys, group_keys_with_extra_group_keys, df_aggs_curr_level
        df_aggs_prev_level = df_aggs_curr_level

        # import pdb;
        # pdb.set_trace()
        collect_list_cols = _split_collect_list_or_set_cols(
            df_data,
            group_keys_with_extra_group_keys,
            collect_list_cols,
            concat_list_cols
        )

        collect_set_cols = _split_collect_list_or_set_cols(
            df_data,
            group_keys_with_extra_group_keys,
            collect_set_cols,
            concat_set_cols
        )


def iter_leveled_aggregations_(
        df_data: DataFrame,
        group_keys_ladder: Iterable[Iterable[Iterable[str]]],
        extra_group_keys: Iterable[NameOrColumn],
        stats_agg_args: LeveledStatsAggregationArgs,
        count_col: ColumnOrAliasedColumn,
        extra_key_count_col: ColumnOrAliasedColumn,
        popularity_colname: str,
        # region optional args for leveled aggregation
        weighted_aggregation_by_count_col: bool,
        unpersist_input_data: bool = True,
        aggregation_label: str = None,
        aggregation_label_for_extra_group_keys: str = None,
        # endregion
        # region supports pairwise aggregations & transition prob
        pairwise_group_key_suffix_first: str = None,
        pairwise_group_key_suffix_second: str = None,
        aggregate_transition_prob: bool = False,
        aggregate_pair_ratio: bool = False,
        transition_source_count_colname: str = None,
        transition_prob_colname: str = None,
        transition_source_count_colname_for_extra_group_keys: str = None,
        transition_prob_colname_for_extra_group_keys: str = None,
        transition_source_count_exclude_self_loop_colname: str = None,
        transition_prob_exclude_self_loop_colname: str = None,
        transition_source_count_exclude_self_loop_colname_for_extra_group_keys: str = None,
        transition_prob_exclude_self_loop_colname_for_extra_group_keys: str = None,
        keep_transition_count_cols: bool = True,
        # endregion
        verbose: bool = VERBOSE,
):
    avg_colnames = stats_agg_args.avg_colnames
    extra_key_avg_colnames = stats_agg_args.extra_key_avg_colnames
    collect_list_colnames = stats_agg_args.collect_list_colnames
    collect_list_max_size = stats_agg_args.collect_list_max_size
    extra_key_collect_list_colnames = stats_agg_args.extra_key_collect_list_colnames
    extra_key_collect_list_max_size = stats_agg_args.extra_key_collect_list_max_size
    collect_set_colnames = stats_agg_args.collect_set_colnames
    collect_set_max_size = stats_agg_args.collect_set_max_size
    extra_key_collect_set_colnames = stats_agg_args.extra_key_collect_set_colnames
    extra_key_collect_set_max_size = stats_agg_args.extra_key_collect_set_max_size
    other_agg_args = stats_agg_args.other_agg_args
    extra_key_other_agg_args = stats_agg_args.extra_key_other_agg_args

    # region resolve stats and count columns
    extra_group_keys = make_list(extra_group_keys, atom_types=(str, Column))
    if isinstance(extra_key_avg_colnames, Mapping):
        if isinstance(avg_colnames, Mapping):
            avg_colnames = join_mapping_by_values(
                avg_colnames, extra_key_avg_colnames,
                keep_original_value_for_mis_join=True
            )
        else:
            raise ValueError("'avg_colnames' must be a name mapping "
                             "when 'extra_key_avg_colnames' is a name mapping; "
                             f"got {avg_colnames}")

    if isinstance(extra_key_collect_list_colnames, Mapping):
        if isinstance(collect_list_colnames, Mapping):
            collect_list_colnames = join_mapping_by_values(
                collect_list_colnames, extra_key_collect_list_colnames,
                keep_original_value_for_mis_join=True
            )
        else:
            raise ValueError("'collect_list_colnames' must be a name mapping "
                             "when 'extra_key_collect_list_colnames' is a name mapping; "
                             f"got {collect_list_colnames}")

    if isinstance(extra_key_collect_set_colnames, Mapping):
        if isinstance(collect_set_colnames, Mapping):
            collect_set_colnames = join_mapping_by_values(
                collect_set_colnames, extra_key_collect_set_colnames,
                keep_original_value_for_mis_join=True
            )
        else:
            raise ValueError("'collect_set_colnames' must be a name mapping "
                             "when 'extra_key_collect_set_colnames' is a name mapping; "
                             f"got {collect_set_colnames}")

    if other_agg_args is not None and extra_key_other_agg_args is not None:
        _other_agg_kwargs = {}
        for k, v in other_agg_args.items():
            if isinstance(v, Mapping) and k in extra_key_other_agg_args:
                _other_agg_kwargs[k] = join_mapping_by_values(
                    v, extra_key_other_agg_args[k],
                    keep_original_value_for_mis_join=True
                )
            else:
                _other_agg_kwargs[k] = v
        other_agg_args = _other_agg_kwargs

    if isinstance(extra_key_count_col, str):
        count_colname_trg2 = count_colname_src2 = extra_key_count_col
    else:
        count_colname_trg2, count_colname_src2 = next(iter(extra_key_count_col.items()))
    if isinstance(count_col, str):
        count_colname_trg1 = count_colname_src1 = count_col
    else:
        count_colname_trg1, count_colname_src1 = next(iter(count_col.items()))

    if count_colname_trg1 == count_colname_trg2:
        raise ValueError(
            f"argument 'count_col' ({count_col}) conflicts with "
            f"argument 'count_col_for_extra_group_keys' ({extra_key_count_col})."
        )
    if isinstance(count_col, str) or count_colname_src1 == count_colname_src2:
        count_col = {count_colname_trg1: count_colname_trg2}
    count_colname = next(iter__(count_col))
    is_pairwise_aggregation = (
            bool(pairwise_group_key_suffix_first) and bool(pairwise_group_key_suffix_second)
    )

    if verbose:
        hprint_message(
            'df_data', df_data,
            'group_keys_ladder', group_keys_ladder,
            'count_col', count_col,
            'count_colname', count_colname,
            'avg_colnames', avg_colnames,
            'extra_key_avg_colnames', extra_key_avg_colnames,
            'collect_list_colnames', collect_list_colnames,
            'collect_list_max_size', collect_list_max_size,
            'extra_key_collect_list_colnames', extra_key_collect_list_colnames,
            'extra_key_collect_list_max_size', extra_key_collect_list_max_size,
            'collect_set_colnames', collect_set_colnames,
            'collect_set_max_size', collect_set_max_size,
            'extra_key_collect_set_colnames', extra_key_collect_set_colnames,
            'extra_key_collect_set_max_size', extra_key_collect_set_max_size,
            'other_agg_args', other_agg_args,
            'extra_key_other_agg_args', extra_key_other_agg_args,
            'pairwise_group_key_suffix_first', pairwise_group_key_suffix_first,
            'pairwise_group_key_suffix_second', pairwise_group_key_suffix_second,
            'is_pairwise_aggregation', is_pairwise_aggregation,
            'aggregate_transition_prob', aggregate_transition_prob,
            'transition_source_count_colname', transition_source_count_colname,
            'transition_prob_colname', transition_prob_colname,
            'keep_transition_count_cols', keep_transition_count_cols,
            title=aggregation_label
        )

    # endregion

    concat_list_colnames, concat_set_colnames = [], []
    for i, (
            group_keys,
            group_keys_with_extra_group_keys,
            df_leveled_aggs_for_extra_group_keys
    ) in enumerate(
        iter_leveled_aggregations(
            df_data=df_data,
            group_keys_ladder=group_keys_ladder,
            avg_cols=extra_key_avg_colnames,
            count_col=extra_key_count_col,
            weighted_aggregation_by_count_col=weighted_aggregation_by_count_col,
            collect_list_cols=extra_key_collect_list_colnames,
            collect_list_max_size=extra_key_collect_list_max_size,
            collect_set_cols=extra_key_collect_set_colnames,
            collect_set_max_size=extra_key_collect_set_max_size,
            extra_group_keys=extra_group_keys,
            unpersist_input_data=unpersist_input_data,
            aggregation_label=aggregation_label_for_extra_group_keys,
            other_agg_kwargs=extra_key_other_agg_args,
            aggregate_transition_prob=aggregate_transition_prob,
            aggregate_pair_ratio=aggregate_pair_ratio,
            pairwise_group_key_suffix_first=pairwise_group_key_suffix_first,
            pairwise_group_key_suffix_second=pairwise_group_key_suffix_second,
            transition_source_count_colname=transition_source_count_colname_for_extra_group_keys,
            transition_prob_colname=transition_prob_colname_for_extra_group_keys,
            pair_source_count_colname=transition_source_count_exclude_self_loop_colname_for_extra_group_keys,
            pair_ratio_colname=transition_prob_exclude_self_loop_colname_for_extra_group_keys,
            keep_transition_count_cols=keep_transition_count_cols,
            verbose=verbose
        )
    ):
        # import pdb;
        # pdb.set_trace()
        collect_list_colnames = _split_collect_list_or_set_cols(
            df_data,
            group_keys_with_extra_group_keys,
            collect_list_colnames,
            concat_list_colnames
        )

        collect_set_colnames = _split_collect_list_or_set_cols(
            df_data,
            group_keys_with_extra_group_keys,
            collect_set_colnames,
            concat_set_colnames
        )

        # import pdb;
        # pdb.set_trace()

        df_leveled_aggs = []
        for (
                group_keys_no_extra_group_keys, df_leveled_agg_for_extra_group_keys
        ) in zip(group_keys, df_leveled_aggs_for_extra_group_keys):
            if is_pairwise_aggregation:
                group_keys_no_extra_group_keys = _add_group_keys_suffix(
                    group_keys=group_keys_no_extra_group_keys,
                    pairwise_group_key_suffix_first=pairwise_group_key_suffix_first,
                    pairwise_group_key_suffix_second=pairwise_group_key_suffix_second
                )
            df_popularity = (
                df_leveled_agg_for_extra_group_keys.select(
                    *iter__(extra_group_keys, atom_types=(str, Column)),
                    *group_keys_no_extra_group_keys
                ).distinct().groupBy(
                    *group_keys_no_extra_group_keys
                ).agg(
                    F.count('*').alias(popularity_colname)
                )
            )

            df_leveled_agg = sparku.cache__(
                sparku.aggregate(
                    df_leveled_agg_for_extra_group_keys,
                    group_cols=group_keys_no_extra_group_keys,
                    # aggregations without extra group keys are performed
                    # on the already aggregated signal columns with the extra group keys,
                    # so it might need a name mapping
                    # if the aggregation column names
                    # are different from the aggregated columns with extra group keys
                    avg_cols=avg_colnames,
                    count_col=count_col,
                    avg_by_count_col=True,
                    weighted_avg_by_count_col=True,
                    collect_list_cols=collect_list_colnames,
                    collect_list_max_size=collect_list_max_size,
                    collect_set_cols=collect_set_colnames,
                    collect_set_max_size=collect_set_max_size,
                    concat_list_cols=concat_list_colnames,
                    concat_list_max_size=collect_list_max_size,
                    concat_set_cols=concat_set_colnames,
                    concat_set_max_size=collect_set_max_size,
                    ignore_agg_cols_of_conflict_names=True,
                    **(other_agg_args or {})
                ).join(df_popularity, group_keys_no_extra_group_keys),
                name=f'{aggregation_label or "leveled aggregation"} ({group_keys_no_extra_group_keys})',
            )

            if aggregate_transition_prob:
                df_leveled_agg = _add_trans_prob(
                    df_leveled_agg=df_leveled_agg,
                    aggregate_transition_prob=aggregate_transition_prob,
                    aggregate_pair_ratio=aggregate_pair_ratio,
                    transition_source_group_keys=first_half(group_keys_no_extra_group_keys),
                    transition_target_group_keys=second_half(group_keys_no_extra_group_keys),
                    extra_group_keys=None,
                    count_colname=count_colname,
                    transition_source_count_colname=transition_source_count_colname,
                    transition_prob_colname=transition_prob_colname,
                    pair_source_count_colname=transition_source_count_exclude_self_loop_colname,
                    pair_ratio_colname=transition_prob_exclude_self_loop_colname,
                    keep_transition_count_cols=keep_transition_count_cols
                )

                if verbose:
                    hprint_message(
                        'level', i,
                        'group_keys', group_keys_no_extra_group_keys,
                        'df_leveled_agg', df_leveled_agg,
                        'extra_group_keys', extra_group_keys,
                        title=f'{aggregation_label} ({group_keys_no_extra_group_keys})'
                    )
            df_leveled_aggs.append(df_leveled_agg)

        yield group_keys, df_leveled_aggs, df_leveled_aggs_for_extra_group_keys


def build_personalized_leveled_aggregations(
        df_input_data: DataFrame,
        stats_agg_args: LeveledStatsAggregationArgs,
        global_count_col: ColumnOrAliasedColumn,
        customer_count_col: ColumnOrAliasedColumn,
        stats_group_keys_ladder: Iterable[Iterable[Iterable[str]]],
        customer_id_colname: str,
        popularity_colname: str,
        raw_stats_are_sum: bool,
        unpersist_input_data: bool = True,
        verbose: bool = VERBOSE,
        # region supports pairwise transition prob
        pairwise_group_key_suffix_first: str = None,
        pairwise_group_key_suffix_second: str = None,
        aggregate_transition_prob: bool = False,
        aggregate_pair_ratio: bool = False,
        global_transition_source_count_colname: str = None,
        global_transition_prob_colname: str = None,
        customer_transition_source_count_colname: str = None,
        customer_transition_prob_colname: str = None,
        global_transition_source_count_exclude_self_loop_colname: str = None,
        global_transition_prob_exclude_self_loop_colname: str = None,
        customer_transition_source_count_exclude_self_loop_colname: str = None,
        customer_transition_prob_exclude_self_loop_colname: str = None,
        keep_transition_count_cols: bool = True,
        # endregion
):
    df_customer_level_aggs_collection = []
    df_global_level_aggs_collection = []
    group_keys_collection = []

    for group_keys, df_global_level_aggs, df_customer_level_aggs in iter_leveled_aggregations_(
            df_data=df_input_data,
            stats_agg_args=stats_agg_args,
            count_col=global_count_col,
            extra_key_count_col=customer_count_col,
            group_keys_ladder=stats_group_keys_ladder,
            extra_group_keys=customer_id_colname,
            popularity_colname=popularity_colname,
            weighted_aggregation_by_count_col=not raw_stats_are_sum,
            unpersist_input_data=unpersist_input_data,
            aggregation_label="global aggregation",
            aggregation_label_for_extra_group_keys="customer aggregation",
            pairwise_group_key_suffix_first=pairwise_group_key_suffix_first,
            pairwise_group_key_suffix_second=pairwise_group_key_suffix_second,
            aggregate_transition_prob=aggregate_transition_prob,
            aggregate_pair_ratio=aggregate_pair_ratio,
            transition_source_count_colname=global_transition_source_count_colname,
            transition_prob_colname=global_transition_prob_colname,
            transition_source_count_colname_for_extra_group_keys=customer_transition_source_count_colname,
            transition_prob_colname_for_extra_group_keys=customer_transition_prob_colname,
            transition_source_count_exclude_self_loop_colname=global_transition_source_count_exclude_self_loop_colname,
            transition_prob_exclude_self_loop_colname=global_transition_prob_exclude_self_loop_colname,
            transition_source_count_exclude_self_loop_colname_for_extra_group_keys=customer_transition_source_count_exclude_self_loop_colname,
            transition_prob_exclude_self_loop_colname_for_extra_group_keys=customer_transition_prob_exclude_self_loop_colname,
            keep_transition_count_cols=keep_transition_count_cols,
            verbose=verbose
    ):
        df_global_level_aggs_collection.extend(df_global_level_aggs)
        df_customer_level_aggs_collection.extend(df_customer_level_aggs)
        group_keys_collection.extend(group_keys)

    return (
        group_keys_collection,
        df_global_level_aggs_collection,
        df_customer_level_aggs_collection
    )


def iter_merge_leveled_aggregations(
        df_leveled_aggs: Iterable[DataFrame],
        group_keys_ladder: Iterable[Iterable[Iterable[str]]],
        stats_agg_args: LeveledStatsAggregationArgs,
        count_col: ColumnOrAliasedColumn,
        popularity_colname: str,
        # region optional args for aggregation with extra group keys
        df_leveled_aggs_for_extra_group_keys: Optional[Iterable[DataFrame]] = None,
        extra_group_keys: Optional[NamesOrColumns] = None,
        extra_key_count_col: Optional[ColumnOrAliasedColumn] = None,
        # endregion
        # region optional args for leveled aggregation
        aggregation_label: str = None,
        aggregation_label_for_extra_group_keys: str = None,
        # endregion
        # region supports pairwise aggregations & transition probilities
        pairwise_group_key_suffix_first: str = None,
        pairwise_group_key_suffix_second: str = None,
        aggregate_transition_prob: bool = False,
        aggregate_pair_ratio: bool = False,
        transition_source_count_colname: str = None,
        transition_prob_colname: str = None,
        transition_source_count_colname_for_extra_group_keys: str = None,
        transition_prob_colname_for_extra_group_keys: str = None,
        transition_source_count_exclude_self_loop_colname: str = None,
        transition_prob_exclude_self_loop_colname: str = None,
        transition_source_count_exclude_self_loop_colname_for_extra_group_keys: str = None,
        transition_prob_exclude_self_loop_colname_for_extra_group_keys: str = None,
        keep_transition_count_cols: bool = True,
        # endregion
        verbose: bool = VERBOSE,
):
    # region STEP1: resolves post-aggregation column names

    # This function is trying to merge multiple aggregation dataframes `df_leveled_aggs`
    # and `df_leveled_aggs_for_extra_group_keys` are already aggregated dataframes,
    # then we want to obtain the post-aggregation column names from the aggregation column args

    # for example, if `avg_colnames` is a mapping of
    # {'avg_colname1': 'source_colname1', 'avg_colname2': 'source_colname2'},
    # then we replace `avg_colnames` by the assumed post-aggregation column names
    # ['avg_colname1', 'avg_colname2']

    count_col = resolve_name_or_column(count_col)
    extra_key_count_col = resolve_name_or_column(extra_key_count_col)

    avg_colnames = resolve_names_or_columns(stats_agg_args.avg_colnames)
    extra_key_avg_colnames = resolve_names_or_columns(stats_agg_args.extra_key_avg_colnames)

    collect_list_colnames = resolve_names_or_columns(stats_agg_args.collect_list_colnames)
    extra_key_collect_list_colnames = resolve_names_or_columns(
        stats_agg_args.extra_key_collect_list_colnames
    )
    collect_list_max_size = stats_agg_args.collect_list_max_size
    extra_key_collect_list_max_size = stats_agg_args.extra_key_collect_list_max_size

    collect_set_colnames = resolve_names_or_columns(stats_agg_args.collect_set_colnames)
    extra_key_collect_set_colnames = resolve_names_or_columns(
        stats_agg_args.extra_key_collect_set_colnames
    )
    collect_set_max_size = stats_agg_args.collect_set_max_size
    extra_key_collect_set_max_size = stats_agg_args.extra_key_collect_set_max_size

    other_agg_args = stats_agg_args.other_agg_args
    if other_agg_args:
        other_agg_args = {
            k: (resolve_names_or_columns(v) if k.endswith('_cols') else v)
            for k, v in other_agg_args.items()
        }
    extra_key_other_agg_args = stats_agg_args.extra_key_other_agg_args
    if extra_key_other_agg_args:
        extra_key_other_agg_args = {
            k: (resolve_names_or_columns(v) if k.endswith('_cols') else v)
            for k, v in extra_key_other_agg_args.items()
        }
    # endregion

    # region STEP2: resolves other arguments

    # making `extra_group_keys` a list, e.g. 'customer_id' as ['customer_id']
    extra_group_keys = make_list(extra_group_keys, atom_types=(str, Column))

    # `is_pairwise_aggregation` is set True when both `pairwise_group_key_suffix_first` and
    # `pairwise_group_key_suffix_second` are specified
    is_pairwise_aggregation = (
            bool(pairwise_group_key_suffix_first) and
            bool(pairwise_group_key_suffix_second)
    )

    if verbose:
        hprint_message(
            'df_leveled_aggs', df_leveled_aggs,
            'df_leveled_aggs_for_extra_group_keys', df_leveled_aggs_for_extra_group_keys,
            'group_keys_ladder', group_keys_ladder,
            'count_col', count_col,
            'extra_key_count_col', extra_key_count_col,
            'avg_colnames', avg_colnames,
            'extra_key_avg_colnames', extra_key_avg_colnames,
            'collect_list_colnames', collect_list_colnames,
            'collect_list_max_size', collect_list_max_size,
            'extra_key_collect_list_colnames', extra_key_collect_list_colnames,
            'extra_key_collect_list_max_size', extra_key_collect_list_max_size,
            'collect_set_colnames', collect_set_colnames,
            'collect_set_max_size', collect_set_max_size,
            'extra_key_collect_set_colnames', extra_key_collect_set_colnames,
            'extra_key_collect_set_max_size', extra_key_collect_set_max_size,
            'other_agg_args', other_agg_args,
            'extra_key_other_agg_args', extra_key_other_agg_args,
            'pairwise_group_key_suffix_first', pairwise_group_key_suffix_first,
            'pairwise_group_key_suffix_second', pairwise_group_key_suffix_second,
            'is_pairwise_aggregation', is_pairwise_aggregation,
            'aggregate_transition_prob', aggregate_transition_prob,
            'transition_source_count_colname', transition_source_count_colname,
            'transition_prob_colname', transition_prob_colname,
            'keep_transition_count_cols', keep_transition_count_cols,
            title=aggregation_label
        )

    # endregion

    df_iter = zip__(df_leveled_aggs_for_extra_group_keys, df_leveled_aggs, atom_types=(DataFrame,))
    for i, group_keys in enumerate(group_keys_ladder):
        for _group_keys in group_keys:
            if is_pairwise_aggregation:
                _group_keys = _add_group_keys_suffix(
                    group_keys=_group_keys,
                    pairwise_group_key_suffix_first=pairwise_group_key_suffix_first,
                    pairwise_group_key_suffix_second=pairwise_group_key_suffix_second
                )
            df_leveled_agg_for_extra_group_keys, df_leveled_agg = next(df_iter)

            # region STEP3: merge leveled aggregations with extra keys
            if df_leveled_agg_for_extra_group_keys is not None:
                _group_keys_with_extra_group_keys = [*extra_group_keys, *_group_keys]
                df_leveled_agg_for_extra_group_keys = sparku.cache__(
                    sparku.aggregate(
                        df_leveled_agg_for_extra_group_keys,
                        group_cols=_group_keys_with_extra_group_keys,
                        avg_cols=extra_key_avg_colnames,
                        count_col=extra_key_count_col,
                        avg_by_count_col=True,
                        weighted_avg_by_count_col=True,
                        concat_list_cols=extra_key_collect_list_colnames,
                        concat_list_max_size=extra_key_collect_list_max_size,
                        concat_set_cols=extra_key_collect_set_colnames,
                        concat_set_max_size=extra_key_collect_set_max_size,
                        ignore_agg_cols_of_conflict_names=True,
                        **(extra_key_other_agg_args or {})
                    ),
                    name=f'{aggregation_label_for_extra_group_keys or "leveled aggregation"} '
                         f'({_group_keys_with_extra_group_keys})',
                    unpersist=df_leveled_agg_for_extra_group_keys
                )

                # region re-compute transition probabilities
                if aggregate_transition_prob:
                    df_leveled_agg_for_extra_group_keys = _add_trans_prob(
                        df_leveled_agg=df_leveled_agg_for_extra_group_keys,
                        aggregate_transition_prob=aggregate_transition_prob,
                        aggregate_pair_ratio=aggregate_pair_ratio,
                        transition_source_group_keys=first_half(_group_keys),
                        transition_target_group_keys=second_half(_group_keys),
                        extra_group_keys=extra_group_keys,
                        count_colname=extra_key_count_col,
                        transition_source_count_colname=transition_source_count_colname_for_extra_group_keys,
                        transition_prob_colname=transition_prob_colname_for_extra_group_keys,
                        pair_source_count_colname=transition_source_count_exclude_self_loop_colname_for_extra_group_keys,
                        pair_ratio_colname=transition_prob_exclude_self_loop_colname_for_extra_group_keys,
                        keep_transition_count_cols=keep_transition_count_cols
                    )
            # endregion

            # region STEP4: merge leveled aggregations without extra keys

            # re-compute popularity for the merged aggregation
            if df_leveled_agg is not None:
                df_popularity = (
                    df_leveled_agg_for_extra_group_keys.select(
                        *extra_group_keys, *_group_keys
                    ).distinct().groupBy(
                        *_group_keys
                    ).agg(
                        F.count('*').alias(popularity_colname)
                    )
                )

                df_leveled_agg = sparku.cache__(
                    sparku.aggregate(
                        df_leveled_agg,
                        group_cols=_group_keys,
                        avg_cols=avg_colnames,
                        count_col=count_col,
                        avg_by_count_col=True,
                        weighted_avg_by_count_col=True,
                        concat_list_cols=collect_list_colnames,
                        concat_list_max_size=collect_list_max_size,
                        concat_set_cols=collect_set_colnames,
                        concat_set_max_size=collect_set_max_size,
                        ignore_agg_cols_of_conflict_names=True,
                        **(other_agg_args or {})
                    ).join(df_popularity, _group_keys),
                    name=f'{aggregation_label or "leveled aggregation"} ({_group_keys})',
                )

                # re-compute transition probabilities
                if aggregate_transition_prob:
                    df_leveled_agg = _add_trans_prob(
                        df_leveled_agg=df_leveled_agg,
                        aggregate_transition_prob=aggregate_transition_prob,
                        aggregate_pair_ratio=aggregate_pair_ratio,
                        transition_source_group_keys=first_half(_group_keys),
                        transition_target_group_keys=second_half(_group_keys),
                        extra_group_keys=None,
                        count_colname=count_col,
                        transition_source_count_colname=transition_source_count_colname,
                        transition_prob_colname=transition_prob_colname,
                        pair_source_count_colname=transition_source_count_exclude_self_loop_colname,
                        pair_ratio_colname=transition_prob_exclude_self_loop_colname,
                        keep_transition_count_cols=keep_transition_count_cols
                    )

            # endregion
            yield df_leveled_agg, df_leveled_agg_for_extra_group_keys


def merge_personalized_leveled_aggregations(
        df_global_aggregations: Iterable[DataFrame],
        df_customer_aggregations: Iterable[DataFrame],
        stats_agg_args: LeveledStatsAggregationArgs,
        global_count_col: ColumnOrAliasedColumn,
        customer_count_col: ColumnOrAliasedColumn,
        stats_group_keys_ladder: Iterable[Iterable[Iterable[str]]],
        customer_id_colname: str,
        popularity_colname: str,
        verbose: bool = VERBOSE,
        # region supports pairwise transition prob
        pairwise_group_key_suffix_first: str = None,
        pairwise_group_key_suffix_second: str = None,
        aggregate_transition_prob: bool = False,
        aggregate_pair_ratio: bool = False,
        global_transition_source_count_colname: str = None,
        global_transition_prob_colname: str = None,
        customer_transition_source_count_colname: str = None,
        customer_transition_prob_colname: str = None,
        global_transition_source_count_exclude_self_loop_colname: str = None,
        global_transition_prob_exclude_self_loop_colname: str = None,
        customer_transition_source_count_exclude_self_loop_colname: str = None,
        customer_transition_prob_exclude_self_loop_colname: str = None,
        keep_transition_count_cols: bool = True,
        # endregion
):
    # region DEBUG
    # df_leveled_aggs = df_global_aggregations
    # df_leveled_aggs_for_extra_group_keys = df_customer_aggregations
    # stats_agg_args = stats_agg_args
    # count_col = global_count_col
    # extra_key_count_col = customer_count_col
    # group_keys_ladder = stats_group_keys_ladder
    # extra_group_keys = customer_id_colname
    # popularity_colname = popularity_colname
    # aggregation_label = "global aggregation"
    # aggregation_label_for_extra_group_keys = "customer aggregation"
    # pairwise_group_key_suffix_first = c.NAME_SUFFIX_FIRST
    # pairwise_group_key_suffix_second = c.NAME_SUFFIX_SECOND
    # aggregate_transition_prob = aggregate_transition_prob
    # aggregate_pair_ratio = aggregate_pair_ratio
    # transition_source_count_colname = global_transition_source_count_colname
    # transition_prob_colname = global_transition_prob_colname
    # transition_source_count_colname_for_extra_group_keys = customer_transition_source_count_colname
    # transition_prob_colname_for_extra_group_keys = customer_transition_prob_colname
    # transition_source_count_exclude_self_loop_colname = global_transition_source_count_exclude_self_loop_colname
    # transition_prob_exclude_self_loop_colname = global_transition_prob_exclude_self_loop_colname
    # transition_source_count_exclude_self_loop_colname_for_extra_group_keys = customer_transition_source_count_exclude_self_loop_colname
    # transition_prob_exclude_self_loop_colname_for_extra_group_keys = customer_transition_prob_exclude_self_loop_colname
    # keep_transition_count_cols = keep_transition_count_cols
    # verbose = verbose
    # endregio

    df_global_level_aggs, df_customer_level_aggs = zip(*iter_merge_leveled_aggregations(
        df_leveled_aggs=df_global_aggregations,
        df_leveled_aggs_for_extra_group_keys=df_customer_aggregations,
        stats_agg_args=stats_agg_args,
        count_col=global_count_col,
        extra_key_count_col=customer_count_col,
        group_keys_ladder=stats_group_keys_ladder,
        extra_group_keys=customer_id_colname,
        popularity_colname=popularity_colname,
        aggregation_label="global aggregation",
        aggregation_label_for_extra_group_keys="customer aggregation",
        pairwise_group_key_suffix_first=pairwise_group_key_suffix_first,
        pairwise_group_key_suffix_second=pairwise_group_key_suffix_second,
        aggregate_transition_prob=aggregate_transition_prob,
        aggregate_pair_ratio=aggregate_pair_ratio,
        transition_source_count_colname=global_transition_source_count_colname,
        transition_prob_colname=global_transition_prob_colname,
        transition_source_count_colname_for_extra_group_keys=customer_transition_source_count_colname,
        transition_prob_colname_for_extra_group_keys=customer_transition_prob_colname,
        transition_source_count_exclude_self_loop_colname=global_transition_source_count_exclude_self_loop_colname,
        transition_prob_exclude_self_loop_colname=global_transition_prob_exclude_self_loop_colname,
        transition_source_count_exclude_self_loop_colname_for_extra_group_keys=customer_transition_source_count_exclude_self_loop_colname,
        transition_prob_exclude_self_loop_colname_for_extra_group_keys=customer_transition_prob_exclude_self_loop_colname,
        keep_transition_count_cols=keep_transition_count_cols,
        verbose=verbose
    ))

    return df_global_level_aggs, df_customer_level_aggs
