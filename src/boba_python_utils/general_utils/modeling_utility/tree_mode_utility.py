import xgboost as xgb
import lightgbm as lgb
from os import path
import numpy as np
import warnings
from typing import Dict

from boba_python_utils.common_utils.iter_helper import zip__
from boba_python_utils.general_utils.console_util import hprint_message
from boba_python_utils.io_utils.csv_io import iter_csv, write_csv
from boba_python_utils.io_utils.pickle_io import pickle_save
from boba_python_utils.path_utility.path_string_operations import replace_ext_name
from boba_python_utils.time_utils.tictoc import tic

XGBOOST_MODEL_TAG = 'xgb'
LGBOOST_MODEL_TAG = 'lgb'


class TreeModel:
    def __init__(
            self,
            lib=XGBOOST_MODEL_TAG,
            params=None,
            max_rounds=3000,
            early_stopping_rounds=100,
            num_parallel_tree=100,
            max_depth=None,
            rank_eval_metric='ndcg@10',
            gpu=None,
            model_path=None,
            importance_type=None,
            **kwargs
    ):
        self._lib = lib
        if lib == XGBOOST_MODEL_TAG:
            max_depth = max_depth or 10
            self.importance_type = importance_type or 'gain'
            if isinstance(params, str):
                if params == 'rank':
                    self._params = {'objective': 'rank:pairwise', 'learning_rate': 0.2, 'min_split_loss': 1.0, 'min_child_weight': 0.1, 'max_depth': max_depth, 'eval_metric': rank_eval_metric}
                elif params == 'rank_ndcg':
                    self._params = {'objective': 'rank:ndcg', 'learning_rate': 0.2, 'min_split_loss': 1.0, 'min_child_weight': 0.1, 'max_depth': max_depth, 'eval_metric': rank_eval_metric}
                elif params == 'rank_logistic':
                    self._params = {'objective': 'binary:logistic', 'learning_rate': 0.2, 'gamma': 1.5, 'min_child_weight': 1.5, 'max_depth': max_depth, 'eval_metric': rank_eval_metric}
                elif params == 'rank_rf':
                    self._params = {'objective': 'binary:logistic', 'learning_rate': 0.2, 'gamma': 1.5, 'min_child_weight': 1.5, 'max_depth': max_depth, 'num_parallel_tree': num_parallel_tree, 'eval_metric': rank_eval_metric}
            elif params is None:
                self._params = params if params else {'objective': 'binary:logistic', 'learning_rate': 0.2, 'gamma': 1.5, 'min_child_weight': 1.5, 'max_depth': max_depth}
            else:
                self._params = params
            if gpu is not None and isinstance(gpu, int) and gpu >= 0:
                self._params['gpu_id'] = gpu
                self._params['tree_method'] = 'gpu_hist'
        elif lib == LGBOOST_MODEL_TAG:
            self.importance_type = importance_type or 'split'
            max_depth = max_depth or -1
            self._params = params if params else {'objective': 'binary', 'learning_rate': 0.1, 'min_split_gain': 0, 'min_child_weight': 1e-3, 'max_depth': max_depth}
            if gpu is not None:
                if isinstance(gpu, int):
                    self._params['gpu_platform_id'], self._params['gpu_device_id'] = 0, gpu
                else:
                    self._params['gpu_platform_id'], self._params['gpu_device_id'] = gpu
                self._params['device_type'] = 'gpu'

        if kwargs:
            self._params.update(kwargs)

        self._max_rounds = max_rounds
        self._model = None
        self._early_stopping_rounds = early_stopping_rounds
        self.feature_importances_ = []
        self.feature_names = []
        if model_path is not None:
            self.load_model(model_path)

    @staticmethod
    def _sanitize_inpput(_X, _y):
        if isinstance(_X, (list, tuple)):
            _X = np.array(_X)
        if isinstance(_y, (list, tuple)):
            _y = np.array(_y)
            _y = np.array(_y)
        return _X, _y

    @staticmethod
    def _set_group(data, _group):
        if _group is not None:
            if isinstance(_group[0], tuple):
                _group = [x[1] - x[0] for x in _group]
            data.set_group(_group)

    def fit(self, X, y, group=None, max_rounds=None, evals: Dict = None, early_stopping_rounds=None, verbose_eval=True, feature_name=None):
        if max_rounds is None:
            max_rounds = self._max_rounds
        if early_stopping_rounds is None:
            early_stopping_rounds = self._early_stopping_rounds
        elif early_stopping_rounds <= 0:
            early_stopping_rounds = None

        if self._lib == XGBOOST_MODEL_TAG:
            def _proc_input(_X, _y, _group=None):
                _X, _y = self._sanitize_inpput(_X, _y)
                data = xgb.DMatrix(_X, label=_y)
                self._set_group(data, _group)
                return data
        elif self._lib == LGBOOST_MODEL_TAG:
            def _proc_input(_X, _y, _group=None):
                _X, _y = self._sanitize_inpput(_X, _y)
                if feature_name is None:
                    data = lgb.Dataset(_X, label=_y)
                else:
                    data = lgb.Dataset(_X, label=_y, feature_name=feature_name)
                self._set_group(data, _group)
                return data
        train_data = _proc_input(X, y, group)
        if evals is not None:
            evals_has_train = 'run' in evals
            evals = [((_proc_input(*v) if isinstance(v, (list, tuple)) else v), k) for k, v in evals.items()]
            if not evals_has_train:
                evals = [(train_data, 'run')] + evals
        else:
            evals = [(train_data, 'run')]

        if self._lib == XGBOOST_MODEL_TAG:
            self._model = xgb.train(
                params=self._params,
                dtrain=train_data,
                num_boost_round=max_rounds,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval
            )
        elif self._lib == LGBOOST_MODEL_TAG:
            valid_sets, valid_names = zip(*evals)
            self._model = lgb.train(
                params=self._params,
                train_set=train_data,
                num_boost_round=max_rounds,
                valid_sets=valid_sets,
                valid_names=valid_names,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval
            )
            self.feature_importances_ = self._model.feature_importance(importance_type=self.importance_type)

    def predict_proba(self, data, group=None):
        # import pdb; pdb.set_trace()
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        if self._lib == XGBOOST_MODEL_TAG:
            data = xgb.DMatrix(data)
            self._set_group(data, group)
        # light GBM does not support setting groups for prediction
        scores = self._model.predict(data)
        if len(scores.shape) == 2 and scores.shape[1] == 0:
            raise ValueError('scores returned by xgboost are not in good shape; make sure the xgboost version in use matches the saved model')
        return scores

    @staticmethod
    def _get_model_file_paths(model_path):
        return path.join(model_path, f'model.bin'), path.join(model_path, f'model_features.txt')

    def save_model(self, model_path):
        model_file_path, model_feature_list_path = self._get_model_file_paths(model_path)

        hprint_message('saving model to', model_file_path)
        xgb_model = self._model
        xgb_model.save_model(model_file_path)
        pickle_save(xgb_model, replace_ext_name(model_file_path, 'pkl'))

        if self._lib == XGBOOST_MODEL_TAG:
            feature_importance = self._model.get_score(importance_type=self.importance_type)
            feature_names = xgb_model.feature_names
            feature_types = xgb_model.feature_types
        else:
            feature_names = xgb_model.feature_name()
            feature_importance = self._model.feature_importance(importance_type=self.importance_type)
            len_feature_names, len_feature_importance = len(feature_names), len(feature_importance)
            if len_feature_names != len_feature_importance:
                raise ValueError(f"the number of feature names ({len_feature_names}) does not match the number of feature importance values (f{len_feature_importance})")
            feature_importance = dict(zip(feature_names, feature_importance))
            feature_types = ['None'] * len_feature_names

        write_csv(
            ((name, feat_type, feature_importance.get(name, None)) for name, feat_type in zip__(feature_names, feature_types)),
            output_csv_path=model_feature_list_path,
            header=('name', 'type', 'importance')
        )

    def load_model(self, model_path):
        model_file_path, model_feature_list_path = self._get_model_file_paths(model_path)
        feat_names, feat_types, feature_importance = [], [], []
        if path.exists(model_feature_list_path):
            for feat_name, feat_type, importance in iter_csv(model_feature_list_path):
                if feat_name == 'None':
                    warnings.warn('invalid feature name as `None`')
                    feat_names = None
                    break

                feat_names.append(feat_name)
                feature_importance.append(float(importance) if importance != 'None' else 0)
                if feat_type != 'None':
                    feat_types.append(feat_type)

        self.feature_importances_ = feature_importance
        self.feature_names = feat_names

        tic(f'Loading {self._lib} model from path {model_file_path}.')

        if self._lib == XGBOOST_MODEL_TAG:
            self._model = xgb_model = xgb.Booster()
            xgb_model.load_model(model_file_path)

            if self.feature_names:
                xgb_model.feature_names = self.feature_names
                if feat_types:
                    if len(feat_types) == len(feat_names):
                        xgb_model.feature_types = feat_types
                    else:
                        warnings.warn(f'expected {len(feat_types)} feature types; got {len(feat_names)}')
        else:
            self._model = xgb_model = lgb.Booster(model_file=model_file_path)

            feat_names = xgb_model.feature_name()
            feature_importance = xgb_model.feature_importance(importance_type=self.importance_type)
            if isinstance(feature_importance, np.ndarray):
                feature_importance = feature_importance.tolist()
            if self.feature_names:
                if feat_names != self.feature_names:
                    raise ValueError(f'feature name mismatch for loaded model; expected {self.feature_names}; got {feat_names}')
            else:
                self.feature_names = feat_names
            if self.feature_importances_:
                if feature_importance != self.feature_importances_:
                    raise ValueError(f'feature importance mismatch for loaded model; expected {self.feature_importances_}; got {feature_importance}')
            else:
                self.feature_importances_ = feature_importance

    def plot_tree(self, **kwargs):
        if self._lib == XGBOOST_MODEL_TAG:
            pass
        elif self._lib == LGBOOST_MODEL_TAG:
            return lgb.plot_tree(self._model, **kwargs)

