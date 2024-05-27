from os import path
from typing import List, Union

from attr import attrs, attrib

from boba_python_utils.common_utils.iter_helper import iter__
from boba_python_utils.io_utils.text_io import read_all_lines


@attrs(slots=True)
class FeatureSelection:
    name = attrib(type=str)
    selectors = attrib(type=Union[str, List])

    FeatureSelectionNameAllFeatures = 'all_feats'

    @classmethod
    def all_features(cls):
        return cls(
            name=cls.FeatureSelectionNameAllFeatures,
            selectors=cls.FeatureSelectionNameAllFeatures
        )

    def select_features_from_list(self, feature_list):
        out = []
        for feature_selector in iter__(self.selectors):
            if feature_selector == self.FeatureSelectionNameAllFeatures:
                return feature_list
            selected_feats = _select_features(feature_list, feature_selector)
            if selected_feats is None:
                raise ValueError(f"'{feature_selector}' is not a supported feature selector")
            out.extend(selected_feats)

        has_mask_values = False
        for x in out:
            if isinstance(x, (tuple, list)):
                has_mask_values = True
                break
        if has_mask_values:
            out_with_mask_vals = {}
            for x in out:
                if isinstance(x, (tuple, list)):
                    out_with_mask_vals[x[0]] = x[1]
                else:
                    out_with_mask_vals[x] = None
            return out_with_mask_vals
        else:
            return out


# @attrs(slots=True)
# class TreeModelTopImportanceFeatureSelector:
#     model_path = attrib(type=str)
#     top = attrib(type=float, default=0.3, converter=float)
#
#     def get_feature_names(self, feature_names=None):
#         model: statu.XgBoostSklearnWrapper = statu.load_model(self.model_path)
#         if feature_names is None:
#             feature_names = model.feature_names
#         else:
#             if not model.feature_names[0].startswith('Column_') and model.feature_names != feature_names:
#                 raise ValueError('feature mismatch')
#
#         num_feats = len(model.feature_importances_)
#         top_feat_indexes = sorted__(
#             range(num_feats),
#             key=model.feature_importances_,
#             reverse=True
#         )[:int(self.top * num_feats)]
#         return [feature_names[i] for i in top_feat_indexes]


def _select_features(feature_list, feature_selector):
    if callable(feature_selector):
        # the function processes each feature name in the feature list;
        # it can return True to indicate the feature is selected;
        # it can return False or None to indicate the 
        out = []
        for feature in feature_list:
            x = feature_selector(feature)
            if x is True:
                out.append(feature)
            elif x is not None and x is not False:
                if isinstance(x, (tuple, list)) and x[0] is True:
                    out.append((feature, *x[1:]))
                else:
                    out.append(x)
        return out
    elif isinstance(feature_selector, list):
        out = []
        for _feature_selector in feature_selector:
            for feature_name in feature_list:
                if feature_name.startswith(_feature_selector) and feature_name not in out:
                    out.append(feature_name)
        return out
    elif isinstance(feature_selector, str):
        if path.exists(feature_selector):
            feature_selector: List[str] = read_all_lines(feature_selector)
            return list(filter(lambda x: x in feature_list, feature_selector))
        # elif ':' in feature_selector:
        #     feature_selector: TreeModelTopImportanceFeatureSelector = str2obj(feature_selector, TreeModelTopImportanceFeatureSelector)
        #     return feature_selector.get_feature_names(feature_names=feature_list)
