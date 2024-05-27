from boba_python_utils.io_utils.text_io import iter_all_lines



def read_feature_list(file, check_feature_index=True, **kwargs):
    line_idx = 0
    out = []
    for line in iter_all_lines(file, **kwargs):
        line = line.strip()
        if line:
            if ' ' in line:
                # supports indexed feature names, e.g. "f37 bm25_defect", or "37 bm25_defect"
                feat_idx, feat_name = line.split(maxsplit=1)
                feat_idx = int(feat_idx[1:] if feat_idx[0] == 'f' else feat_idx)
                if check_feature_index and feat_idx != line_idx:
                    raise ValueError(f"for feature {feat_name}, expected feature index {line_idx}, got {feat_idx}")

                if not feat_name:
                    raise ValueError(f"feature of index {feat_idx} has empty name")
            else:
                # unindexed feature names
                feat_name = line

            feat_name = feat_name.strip('\'",;')
            feat_name = feat_name.strip()
            if feat_name in out:
                raise ValueError(f"duplicate feature name {feat_name}")
            out.append(feat_name)
            line_idx += 1
    return out
