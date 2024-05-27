from torch_geometric.data.data import Data


def is_new_data_object(data: Data):
    return '_store' in data.__dict__
