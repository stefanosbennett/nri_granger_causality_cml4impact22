import torch
from torch.utils.data import Dataset
import os

#%% Multivariate time series dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, mode):
        """
        Load features (i.e. the MV time series dataset) and edges from saved files under data_path directory
        mode determines whether to load train, valid or test data
        """

        self.mode = mode
        self.data_path = data_path

        if self.mode == 'train':
            path = os.path.join(data_path, 'train_feats')
            edge_path = os.path.join(data_path, 'train_edges')
        elif self.mode == 'val':
            path = os.path.join(data_path, 'val_feats')
            edge_path = os.path.join(data_path, 'val_edges')
        elif self.mode == 'test':
            path = os.path.join(data_path, 'test_feats')
            edge_path = os.path.join(data_path, 'test_edges')

        if os.path.basename(os.path.normpath(data_path)) == 'var':
            ar_coef_path = os.path.join(data_path, 'ar_coef')
            self.ar_coef = torch.load(ar_coef_path)

        self.feats = torch.load(path)
        self.edges = torch.load(edge_path)

    def __getitem__(self, idx):
        return {'inputs': self.feats[idx], 'edges': self.edges[idx]}

    def __len__(self):
        return len(self.feats)
