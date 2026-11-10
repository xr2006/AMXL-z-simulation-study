import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

J = 3
ATTR_ORDER = ['CO', 'TT', 'POI']     # rows of x, in the order models.Utility expects


def load_split(csv_path, t_min=1, t_max=None):
    """Choice occasions t_min <= t <= t_max of a panel CSV -> x (n, K, J), raw z = (INCOME, TOD, DUR) (n, 3)
    and y (n,). Each occasion is one observation: TasteNet has no driver-level random effect."""
    wide = pd.read_csv(csv_path)
    keep = wide['t'] >= t_min
    if t_max is not None:
        keep &= wide['t'] <= t_max
    wide = wide[keep]
    x = np.stack([np.stack([wide[f'ALT{j}_{a}'].values for j in range(J)], axis=1) for a in ATTR_ORDER], axis=1)
    return {'x': x, 'z': wide[['INCOME', 'TOD', 'DUR']].values.astype(float), 'y': wide['CHOICE'].values.astype(int)}


class ChoiceDataset(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data['x'], dtype=torch.float32)    # (N, K, J)
        self.z = torch.tensor(data['z'], dtype=torch.float32)    # (N, D)
        self.y = torch.tensor(data['y'], dtype=torch.long)       # (N,)
        self.N, self.K, self.J = self.x.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx], 'z': self.z[idx]}
