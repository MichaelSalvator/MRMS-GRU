from torch.utils.data import Dataset
import numpy as np
import torch

class RemoteSensingDataset(Dataset):
    def __init__(self, data_file, label_file):
        data = np.load(data_file)
        self.data = data.transpose(2, 1, 0)
        self.labels = np.load(label_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if sample.ndim == 2:  # 如果是(band, time)，不需要额外处理
            return torch.FloatTensor(sample), torch.LongTensor([self.labels[idx]])
        else:
            raise ValueError("Unexpected data dimensions")