import numpy as np
import torch
from torch.utils.data import Dataset


class GameDataset(Dataset):
    """Bimatrix game dataset"""

    def __init__(self, data, labels, seed, device):
        """
        Args:
            data: game payoffs; (num_samples, channels, row, col)
            labels: frequency of actions; (num_samples, freq)
            seed: random seed
            device: cuda device
        """
        self.samples = []
        self.data = data
        self.labels = labels
        self.device = device
        self.load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, ix):
        return self.samples[ix]

    def load_samples(self):
        for ix in range(self.data.shape[0]):
            self.samples.append((
                torch.tensor(self.data[ix], dtype=torch.float).to(self.device),
                torch.tensor(self.labels[ix], dtype=torch.float).to(self.device)))


def load_game_data(data, labels, batch_size, seed, device, shuffle=True):
    torch.manual_seed(seed)
    dataset = GameDataset(data, labels, seed, device)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
