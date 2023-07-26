from torch.utils.data import Dataset, DataLoader
import h5py

class GWdataset(Dataset):
    def __init__(self, path, transform=None):
        self.data = h5py.File(path, 'r')['data']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample[:,None]

    def get_shape(self) -> tuple:
        return self.data.shape[1:]