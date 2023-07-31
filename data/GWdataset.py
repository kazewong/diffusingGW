from torch.utils.data import Dataset, DataLoader
import h5py

class GWdataset(Dataset):
    def __init__(self, path, transform=lambda x: x*1e20):
        self.data = h5py.File(path, 'r')['data']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample[None,:]

    def get_shape(self) -> tuple:
        return (1,)+self.data.shape[1:]

