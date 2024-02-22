from kazeML.jax.diffusion.diffusion_dataset import DiffusionDataset
import h5py


class GWdataset(DiffusionDataset):
    def __init__(self, path, transform=lambda x: x * 1e20, has_conditional=False):
        super().__init__(path, transform=transform, has_conditional=has_conditional)
        self.data = h5py.File(path, "r")["data"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
