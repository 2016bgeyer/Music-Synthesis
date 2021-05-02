import numpy as np
from data_loader.dataset import MidiDataset
from data_loader.collate import midi_collate_fn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class MidiDataLoader(DataLoader):
    """
    Midi Dataloader class
    """
    def __init__(self, data_path, batch_size, shuffle, validation_split, num_workers, **kwargs):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.dataset = MidiDataset(data_path)

        self.batch_idx = 0
        self.n_samples = len(self.dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'collate_fn': midi_collate_fn
            }
        super(MidiDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None       # test_data_loader in config uses a split of 0.0

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)