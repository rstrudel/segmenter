from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import segm.utils.torch as ptu


class Loader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, distributed, split):
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=True)
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
            )
        else:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )

        self.base_dataset = self.dataset

    @property
    def unwrapped(self):
        return self.base_dataset.unwrapped

    def set_epoch(self, epoch):
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

    def get_diagnostics(self, logger):
        return self.base_dataset.get_diagnostics(logger)

    def get_snapshot(self):
        return self.base_dataset.get_snapshot()

    def end_epoch(self, epoch):
        return self.base_dataset.end_epoch(epoch)
