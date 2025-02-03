from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional
from torchvision.transforms import v2


class ImageDataset:
    def __init__(
        self,
        data: Tensor,
        targets: Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        self.data = data.to(device=device)
        self.targets = targets.to(device=device)
        self.device = device

    @staticmethod
    def from_dataset(
        dataset: Dataset,
        device: Optional[torch.device] = None,
    ) -> ImageDataset:
        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)])

        imgs, targets = [], []
        for img, target in dataset:
            img = transform(img)
            imgs.append(img)
            targets.append(target)

        data = torch.stack(imgs).to(device)
        targets = torch.tensor(targets, device=device)

        return ImageDataset(data, targets, device=device)

    def to(self, device: torch.device) -> ImageDataset:
        self.data = self.data.to(device=device)
        self.targets = self.targets.to(device=device)
        return self

    def subset(self, indices: list[int]) -> ImageDataset:
        data, targets = self.data[indices], self.targets[indices]
        return ImageDataset(data, targets, device=self.device)

    def slice(self, start: int, end: int) -> tuple[Tensor, Tensor]:
        imgs, targets = self.data[start:end], self.targets[start:end]
        return imgs, targets

    def transform_(self, transform) -> ImageDataset:
        self.data = transform(self.data)
        return self

    def __len__(self) -> int:
        return len(self.data)


class ImageDataLoader:
    def __init__(
        self,
        dataset: ImageDataset,
        batch_size: int,
        shuffle: bool = False,
        channels_last: bool = False,
        transform=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.channels_last = channels_last
        self.transform = transform

        self._idx = 0

    def __len__(self) -> int:
        left = len(self.dataset) - self._idx
        batches = left // self.batch_size
        if left % self.batch_size != 0:
            batches += 1
        return batches

    def __iter__(self):
        self._idx = 0

        if self.shuffle:
            with torch.no_grad():
                idx = torch.randperm(len(self.dataset.data), device=self.dataset.device)
                self.dataset.data, self.dataset.targets = (
                    self.dataset.data[idx],
                    self.dataset.targets[idx],
                )

        return self

    def __next__(self):
        if self._idx >= len(self.dataset):
            raise StopIteration

        imgs, targets = self.dataset.slice(self._idx, self._idx + self.batch_size)
        self._idx += self.batch_size

        if self.transform is not None:
            with torch.no_grad():
                imgs = self.transform(imgs)

        if self.channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)

        return imgs, targets
