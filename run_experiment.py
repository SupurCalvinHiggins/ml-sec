import argparse
import math
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch import nn
from torch import optim
from torch import Tensor
import torch
import torchvision
from torchvision.transforms import v2
import itertools as it
import json
import shutil
import copy
import numpy as np
import random
import time
import contextlib
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=Path)
    parser.add_argument("--clean", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profile", default=False, action="store_true")
    return parser


class ImageDataset(Dataset):
    def __init__(self, data, targets, device=None):
        self.data = data.to(device)
        self.targets = targets.to(device)
        self.device = device

    def subset(self, indices: list[int]):
        data, targets = self.data[indices], self.targets[indices]
        return ImageDataset(data, targets, device=self.device)

    def slice(self, start: int, end: int):
        imgs, targets = self.data[start:end], self.targets[start:end]
        return imgs, targets

    def transform_(self, transform):
        self.data, self.targets = transform(self.data, self.targets)
        return self

    def transform_data_(self, transform):
        self.data = transform(self.data)
        return self

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self) -> int:
        return len(self.data)


class ImageDataLoader:
    def __init__(self, dataset: ImageDataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._idx = 0

    def __iter__(self):
        self._idx = 0

        if self.shuffle:

            @torch.no_grad()
            def shuffle_transform(data, targets):
                idx = torch.randperm(len(data), device=data.device)
                return data[idx], targets[idx]

            self.dataset.transform_(shuffle_transform)

        return self

    def __next__(self):
        if self._idx >= len(self.dataset):
            raise StopIteration
        imgs, targets = self.dataset.slice(self._idx, self._idx + self.batch_size)
        self._idx += self.batch_size
        return imgs, targets


def make_transforms(dataset_name: str):
    dataset_name_to_mean = {
        "cifar-10": (0.4914, 0.4822, 0.4465),
        "cifar-100": (0.5071, 0.4865, 0.4409),
    }
    dataset_name_to_std = {
        "cifar-10": (0.2471, 0.2435, 0.2616),
        "cifar-100": (0.2673, 0.2564, 0.2762),
    }

    test_transform = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                dataset_name_to_mean[dataset_name],
                dataset_name_to_std[dataset_name],
            ),
        ]
    )
    train_transform = v2.Compose(
        [
            v2.RandomCrop(32, padding=4, padding_mode="reflect"),
            v2.RandomHorizontalFlip(),
            test_transform,
        ]
    )

    return train_transform, test_transform


def make_datasets(dataset_name: str) -> tuple[ImageDataset, ImageDataset]:
    dataset_name_to_cls = {"cifar-10": datasets.CIFAR10, "cifar-100": datasets.CIFAR100}
    cls = dataset_name_to_cls[dataset_name]
    train_dataset = cls(root="./data", train=True, download=True)
    test_dataset = cls(root="./data", train=False, download=False)

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)])

    train_data, train_targets = zip(
        *[(transform(img), target) for img, target in train_dataset]
    )
    test_data, test_targets = zip(
        *[(transform(img), target) for img, target in test_dataset]
    )

    train_data, train_targets = torch.stack(train_data), torch.tensor(train_targets)
    test_data, test_targets = torch.stack(test_data), torch.tensor(test_targets)

    train_dataset = ImageDataset(train_data, train_targets, device=device)
    test_dataset = ImageDataset(test_data, test_targets, device=device)

    return train_dataset, test_dataset


def make_model(model_hp: dict) -> nn.Module:
    model_hp = copy.deepcopy(model_hp)
    name = model_hp.pop("name")
    name_to_cls = {"resnet-18": torchvision.models.resnet18}
    cls = name_to_cls[name]
    return cls(**model_hp)


def make_optimizer(model: nn.Module, optimizer_hp: dict) -> optim.Optimizer:
    optimizer_hp = copy.deepcopy(optimizer_hp)
    name = optimizer_hp.pop("name")
    name_to_cls = {
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
    }
    cls = name_to_cls[name]
    return cls(model.parameters(), **optimizer_hp)


def make_criterion(dataset_name: str) -> nn.Module:
    dataset_name_to_cls = {
        "cifar-10": nn.CrossEntropyLoss,
        "cifar-100": nn.CrossEntropyLoss,
    }
    cls = dataset_name_to_cls[dataset_name]
    return cls()


@torch.enable_grad()
def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
    transform=None,
    debug: bool = False,
    profile: bool = False,
) -> tuple[float, float]:
    model.train()

    total_loss = torch.tensor(0.0, device=device, requires_grad=False)
    total_correct = torch.tensor(0, device=device, requires_grad=False)
    total_samples = 0

    scaler = torch.amp.GradScaler("cuda", enabled=not debug)

    prof = (
        torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=5, warmup=5, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/resnet18"),
            with_stack=True,
        )
        if profile
        else contextlib.nullcontext()
    )

    with prof:
        for x, y in loader:
            if profile:
                prof.step()

            if transform is not None:
                with torch.no_grad():
                    x = transform(x)
            x = x.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=not debug
            ):
                pred_y = model(x)
                loss = criterion(pred_y, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss = loss.detach()
            pred_y = pred_y.detach()

            total_loss += loss
            total_correct += (pred_y.argmax(dim=1) == y).sum()
            total_samples += y.size(0)

    total_loss = total_loss.item()
    total_correct = total_correct.item()

    acc = total_correct / total_samples
    avg_loss = total_loss / total_samples

    return avg_loss, acc


@torch.enable_grad()
def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
    max_epochs: int,
    transform=None,
    debug: bool = False,
    profile: bool = False,
) -> tuple[list[float], list[float]]:
    model.train()
    avg_losses, accs = [], []
    for _ in range(max_epochs):
        avg_loss, acc = train_epoch(
            model,
            optimizer,
            criterion,
            loader,
            transform=transform,
            debug=debug,
            profile=profile,
        )
        avg_losses.append(avg_loss)
        accs.append(acc)
    return avg_losses, accs


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    debug: bool = False,
) -> tuple[float, float]:
    model.eval()

    total_loss = torch.tensor(0.0, device=device, requires_grad=False)
    total_correct = torch.tensor(0, device=device, requires_grad=False)
    total_samples = 0

    for x, y in loader:
        x = x.to(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not debug):
            pred_y = model(x)
            loss = criterion(pred_y, y)

        loss = loss.detach()
        pred_y = pred_y.detach()

        total_loss += loss
        total_correct += (pred_y.argmax(dim=1) == y).sum()
        total_samples += y.size(0)

    total_loss = total_loss.item()
    total_correct = total_correct.item()

    acc = total_correct / total_samples
    avg_loss = total_loss / total_samples

    return avg_loss, acc


@torch.no_grad()
def eval(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    debug: bool = False,
) -> tuple[float, float]:
    model.eval()
    return eval_epoch(model, criterion, loader, debug=debug)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        k = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, k, sep=sep).items())
        else:
            items.append((k, v))
    return dict(items)


def unflatten_dict(d: dict, sep=".") -> dict:
    sweep = {}
    for k, v in d.items():
        keys = k.split(sep)
        d = sweep
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return sweep


def make_combinations(hps: dict) -> list[dict]:
    flat_hps = flatten_dict(hps)
    keys, values = zip(*flat_hps.items())
    return [
        unflatten_dict(dict(zip(keys, combination)))
        for combination in it.product(*values)
    ]


def set_seed(x: int) -> None:
    torch.manual_seed(x)
    random.seed(x)
    np.random.seed(x)


@torch.no_grad()
def weight_reset(module: nn.Module) -> None:
    reset_parameters = getattr(module, "reset_parameters", None)
    if callable(reset_parameters):
        module.reset_parameters()


def cv_main(
    cfg: dict, seed: int, debug: bool = False, profile: bool = False
) -> tuple[nn.Module, dict]:
    set_seed(seed)

    dataset, test_dataset = make_datasets(dataset_name=cfg["dataset_name"])
    train_transform, test_transform = make_transforms(dataset_name=cfg["dataset_name"])
    criterion = make_criterion(dataset_name=cfg["dataset_name"]).to(device)

    kfold = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True)
    idx = list(range(len(dataset)))
    folds = list(kfold.split(idx, dataset.targets.cpu()))

    torch.backends.cudnn.benchmark = True

    hps = make_combinations(cfg["hps"])
    sweep = {"hps": [], "best_hp": {}}
    best_hp_idx = 0
    for hp_idx, hp in enumerate(hps):
        print(f"[Model {hp_idx + 1}/{len(hps)}] hp = {hp}")
        sweep["hps"].append(
            {
                "hp": hp,
                "train_avg_losses": [],
                "train_accs": [],
                "val_loss": [],
                "val_acc": [],
            }
        )

        model = make_model(model_hp=hp["model"]).to(
            device, memory_format=torch.channels_last
        )
        torch.compile(model, mode="max-autotune")
        optimizer = make_optimizer(model, optimizer_hp=hp["optimizer"])

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            model.apply(weight_reset)

            train_dataset = dataset.subset(train_idx)
            train_loader = ImageDataLoader(
                train_dataset,
                batch_size=hp["batch_size"],
                shuffle=True,
            )

            val_dataset = dataset.subset(val_idx).transform_data_(test_transform)
            val_loader = DataLoader(val_dataset, batch_size=hp["batch_size"])

            train_start = time.time()
            train_avg_losses, train_accs = train(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                loader=train_loader,
                transform=train_transform,
                max_epochs=hp["max_epochs"],
                debug=debug,
                profile=profile,
            )
            train_end = time.time()
            train_time = train_end - train_start
            val_loss, val_acc = eval(
                model=model,
                criterion=criterion,
                loader=val_loader,
                debug=debug,
            )
            print(
                f"\t[Fold {fold_idx + 1}/{len(folds)}] {val_loss = :.4f}, {val_acc = :.4f}, {train_time = :.2f}"
            )
            print(
                f"\t[Fold {fold_idx + 1}/{len(folds)}] {train_avg_losses=}, {train_accs=}"
            )

            sweep["hps"][-1]["train_avg_losses"].append(train_avg_losses)
            sweep["hps"][-1]["train_accs"].append(train_accs)
            sweep["hps"][-1]["val_loss"].append(val_loss)
            sweep["hps"][-1]["val_acc"].append(val_acc)

        avg_val_acc = np.mean(sweep["hps"][-1]["val_acc"])
        best_avg_val_acc = np.mean(sweep["hps"][best_hp_idx]["val_acc"])
        if avg_val_acc > best_avg_val_acc:
            best_hp_idx = hp_idx

    best_hp = sweep["hps"][best_hp_idx]["hp"]

    train_dataset = dataset
    train_loader = ImageDataLoader(
        train_dataset, batch_size=best_hp["batch_size"], shuffle=True
    )

    test_dataset = test_dataset.transform_data_(test_transform)
    test_loader = DataLoader(test_dataset, batch_size=best_hp["batch_size"])

    model = make_model(model_hp=best_hp["model"]).to(device)
    optimizer = make_optimizer(model, optimizer_hp=best_hp["optimizer"])

    train_avg_losses, train_accs = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        loader=train_loader,
        max_epochs=best_hp["max_epochs"],
        transform=train_transform,
        debug=debug,
        profile=profile,
    )
    test_loss, test_acc = eval(
        model=model, criterion=criterion, loader=test_loader, debug=debug
    )
    print(
        f"[Best Model] hp = {best_hp}, test_loss = {test_loss}, test_acc = {test_acc}"
    )

    sweep["best_hp"] = {
        "best_hp_idx": best_hp_idx,
        "hp": sweep["hps"][best_hp_idx]["hp"],
        "train_avg_losses": train_avg_losses,
        "train_accs": train_accs,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    return model, sweep


def main():
    parser = make_parser()
    args = parser.parse_args()
    assert args.cfg_path.exists()
    result_path = args.cfg_path.parent / args.cfg_path.stem
    if not args.clean and result_path.exists():
        return

    if args.clean and result_path.exists():
        shutil.rmtree(result_path)

    result_path.mkdir()

    cfg = json.loads(args.cfg_path.read_text())
    seeds = cfg.pop("seeds")
    for seed in seeds:
        model, sweep = cv_main(cfg, seed=seed, debug=args.debug, profile=args.profile)

        model_path = result_path / f"model-{seed}.pth"
        sweep_path = result_path / f"sweep-{seed}.json"

        torch.save(model.state_dict(), model_path)
        sweep_path.write_text(json.dumps(sweep))


if __name__ == "__main__":
    main()
