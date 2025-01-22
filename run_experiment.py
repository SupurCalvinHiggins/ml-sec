import argparse
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from torch import nn
from torch import optim
import torch
import torchvision
import itertools as it
import json
import shutil
import copy
import numpy as np
import random
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=Path)
    parser.add_argument("--clean", default=False, action="store_true")
    return parser


def make_normalize_transform(dataset_name: str):
    dataset_name_to_mean = {
        "cifar-10": (0.4914, 0.4822, 0.4465),
        "cifar-100": (0.5071, 0.4865, 0.4409),
    }
    dataset_name_to_std = {
        "cifar-10": (0.2471, 0.2435, 0.2616),
        "cifar-100": (0.2673, 0.2564, 0.2762),
    }
    return transforms.Normalize(
        dataset_name_to_mean[dataset_name], dataset_name_to_std[dataset_name]
    )


def make_transforms(dataset_name: str):
    normalize_transform = make_normalize_transform(dataset_name)
    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    train_transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform,
        ]
    )
    return train_transform, test_transform


def make_datasets(dataset_name: str) -> tuple[Dataset, Dataset]:
    dataset_name_to_cls = {"cifar-10": datasets.CIFAR10, "cifar-100": datasets.CIFAR100}
    cls = dataset_name_to_cls[dataset_name]
    train_dataset = cls(root="./data", train=True, download=True)
    test_dataset = cls(root="./data", train=False, download=False)
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


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        pred_y = model(x)
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (pred_y.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    acc = correct / total
    avg_loss = total_loss / total
    return avg_loss, acc


@torch.enable_grad()
def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
    max_epochs: int,
) -> tuple[list[float], list[float]]:
    model.train()
    avg_losses, accs = [], []
    for _ in range(max_epochs):
        avg_loss, acc = train_epoch(model, optimizer, criterion, loader)
        avg_losses.append(avg_loss)
        accs.append(acc)
    return avg_losses, accs


def eval_epoch(
    model: nn.Module, criterion: nn.Module, loader: DataLoader
) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred_y = model(x)
        loss = criterion(pred_y, y)

        total_loss += loss.item() * y.size(0)
        correct += (pred_y.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    acc = correct / total
    avg_loss = total_loss / total
    return avg_loss, acc


@torch.no_grad()
def eval(
    model: nn.Module, criterion: nn.Module, loader: DataLoader
) -> tuple[float, float]:
    model.eval()
    return eval_epoch(model, criterion, loader)


class TransformedDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


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


def seed(x: int) -> None:
    torch.manual_seed(x)
    random.seed(x)
    np.random.seed(x)


def cv_main(cfg: dict) -> tuple[nn.Module, dict]:
    seed(cfg["seed"])

    dataset, test_dataset = make_datasets(dataset_name=cfg["dataset_name"])
    train_transform, test_transform = make_transforms(dataset_name=cfg["dataset_name"])
    criterion = make_criterion(dataset_name=cfg["dataset_name"]).to(device)

    kfold = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True)
    idx = list(range(len(dataset)))
    y = [y for _, y in dataset]
    folds = list(kfold.split(idx, y))

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
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            train_dataset = TransformedDataset(
                Subset(dataset, train_idx), train_transform
            )
            train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], pin_memory=True, num_workers=4)

            val_dataset = TransformedDataset(Subset(dataset, val_idx), test_transform)
            val_loader = DataLoader(val_dataset, batch_size=hp["batch_size"])

            model = make_model(model_hp=hp["model"]).to(device)
            optimizer = make_optimizer(model, optimizer_hp=hp["optimizer"])

            train_avg_losses, train_accs = train(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                loader=train_loader,
                max_epochs=hp["max_epochs"],
            )
            val_loss, val_acc = eval(
                model=model, criterion=criterion, loader=val_loader
            )
            print(
                f"\t[Fold {fold_idx + 1}/{len(folds)}] val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}"
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

    train_dataset = TransformedDataset(dataset, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=best_hp["batch_size"])

    test_dataset = TransformedDataset(test_dataset, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=best_hp["batch_size"])

    model = make_model(model_hp=best_hp["model"]).to(device)
    optimizer = make_optimizer(model, optimizer_hp=best_hp["optimizer"])

    train_avg_losses, train_accs = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        loader=train_loader,
        max_epochs=best_hp["max_epochs"],
    )
    test_loss, test_acc = eval(model=model, criterion=criterion, loader=test_loader)
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
    model, sweep = cv_main(cfg)

    model_path = result_path / "model.pth"
    sweep_path = result_path / "sweep.json"

    torch.save(model.state_dict(), model_path)
    sweep_path.write_text(json.dumps(sweep))


if __name__ == "__main__":
    main()
