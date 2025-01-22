import argparse
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from torch import nn
from torch import optim
import torch
import torchvision
import itertools as it


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default="cifar-10", choices=["cifar-10", "cifar-100"]
    )
    parser.add_argument("--model_name", default="resnet-18", choices=["resnet-18"])
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
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
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
    name = model_hp["name"]
    name_to_cls = {"resnet-18": torchvision.models.resnet18}
    cls = name_to_cls[name]
    return cls(**model_hp)


def make_optimizer(optimizer_hp: dict) -> optim.Optimizer:
    name = optimizer_hp["name"]
    name_to_cls = {
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
    }
    cls = name_to_cls[name]
    return cls(**optimizer_hp)


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

        pred_y = model(y)
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

        pred_y = model(y)
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
        k = f"{parent_key}{sep}{k}"
        if isinstance(v, dict):
            items.extend(flatten_dict(v, k, sep=sep).items())
        else:
            items.append((k, v))
    return dict(items)


def unflatten_dict(d: dict, sep=".") -> dict:
    result = {}
    for k, v in d.items():
        keys = k.split(sep)
        d = result
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return result


def make_combinations(hps: dict) -> list[dict]:
    flat_hps = flatten_dict(hps)
    keys, values = zip(*flat_hps.items())
    return [
        unflatten_dict(dict(zip(keys, combination)))
        for combination in it.product(*values)
    ]


def cv_main(cfg):
    dataset, test_dataset = make_datasets(dataset_name=cfg["dataset_name"])
    train_transform, test_transform = make_transforms(dataset_name=cfg["dataset_name"])
    criterion = make_criterion(dataset_name=cfg["dataset_name"])

    hps = make_combinations(cfg["hps"])
    kfold = StratifiedKFold(shuffle=True)
    for hp in hps:
        for i, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            train_dataset = TransformedDataset(
                Subset(dataset, train_idx), train_transform
            )
            train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"])

            val_dataset = TransformedDataset(Subset(dataset, val_idx), test_transform)
            val_loader = DataLoader(val_dataset, batch_size=hp["batch_size"])

            model = make_model(model_hp=hp["model"])
            optimizer = make_optimizer(optimizer_hp=hp["optimizer"])

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

    test_dataset = TransformedDataset(test_dataset, test_transform)
    test_loader = DataLoader(test_dataset)


def main():
    parser = make_parser()
    args = parser.parse_args()
    cfg = {}
    cv_main(cfg)


if __name__ == "__main__":
    main()
