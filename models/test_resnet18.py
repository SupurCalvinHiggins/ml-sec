import torch
import random
import numpy as np
from resnet18 import ResNet18
from torchvision.models.resnet import resnet18


def seed(x: int) -> None:
    torch.manual_seed(x)
    random.seed(x)
    np.random.seed(x)


def test_resnet18() -> None:
    seed(0)
    x = torch.rand((1, 3, 224, 224))
    seed(0)
    model1 = ResNet18(num_classes=1000)
    seed(0)
    model2 = resnet18(zero_init_residual=True)
    assert torch.equal(model1(x), model2(x))
