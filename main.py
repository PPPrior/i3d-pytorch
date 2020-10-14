import torch
from models.backbones import resnet3d
from models import *

import torch.nn as nn
from torchvision.models.resnet import *


def test_resnet3d():
    model = resnet3d('resnet50', pretrained2d=True)
    # batch_size x channels x frames x height x width
    dummy_input = torch.rand(8, 3, 10, 224, 224)
    output = model(dummy_input)
    print(output.shape)


def test_i3d():
    model = i3d_resnet50(pretrained2d=True, num_classes=101)
    # batch_size x channels x frames x height x width
    dummy_input = torch.rand(8, 3, 10, 224, 224)
    output = model(dummy_input)
    print(output.shape)


def main():
    test_i3d()


if __name__ == '__main__':
    main()
