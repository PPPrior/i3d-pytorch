import torch
from models.backbones import resnet3d


def main():
    model = resnet3d('resnet50', pretrained2d=True)
    # batch_size x channels x frames x height x width
    dummy_input = torch.rand(8, 3, 10, 224, 224)
    output = model(dummy_input)
    print(output.shape)


if __name__ == '__main__':
    main()
