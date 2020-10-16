import torch
from torch.nn.utils import clip_grad_norm

from models.backbones import resnet3d
from models import i3d
from dataset import I3DDataSet
from transforms import *
from opts import parser


def test_resnet3d():
    model = resnet3d('resnet50', pretrained2d=True)
    # batch_size x channels x frames x height x width
    dummy_input = torch.rand(8, 3, 10, 224, 224)
    output = model(dummy_input)
    print(output.shape)


def test_i3d():
    model = getattr(i3d, 'i3d_resnet50')(pretrained2d=True, num_classes=101)
    # batch_size x channels x frames x height x width
    dummy_input = torch.rand(8, 3, 10, 224, 224)
    output = model(dummy_input)
    print(output.shape)


def test():
    test_i3d()


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = getattr(i3d, args.arch)(pretrained2d=True, num_class=num_class,
                                    dropout_ratio=args.dropout)


if __name__ == '__main__':
    test()
