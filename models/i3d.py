import torch.nn as nn
from .backbones import resnet3d

__all__ = ['i3d_resnet18', 'i3d_resnet34', 'i3d_resnet50', 'i3d_resnet101', 'i3d_resnet152']


class I3D(nn.Module):
    """
    Implements a I3D Network for action recognition.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        classifier (nn.Module): module that takes the features returned from the
            backbone and returns classification scores.
    """

    def __init__(self, backbone, classifier):
        super(I3D, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class I3DHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
    """

    def __init__(self, num_classes, in_channels, dropout_ratio=0.5):
        super(I3DHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


def _load_model(backbone_name, pretrained2d, progress, modality, num_classes, **kwargs):
    backbone = resnet3d(arch=backbone_name, pretrained2d=pretrained2d, progress=progress, modality=modality)
    classifier = I3DHead(num_classes=num_classes, in_channels=2048, **kwargs)
    model = I3D(backbone, classifier)
    return model


def i3d_resnet18(pretrained2d=False, progress=True, modality='RGB', num_classes=21, **kwargs):
    """Constructs a I3D model with a ResNet3d-18 backbone.

    Args:
        pretrained2d (bool): If True, the backbone utilize the pretrained parameters
            in 2d models
        progress (bool): If True, displays a progress bar of the download to stderr
        modality (str): The modality of input data (RGB or Flow)
        num_classes (int): Number of dataset classes
    """
    return _load_model('resnet18', pretrained2d, progress, modality, num_classes, **kwargs)


def i3d_resnet34(pretrained2d=False, progress=True, modality='RGB', num_classes=21, **kwargs):
    """Constructs a I3D model with a ResNet3d-34 backbone.

    Args:
        pretrained2d (bool): If True, the backbone utilize the pretrained parameters
            in 2d models
        progress (bool): If True, displays a progress bar of the download to stderr
        modality (str): The modality of input data (RGB or Flow)
        num_classes (int): Number of dataset classes
    """
    return _load_model('resnet34', pretrained2d, progress, modality, num_classes, **kwargs)


def i3d_resnet50(pretrained2d=False, progress=True, modality='RGB', num_classes=21, **kwargs):
    """Constructs a I3D model with a ResNet3d-50 backbone.

    Args:
        pretrained2d (bool): If True, the backbone utilize the pretrained parameters
            in 2d models
        progress (bool): If True, displays a progress bar of the download to stderr
        modality (str): The modality of input data (RGB or Flow)
        num_classes (int): Number of dataset classes
    """
    return _load_model('resnet50', pretrained2d, progress, modality, num_classes, **kwargs)


def i3d_resnet101(pretrained2d=False, progress=True, modality='RGB', num_classes=21, **kwargs):
    """Constructs a I3D model with a ResNet3d-101 backbone.

    Args:
        pretrained2d (bool): If True, the backbone utilize the pretrained parameters
            in 2d models
        progress (bool): If True, displays a progress bar of the download to stderr
        modality (str): The modality of input data (RGB or Flow)
        num_classes (int): Number of dataset classes
    """
    return _load_model('resnet101', pretrained2d, progress, modality, num_classes, **kwargs)


def i3d_resnet152(pretrained2d=False, progress=True, modality='RGB', num_classes=21, **kwargs):
    """Constructs a I3D model with a ResNet3d-152 backbone.

    Args:
        pretrained2d (bool): If True, the backbone utilize the pretrained parameters
            in 2d models
        progress (bool): If True, displays a progress bar of the download to stderr
        modality (str): The modality of input data (RGB or Flow)
        num_classes (int): Number of dataset classes
    """
    return _load_model('resnet152', pretrained2d, progress, modality, num_classes, **kwargs)
