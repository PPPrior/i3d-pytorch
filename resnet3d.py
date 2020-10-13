import torch.nn as nn
from torch.nn.modules.utils import _triple, _ntuple


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 inflate=True,
                 inflate_style='3x1x1',
                 norm_layer=None):
        super(BasicBlock3d, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.inflate_style = inflate_style
        self.downsample = downsample

        if inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(temporal_stride,
                    spatial_stride, spatial_stride),
            padding=conv1_padding,
            dilation=(1, dilation, dilation),
            bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            conv2_kernel_size,
            stride=(1, 1, 1),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 inflate=True,
                 inflate_style='3x1x1',
                 norm_layer=None):
        super(Bottleneck3d, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample

        if inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            conv1_kernel_size,
            padding=conv1_padding,
            bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            conv2_kernel_size,
            stride=(temporal_stride,
                    spatial_stride, spatial_stride),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv3d(
            planes,
            planes * self.expansion,
            1,
            bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):
    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 base_channels=64,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=(5, 7, 7),
                 conv1_stride_t=2,
                 pool1_stride_t=2,
                 with_pool2=True,
                 inflate=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 **kwargs):
        super(ResNet3d, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(
            dilations) == num_stages
        self.conv1_kernel = conv1_kernel
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_t = pool1_stride_t
        self.with_pool2 = with_pool2
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.inflate_style = inflate_style
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.base_channels

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]  # Default: (1, 2, 2, 2)
            temporal_stride = temporal_strides[i]  # Default: (1, 1, 1, 1)
            dilation = dilations[i]  # Default: (1, 1, 1, 1)
            planes = self.base_channels * 2 ** i  # (64, 128, 256, 512)
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp,
                **kwargs)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2 ** (
                len(self.stage_blocks) - 1)

    def make_res_layer(self,
                       block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       inflate=1,
                       inflate_style='3x1x1',
                       with_cp=False,
                       **kwargs):

        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate,) * blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Conv3d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False)

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                with_cp=with_cp,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    with_cp=with_cp,
                    **kwargs))

        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        norm_layer = nn.BatchNorm3d
        self.conv1 = nn.Conv3d(
            self.in_channels,  # Default: 3
            self.base_channels,  # Default: 64
            kernel_size=self.conv1_kernel,  # Default: (5, 7, 7)
            stride=(self.conv1_stride_t, 2, 2),  # Default: 2
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            bias=False)
        self.bn1 = norm_layer(self.base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(self.pool1_stride_t, 2, 2),  # Default: 2
            padding=(0, 1, 1))

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
