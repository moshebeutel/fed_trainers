import logging
import torch
import torch.nn.functional as F
from torch import nn
from utils import get_n_params


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, gn_groups=4, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(gn_groups, planes, affine=False)
        # self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(gn_groups, planes, affine=False)
        # self.bn2 = nn.BatchNorm2d(planes, affine=False)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out = out + identity
        out = self.relu(out)

        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 use_batchnorm=False,
                 use_dropout=True,
                 activation='relu'):
        super(DenseBlock, self).__init__()
        self._fc = nn.Linear(in_channels, out_channels)
        self._batch_norm = nn.BatchNorm1d(out_channels) if use_batchnorm \
            else nn.Identity(out_channels)
        self._act = nn.ReLU() if activation == 'relu' else nn.ELU()
        self._dropout = nn.Dropout(.5) if use_dropout else nn.Identity(out_channels)

    def forward(self, x):
        # return self._dropout(self._relu(self._batch_norm(self._fc(x))))
        return self._act(self._dropout(self._batch_norm(self._fc(x))))


class FeatureModel(nn.Module):
    def __init__(self, num_features=128,
                 number_of_classes=100,
                 cls_layer=True,
                 use_softmax=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._output_info_fn = logging.info
        self._output_debug_fn = logging.debug
        self.cls_layer = cls_layer
        self.use_softmax = use_softmax
        self._output_info_fn(
            f'FeatureModel: num_features={num_features} cls_layer={cls_layer} use_softmax={use_softmax}')

        # self._dense_block1 = DenseBlock(num_features, 2 * num_features)
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        # self._dense_block2 = DenseBlock(2 * num_features, 2 * num_features)
        # # use_batchnorm=use_group_norm, use_dropout=use_dropout)
        #
        # self._dense_block3 = DenseBlock(2 * num_features, 2 * num_features)
        # # use_batchnorm=use_group_norm, use_dropout=use_dropout)
        #
        # self._dense_block4 = DenseBlock(2 * num_features, num_features,
        #                                 use_dropout=False, activation='elu')
        # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        # self._dense_block5 = DenseBlock(4 * num_features, 2 * num_features)
        # # use_batchnorm=use_group_norm, use_dropout=use_dropout)

        self._dense_block = DenseBlock(num_features, int(0.5 * num_features))

        if self.cls_layer:
            self._output = nn.Linear(int(0.5 * num_features), number_of_classes)

    def forward(self, x):
        self._output_debug_fn(f'input {x.shape}')

        # fc1 = self._dense_block1(x)
        # self._output_debug_fn(f'fc1 {fc1.shape}')
        #
        # fc2 = self._dense_block2(fc1)
        # self._output_debug_fn(f'fc2 {fc2.shape}')
        #
        # fc3 = self._dense_block3(fc2)
        # self._output_debug_fn(f'fc3 {fc3.shape}')
        #
        # fc4 = self._dense_block4(fc3)
        # self._output_debug_fn(f'fc4 {fc4.shape}')

        x = self._dense_block(x)
        self._output_debug_fn(f'x {x.shape}')

        output = x
        if self.cls_layer:
            logits = self._output(x)
            self._output_debug_fn(f'logits {logits.shape}')
            output = logits
            if self.use_softmax:
                probs = F.softmax(logits, dim=1)
                self._output_debug_fn(f'softmax {probs.shape}')
                output = probs

        return output


class ResNet(nn.Module):

    def __init__(self, layers, num_classes=10, basic_block_cls=BasicBlock):
        super(ResNet, self).__init__()

        self._output_info_fn = logging.info
        self._output_debug_fn = logging.debug

        self.gn_groups = 4
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.gn1 = nn.GroupNorm(self.gn_groups, 16, affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.layers = nn.ModuleList(
            [self._make_layer(basic_block_cls, 2 ** (i + 4), layers[i], stride=2 if i > 0 else 1)
             for i in range(len(layers))])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        # self.fc = FeatureModel(num_features=2 ** (int(len(layers) + 3)),
        #                        number_of_classes=num_classes,
        #                        cls_layer=True,
        #                        use_softmax=True)

        # self._output_info_fn(str(self))

        self._output_info_fn(f"Number Parameters: {get_n_params(self)}")

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.GroupNorm(self.gn_groups, self.inplanes, affine=False),
                # nn.BatchNorm2d(self.inplanes, affine=False),
            )

        layers = [block(self.inplanes, planes, stride, self.gn_groups, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        self._output_debug_fn(f'before conv1 input {x.shape}')
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        self._output_debug_fn(f'input {x.shape}')
        for layer in self.layers:
            x = layer(x)
            self._output_debug_fn(f'input {x.shape}')

        self._output_debug_fn(f'before avgpool input {x.shape}')
        x = self.avgpool(x)
        self._output_debug_fn(f'input {x.shape}')
        x = x.view(x.size(0), -1)
        self._output_debug_fn(f'before feature model input {x.shape}')
        x = self.fc(x)
        self._output_debug_fn(f'output {x.shape}')

        return x
