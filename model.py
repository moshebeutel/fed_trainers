import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
    def __init__(self,
                 num_channels=24,
                 num_features=8,
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
        self._num_features = num_features
        self._num_channels = num_channels
        self._num_classes = number_of_classes
        self._blk1 = nn.Conv1d(num_channels, 2 * num_channels, 3, padding=1)
        self._blk2 = nn.Conv1d(2 * num_channels, 4 * num_channels, 3, padding=1)
        self._blk3 = nn.Conv1d(4 * num_channels, 8 * num_channels, 3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8 * num_channels, num_channels)

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

        # self._dense_block = DenseBlock(num_features, int(0.5 * num_features))

        if self.cls_layer:
            self._output = nn.Linear(num_channels, number_of_classes)

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

        x = torch.reshape(x, (x.shape[0], self._num_channels, self._num_features))
        self._output_debug_fn(f'input {x.shape}')

        x = self._blk1(x)
        self._output_debug_fn(f'x after blk1 {x.shape}')

        x = self._blk2(x)
        self._output_debug_fn(f'x after blk2 {x.shape}')

        x = self._blk3(x)
        self._output_debug_fn(f'x after blk3 {x.shape}')

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        self._output_debug_fn(f'x after avgpool {x.shape}')

        x = self.fc(x)
        self._output_debug_fn(f'x after fc {x.shape}')

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

    def __init__(self, layers, num_classes=10, in_channels=3, basic_block_cls=BasicBlock):
        super(ResNet, self).__init__()

        self._output_info_fn = logging.info
        self._output_debug_fn = logging.debug

        self.gn_groups = 4
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(in_channels, 16)
        self.gn1 = nn.GroupNorm(self.gn_groups, 16, affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.layers = nn.ModuleList(
            [self._make_layer(basic_block_cls, 2 ** (i + 4), layers[i], stride=2 if i > 0 else 1)
             for i in range(len(layers))])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2 ** (int(len(layers) + 3)), num_classes)
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


class MLPTarget(nn.Module):
    def __init__(self, num_features=192, num_classes=100, use_softmax=False):
        super(MLPTarget, self).__init__()

        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self._use_softmax = use_softmax

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.softmax(x, dim=1) if self._use_softmax else x
        return x


class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, embedding_dim=84, num_classes=10, use_cls_layer=False, use_softmax=False):
        super(CNNTarget, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        representation_size = (32 if in_channels == 1 else 50) * n_kernels
        self.fc1 = nn.Linear(representation_size, 120)
        self.fc2 = nn.Linear(120, 84)

        self.embed_dim = nn.Linear(84, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, num_classes)
        assert use_cls_layer or (not use_softmax), f'Cannot use softmax on representation layer'
        self._use_cls_layer = use_cls_layer
        self._use_softmax = use_softmax

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.embed_dim(x)
        x = F.relu(self.fc3(x)) if self._use_cls_layer else x
        x = F.softmax(self.fc3(x), dim=1) if self._use_softmax else x
        return x


def initialize_weights(module: nn.Module):
    for m in module.modules():

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()


def get_n_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    number_params = sum([np.prod(p.size()) for p in model_parameters])
    return number_params


def get_model(args):
    num_classes = {'cifar10': 10, 'cifar100': 100, 'putEMG': 8, 'mnist': 10, 'femnist': 62, 'keypressemg': 26}[args.data_name]
    in_channels = 1 if args.data_name in ['mnist', 'femnist'] else 3

    if args.data_name in ['cifar10', 'cifar100', 'mnist', 'femnist']:

        assert args.model_name in ['CNNTarget', 'ResNet'], f'Unxpected model name {args.model_name}'

        if args.model_name == 'CNNTarget':
            model = CNNTarget(in_channels=in_channels, n_kernels=args.n_kernels, embedding_dim=args.embed_dim, use_cls_layer=(not args.use_gp))
        else:
            model = ResNet(layers=[args.block_size] * args.num_blocks, num_classes=num_classes, in_channels=in_channels)

    elif args.data_name == 'keypressemg':
        assert num_classes == 26, 'num_classes should be 26'
        import keypressemg
        from keypressemg.models.feature_model import FeatureModel
        model = FeatureModel(num_features=args.num_features, number_of_classes=args.num_classes, cls_layer=True, depth_power=args.depth_power)
    else:
        assert args.data_name == 'putEMG', 'data_name should be putEMG'
        assert num_classes == 8, 'num_classes should be 8'
        import keypressemg
        from keypressemg.models.feature_model import FeatureModel
        model = FeatureModel(num_features=args.num_features, number_of_classes=args.num_classes, cls_layer=True,
                             depth_power=args.depth_power)
        # model = MLPTarget(num_features=24 * 8, num_classes=num_classes, use_softmax=True)

    initialize_weights(model)
    return model


if __name__ == '__main__':
    model = MLPTarget()
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(num_params)
