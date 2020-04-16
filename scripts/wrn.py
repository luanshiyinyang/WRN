"""
Author: Zhou Chen
Date: 2020/4/16
Desc: desc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_prob=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout_prob = dropout_prob
        self.in_is_out = (in_planes == out_planes)
        # 是否需要shortcut
        self.shortcut = (not self.in_is_out) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                           padding=0, bias=False) or None

    def forward(self, x):
        if not self.in_is_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.in_is_out else x)))
        if self.dropout_prob > 0:
            out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.in_is_out else self.shortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_prob=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_prob)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes=101, k=1, dropout_prob=0.0):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * k, 32 * k, 64 * k]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # block1
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, dropout_prob)
        # block2
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, dropout_prob)
        # block3
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, dropout_prob)

        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 1)
        out = out.view(-1, self.n_channels)
        return self.fc(out)
