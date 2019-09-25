import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, params=None):
        if params is None:
            out = self.conv1(x)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.nChannels)
            return self.fc(out)
        else:
            out = F.conv2d(x, params['conv1.weight'], bias=None, stride=self.conv1.stride, 
                padding=self.conv1.padding, dilation=self.conv1.dilation)
            for b in range(1,4):
                for l in range(4):
                    out = self.BasicBlock(out, params, b, l)
            out = F.batch_norm(out, 
                           params['bn1.running_mean'], 
                           params['bn1.running_var'], 
                           params['bn1.weight'], 
                           params['bn1.bias'],
                           training=self.training,
                           momentum=0.001, 
                           eps=1e-5)
            out = self.relu(out)
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.nChannels)
            out = F.linear(out, params['fc.weight'], params['fc.bias'])
            return out

    def BasicBlock(self, x, params, block_index, layer_index):
        block = getattr(self, 'block{}'.format(block_index))
        layer = block.layer[layer_index]
        if block_index == 1 and layer_index == 0:
            x = F.batch_norm(x, 
                           params['block{}.layer.{}.bn1.running_mean'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn1.running_var'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn1.weight'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn1.bias'.format(block_index, layer_index)],
                           training=self.training,
                           momentum=0.001, 
                           eps=1e-5)
            x = self.relu(x)
            out = F.conv2d(x, 
                params['block{}.layer.{}.conv1.weight'.format(block_index, layer_index)], 
                bias=params['block{}.layer.{}.conv1.bias'.format(block_index, layer_index)], 
                stride=layer.conv1.stride, 
                padding=layer.conv1.padding, 
                dilation=layer.conv1.dilation)
        elif block_index != 1 and layer_index == 0:
            out = F.batch_norm(x, 
                           params['block{}.layer.{}.bn1.running_mean'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn1.running_var'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn1.weight'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn1.bias'.format(block_index, layer_index)],
                           training=self.training,
                           momentum=0.001, 
                           eps=1e-5)
            out = self.relu(out)
            out = F.conv2d(x, 
                params['block{}.layer.{}.conv1.weight'.format(block_index, layer_index)], 
                bias=params['block{}.layer.{}.conv1.bias'.format(block_index, layer_index)], 
                stride=layer.conv1.stride, 
                padding=layer.conv1.padding, 
                dilation=layer.conv1.dilation)

        else:
            out = F.batch_norm(x, 
                           params['block{}.layer.{}.bn1.running_mean'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn1.running_var'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn1.weight'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn1.bias'.format(block_index, layer_index)],
                           training=self.training,
                           momentum=0.001, 
                           eps=1e-5)
            out = self.relu(out)
            out = F.conv2d(out, 
                params['block{}.layer.{}.conv1.weight'.format(block_index, layer_index)], 
                bias=params['block{}.layer.{}.conv1.bias'.format(block_index, layer_index)], 
                stride=layer.conv1.stride, 
                padding=layer.conv1.padding, 
                dilation=layer.conv1.dilation)
        # if self.droprate > 0:
        #     out = F.dropout(out, p=self.droprate, training=self.training)
        out = F.batch_norm(out, 
                           params['block{}.layer.{}.bn2.running_mean'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn2.running_var'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn2.weight'.format(block_index, layer_index)], 
                           params['block{}.layer.{}.bn2.bias'.format(block_index, layer_index)],
                           training=self.training,
                           momentum=0.001, 
                           eps=1e-5)
        out = self.relu(out)
        out = F.conv2d(out, 
            params['block{}.layer.{}.conv2.weight'.format(block_index, layer_index)], 
            bias=params['block{}.layer.{}.conv2.bias'.format(block_index, layer_index)], 
            stride=layer.conv2.stride, 
            padding=layer.conv2.padding, 
            dilation=layer.conv2.dilation)
        if layer.convShortcut is not None:
            res = F.conv2d(x, 
                params['block{}.layer.{}.convShortcut.weight'.format(block_index, layer_index)], 
                bias=params['block{}.layer.{}.convShortcut.bias'.format(block_index, layer_index)], 
                stride=layer.convShortcut.stride, 
                padding=layer.convShortcut.padding, 
                dilation=layer.convShortcut.dilation)
        else:
            res = x
        out = out + res
        return out





