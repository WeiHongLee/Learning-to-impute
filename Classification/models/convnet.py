import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb



class ConvNet(nn.Module):
    def __init__(self, num_classes, droprate=0.5):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1,
                              padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                              padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                              padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(p=droprate)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1,
                              padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                              padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                              padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.bn9 = nn.BatchNorm2d(128)
        self.linear = nn.Linear(128, num_classes)
        self.bnlinear = nn.BatchNorm1d(num_classes)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.droprate = droprate



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
#             pdb.set_trace()
            out = self.conv1(x)
            out = self.bn1(out)
            self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            self.relu(out)
            out = F.max_pool2d(out, 2)
            out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv4(out)
            out = self.bn4(out)
            self.relu(out)
            out = self.conv5(out)
            out = self.bn5(out)
            self.relu(out)
            out = self.conv6(out)
            out = self.bn6(out)
            self.relu(out)
            out = F.max_pool2d(out, 2)
            out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv7(out)
            out = self.bn7(out)
            self.relu(out)
            out = self.conv8(out)
            out = self.bn8(out)
            self.relu(out)
            out = self.conv9(out)
            out = self.bn9(out)
            self.relu(out)
            out = F.avg_pool2d(out, 6)
#             pdb.set_trace()
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            out = self.bnlinear(out)
            return out
        else:
            out = F.conv2d(x, params['conv1.weight'], bias=None, stride=self.conv1.stride, 
                padding=self.conv1.padding, dilation=self.conv1.dilation)
            out = F.batch_norm(out, 
                           params['bn1.running_mean'], 
                           params['bn1.running_var'], 
                           params['bn1.weight'], 
                           params['bn1.bias'],
                           training=self.training,
                           eps=1e-5)
            self.relu(out)
            out = F.conv2d(out, params['conv2.weight'], bias=None, stride=self.conv2.stride, 
                padding=self.conv2.padding, dilation=self.conv2.dilation)
            out = F.batch_norm(out, 
                           params['bn2.running_mean'], 
                           params['bn2.running_var'], 
                           params['bn2.weight'], 
                           params['bn2.bias'],
                           training=self.training,
                           eps=1e-5)
            self.relu(out)
            out = F.conv2d(out, params['conv3.weight'], bias=None, stride=self.conv3.stride, 
                padding=self.conv3.padding, dilation=self.conv3.dilation)
            out = F.batch_norm(out, 
                           params['bn3.running_mean'], 
                           params['bn3.running_var'], 
                           params['bn3.weight'], 
                           params['bn3.bias'],
                           training=self.training,
                           eps=1e-5)
            self.relu(out)
            out = F.max_pool2d(out, 2)
            out = F.conv2d(out, params['conv4.weight'], bias=None, stride=self.conv4.stride, 
                padding=self.conv4.padding, dilation=self.conv4.dilation)
            out = F.batch_norm(out, 
                           params['bn4.running_mean'], 
                           params['bn4.running_var'], 
                           params['bn4.weight'], 
                           params['bn4.bias'],
                           training=self.training,
                           eps=1e-5)
            self.relu(out)
            out = F.conv2d(out, params['conv5.weight'], bias=None, stride=self.conv5.stride, 
                padding=self.conv5.padding, dilation=self.conv5.dilation)
            out = F.batch_norm(out, 
                           params['bn5.running_mean'], 
                           params['bn5.running_var'], 
                           params['bn5.weight'], 
                           params['bn5.bias'],
                           training=self.training,
                           eps=1e-5)
            self.relu(out)
            out = F.conv2d(out, params['conv6.weight'], bias=None, stride=self.conv6.stride, 
                padding=self.conv6.padding, dilation=self.conv6.dilation)
            out = F.batch_norm(out, 
                           params['bn6.running_mean'], 
                           params['bn6.running_var'], 
                           params['bn6.weight'], 
                           params['bn6.bias'],
                           training=self.training,
                           eps=1e-5)
            self.relu(out)
            out = F.max_pool2d(out, 2)
            out = F.conv2d(out, params['conv7.weight'], bias=None, stride=self.conv7.stride, 
                padding=self.conv7.padding, dilation=self.conv7.dilation)
            out = F.batch_norm(out, 
                           params['bn7.running_mean'], 
                           params['bn7.running_var'], 
                           params['bn7.weight'], 
                           params['bn7.bias'],
                           training=self.training,
                           eps=1e-5)
            self.relu(out)
            out = F.conv2d(out, params['conv8.weight'], bias=None, stride=self.conv8.stride, 
                padding=self.conv8.padding, dilation=self.conv8.dilation)
            out = F.batch_norm(out, 
                           params['bn8.running_mean'], 
                           params['bn8.running_var'], 
                           params['bn8.weight'], 
                           params['bn8.bias'],
                           training=self.training,
                           eps=1e-5)
            self.relu(out)
            out = F.conv2d(out, params['conv9.weight'], bias=None, stride=self.conv9.stride, 
                padding=self.conv9.padding, dilation=self.conv9.dilation)
            out = F.batch_norm(out, 
                           params['bn9.running_mean'], 
                           params['bn9.running_var'], 
                           params['bn9.weight'], 
                           params['bn9.bias'],
                           training=self.training,
                           eps=1e-5)
            self.relu(out)
            out = F.avg_pool2d(out, 6)
            out = out.view(out.size(0), -1)
            out = F.linear(out, params['linear.weight'], params['linear.bias'])
            out = F.batch_norm(out, 
                           params['bnlinear.running_mean'], 
                           params['bnlinear.running_var'], 
                           params['bnlinear.weight'], 
                           params['bnlinear.bias'],
                           training=self.training,
                           eps=1e-5)
            return out





