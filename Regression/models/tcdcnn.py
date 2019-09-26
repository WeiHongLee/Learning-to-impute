import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TCDCNN(nn.Module):
	def __init__(self):
		super(TCDCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
		self.conv2 = nn.Conv2d(20,48,kernel_size=5)
		self.conv3 = nn.Conv2d(48,64,kernel_size=3)
		self.conv4=  nn.Conv2d(64,80,kernel_size=3)
		self.fc = nn.Linear(720, 256)
		self.linear = nn.Linear(256, 10)
		self.dropout = nn.Dropout()

	def forward(self, x, params=None, IsUnlabeled=False):
		if params is None:
			x = F.max_pool2d(F.hardtanh(self.conv1(x)),2)
			x = F.max_pool2d(F.hardtanh(self.conv2(x)),2) 
			x = F.max_pool2d(F.hardtanh(self.conv3(x)),2)
			x_tanh = self.conv4(x)
			x = F.hardtanh(x_tanh)
			x = x.view(x.size(0),-1)
			x = self.fc(x)
			if IsUnlabeled:
				x = self.dropout(x)
			out = self.linear(x)
		else:
			x = F.max_pool2d(F.hardtanh(F.conv2d(x, params['conv1.weight'], bias=params['conv1.bias'], stride=self.conv1.stride, 
                padding=self.conv1.padding, dilation=self.conv1.dilation)),2)
			x = F.max_pool2d(F.hardtanh(F.conv2d(x, params['conv2.weight'], bias=params['conv2.bias'], stride=self.conv2.stride, 
                padding=self.conv2.padding, dilation=self.conv2.dilation)),2)
			x = F.max_pool2d(F.hardtanh(F.conv2d(x, params['conv3.weight'], bias=params['conv3.bias'], stride=self.conv3.stride, 
                padding=self.conv3.padding, dilation=self.conv3.dilation)),2)
			x_tanh = F.conv2d(x, params['conv4.weight'], bias=params['conv4.bias'], stride=self.conv4.stride, 
                padding=self.conv4.padding, dilation=self.conv4.dilation)
			x = F.hardtanh(x_tanh)
			x = x.view(x.size(0),-1)
			x = F.linear(x, params['fc.weight'], params['fc.bias'])
			out = F.linear(x, params['linear.weight'], params['linear.bias'])

		return out