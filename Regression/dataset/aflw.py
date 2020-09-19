from __future__ import division

import os
import numpy as np 
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
import pdb
import random
from scipy.io import loadmat
import torchvision.transforms as transforms

from PIL import Image
# from .randaugment import RandAugmentColor

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class RandomTranslateWithReflect:
	def __init__(self, max_translation):
		self.max_translation = max_translation

	def __call__(self, old_image):
		xtranslation, ytranslation = np.random.randint(-self.max_translation,
														self.max_translation + 1,
														size=2)
		xpad, ypad = abs(xtranslation), abs(ytranslation)
		xsize, ysize = old_image.size

		flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
		flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
		flipped_both = old_image.transpose(Image.ROTATE_180)

		new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

		new_image.paste(old_image, (xpad, ypad))

		new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
		new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

		new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
		new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

		new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
		new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
		new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
		new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

		new_image = new_image.crop((xpad - xtranslation,
									ypad - ytranslation,
									xpad + xsize - xtranslation,
									ypad + ysize - ytranslation))

		return new_image, xtranslation, ytranslation
class Flipping:
	def __init__(self, p=0.5):
		self.p = p
	def __call__(self, image, landmarks):
		if random.random() < self.p:
			image = image.transpose(Image.FLIP_LEFT_RIGHT)
			w, h = image.size
			landmarks[:, 0] = w - landmarks[:, 0]
		return image, landmarks
class Scaling:
	def __init__(self, p=0.9, im_size=60):
		self.p = p
		self.im_size = im_size
	def __call__(self, image, landmarks):
		w, h = image.size
		percent = self.p + random.random()
		temp_size = int(round(self.im_size/percent))
		margin = int(round((temp_size - im_size) / 2))
		temp_image = ttf.resize(image, [temp_size, temp_size])
		image = temp_image.crop(margin, margin, margin+im_size, margin+im_size)
		landmarks[:,0] = landmarks[:,0] / w * temp_size - margin
		landmarks[:,1] = landmarks[:,1] / h * temp_size - margin
		return image, landmarks



def get_alfw(root, n_labeled,
                 transform_train=None, transform_val=None, IsVal=False, UseCeleba=False):

	train_data, val_data, test_data = load_dataset(root)
	# pdb.set_trace()
	if n_labeled < 0:
		train_labeled_data = train_data
		train_unlabeled_data = train_data
		train_val_data = train_data
	else:
		if n_labeled < 1.:
			repeat_flag = n_labeled
			n_labeled = int(np.round(len(train_data) * n_labeled))
		idxs = torch.randperm(len(train_data))
		train_labeled_data = [train_data[i] for i in idxs[:n_labeled]]
		train_val_data = [train_data[i] for i in idxs[:n_labeled]]
		train_unlabeled_data = [train_data[i] for i in idxs]
	
	train_labeled_dataset = ALFW(train_labeled_data, transform_train, IsTest=False)
	train_unlabeled_dataset = ALFW_unlabeled(train_unlabeled_data, transform_train)
	stat_labeled_dataset = ALFW(train_labeled_data, transform_train)

	val_dataset = ALFW(val_data, transform_val)
	test_dataset = ALFW(test_data, transform_val)
	if IsVal:
		train_val_dataset = ALFW(val_data, transform_train, IsTest=False)
	else:
		train_val_dataset = ALFW(train_val_data, transform_val, IsTest=False)

	# here we compute the mean and standard deviation 
	targets = [target.unsqueeze(0) for _, target in stat_labeled_dataset]
	targets = torch.cat(targets[:],dim=0)
	mean = targets.mean(dim=0).cuda()
	std = targets.std(dim=0).cuda()

	return train_labeled_dataset, train_unlabeled_dataset, stat_labeled_dataset, train_val_dataset, val_dataset, test_dataset, mean, std

def load_dataset(data_root):



    # image_dir = os.path.join(data_root, 'Img', 'img_align_celeba_hq')
    for load_subset in ['train', 'test']:
    	with open(os.path.join(data_root, 'aflw_' + load_subset + '_images.txt'), 'r') as f:
    		images = f.read().splitlines()
    	mat = loadmat(os.path.join(data_root, 'aflw_' + load_subset + '_keypoints.mat'))
    	keypoints = mat['gt'][:, :, [1, 0]]
    	sizes = mat['hw']
    	if load_subset == 'train':
    		# put the last 10 percent of the training aside for validation
    		n_validation = int(round(0.1 * len(images)))
    		images_train = images[:-n_validation]
    		keypoints_train = keypoints[:-n_validation]
    		sizes_train = sizes[:-n_validation]
    		images_val = images[-n_validation:]
    		keypoints_val = keypoints[-n_validation:]
    		sizes_val = sizes[-n_validation:]
    	if load_subset == 'test':
    		images_test = images
    		keypoints_test = keypoints
    		sizes_test = sizes
    image_dir = os.path.join(data_root, 'output')
    train_data = [(image_dir + '/' + im, label) for (im, label) in zip(images_train, keypoints_train)]
    val_data = [(image_dir + '/' + im, label) for (im, label) in zip(images_val, keypoints_val)]
    test_data = [(image_dir + '/' + im, label) for (im, label) in zip(images_test, keypoints_test)]
    return train_data, val_data, test_data




class ALFW(object):
	def __init__(self, data, transform=None, target_transform=None, IsTest=True):
		self.data = data
		self.transform = transform 
		self.target_transform = target_transform
		self.mean = [0.4908, 0.4038, 0.3528]
		self.std = [0.0196, 0.0193, 0.0192]
		self.randomcrop = RandomTranslateWithReflect(4)
		self.IsTest=IsTest
		self.flipping = Flipping(p=0.5)
	def __getitem__(self, index):
        # """
        # Args:
        #     index (int): Index

        # Returns:
        #     tuple: (image, target) where target is index of the target class.
        # """

		im_size = 60
		# crop_percent = 0.8
		# resize_sz = int(round(im_size / crop_percent))
		# margin = int(round((resize_sz - im_size) / 2.0))

		fpath, target_ = self.data[index]
		target = np.copy(target_)
		img = Image.open(fpath).convert('RGB')
		w, h = img.size
		if self.transform is not None:
			img = self.transform(img)
			if self.IsTest is False:
				img, x, y = self.randomcrop(img)
			else:
				x, y = 0, 0

		
		target = torch.from_numpy(target).contiguous().float()

		target[:,0] = target[:,0] / w * im_size + x
		target[:,1] = target[:,1] / h * im_size + y


		target = target.view(-1)
		img = ttf.to_tensor(ttf.to_grayscale(img))


		return img, target

	def __len__(self):
		return len(self.data)

class ALFW_unlabeled(ALFW):
	def __init__(self, data, transform=None, target_transform=None):
		self.data = data
		self.transform = transform 
		self.target_transform = target_transform
		self.randomcrop = RandomTranslateWithReflect(4)
		self.mean = [0.4908, 0.4038, 0.3528]
		self.std = [0.0196, 0.0193, 0.0192]
	def __getitem__(self, index):
        # """
        # Args:
        #     index (int): Index

        # Returns:
        #     tuple: (image, target) where target is index of the target class.
        # """

		im_size = 60


		fpath, target_ = self.data[index]
		target = np.copy(target_)
		img = Image.open(fpath).convert('RGB')
		w, h = img.size
		if self.transform is not None:
			img1 = self.transform(img)
			img2 = self.transform(img)
		else: 
			img1 = img
			img2 = img

		img2, x2, y2 = self.randomcrop(img2)
		img1, x1, y1 = self.randomcrop(img1)
		img1, img2 = ttf.to_tensor(ttf.to_grayscale(img1)), ttf.to_tensor(ttf.to_grayscale(img2))

		x_random, y_random = 0, 0


		return img1, x1, y1, img2, x2, y2

	def __len__(self):
		return len(self.data)