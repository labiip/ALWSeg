import sys
sys.path.append('..')
from utils import hdf5_reader
from skimage.transform import resize
from torch.utils.data import Dataset
import torch
import numpy as np
import random
# from label_propagation import label_propagation

from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import numpy as np
from skimage.segmentation import felzenszwalb, slic
import torch
import torch.nn.functional as F
from skimage.segmentation import watershed
from scipy.ndimage import median_filter
import cv2


def label_propagation(image, scribble, dataset='ACDC', method='felzenszwalb'):
    """
    image: BCHWD 0~1 torch.tensor
    scribble: BCHWD torch.tensor
    """

    # img 归一化
    image = (image - image.min()) / (image.max() - image.min()) * 255
    # array
    image = image.astype(np.uint8)
    scribble = np.array(scribble)
    if dataset == 'ACDC':
        scribble[scribble == 4] = 0
    elif dataset == 'CHAOS':
        scribble[scribble == 5] = 0
    elif dataset == 'VS':
        scribble[scribble == 2] = 0
    elif dataset == 'RUIJIN':
        scribble[scribble == 2] = 0

    H, W = image.shape
    pseudo_mask = np.zeros(image.shape)
    su_mask = np.zeros(image.shape)

    ## B
    x, y = np.where(scribble[:, :] != 0)
    # if x.size == 0:
    #     continue
    if x.size == 0:
        pseudo_mask = np.zeros(image.shape)
        su_mask = np.zeros(image.shape)
    else:
        x_min, x_max, y_min, y_max = max((x.min() - 10), 0), min((x.max() + 10), scribble.shape[0]), \
                                     max((y.min() - 10), 0), min((y.max() + 10), scribble.shape[1])
        img_fg = image[x_min:x_max, y_min:y_max]
        scr_fg = scribble[x_min:x_max, y_min:y_max]
        H_fg, W_fg = img_fg.shape
        pseudo_fg = np.zeros(img_fg.shape)
        su_fg = np.zeros(img_fg.shape)

        ## d
        img = img_fg[:, :]
        scr = scr_fg[:, :]
        su = felzenszwalb(img, scale=50, sigma=0.5, min_size=30)

        su_fg[:, :] = su
        scribble_value_list = np.unique(scr)
        scribble_value_ignore = 0
        for scribble_value in scribble_value_list:
            if scribble_value != scribble_value_ignore:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                tmp = scr.copy()
                tmp[scr == scribble_value] = 1
                tmp[scr != scribble_value] = 0
                if dataset == 'ACDC':
                    valid_mask = cv2.dilate(tmp, kernel, iterations=1)
                if 'CHAOS' in dataset:
                    valid_mask = cv2.dilate(tmp, kernel, iterations=5)
                if dataset == 'VS':
                    valid_mask = cv2.dilate(tmp, kernel, iterations=1)
                if dataset == 'RUIJIN':
                    valid_mask = cv2.dilate(tmp, kernel, iterations=1)
                supervoxel_under_scribble_marking = np.unique(su[scr == scribble_value])
                tmp_mask = np.zeros(img.shape)
                for i in supervoxel_under_scribble_marking:
                    tmp_mask[su == i] = scribble_value
                if dataset != 'VS':
                    tmp_mask *= valid_mask
                for h in range(H_fg):
                    for w in range(W_fg):
                        if tmp_mask[h, w] != 0:
                            pseudo_fg[h, w] = tmp_mask[h, w]
        pseudo_mask[x_min:x_max, y_min:y_max] = pseudo_fg
        su_mask[x_min:x_max, y_min:y_max] = su_fg
    return pseudo_mask, su_mask
    #
    #
    # x_min, x_max, y_min, y_max = max((x.min() - 10), 0), min((x.max() + 10), scribble.shape[0]), \
    #                                                max((y.min() - 10), 0), min((y.max() + 10), scribble.shape[1])
    # img_fg = image[x_min:x_max, y_min:y_max]
    # scr_fg = scribble[x_min:x_max, y_min:y_max]
    # H_fg, W_fg = img_fg.shape
    # pseudo_fg = np.zeros(img_fg.shape)
    # su_fg = np.zeros(img_fg.shape)
    #
    # ## d
    # img = img_fg[:, :]
    # scr = scr_fg[:, :]
    # su = felzenszwalb(img, scale=50, sigma=0.5, min_size=30)
    #
    # su_fg[:, :] = su
    # scribble_value_list = np.unique(scr)
    # scribble_value_ignore = 0
    # for scribble_value in scribble_value_list:
    #     if scribble_value != scribble_value_ignore:
    #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #         tmp = scr.copy()
    #         tmp[scr == scribble_value] = 1
    #         tmp[scr != scribble_value] = 0
    #         if dataset == 'ACDC':
    #             valid_mask = cv2.dilate(tmp, kernel, iterations=1)
    #         if 'CHAOS' in dataset:
    #             valid_mask = cv2.dilate(tmp, kernel, iterations=5)
    #         if dataset == 'VS':
    #             valid_mask = cv2.dilate(tmp, kernel, iterations=1)
    #         if dataset == 'RUIJIN':
    #             valid_mask = cv2.dilate(tmp, kernel, iterations=1)
    #         supervoxel_under_scribble_marking = np.unique(su[scr == scribble_value])
    #         tmp_mask = np.zeros(img.shape)
    #         for i in supervoxel_under_scribble_marking:
    #             tmp_mask[su == i] = scribble_value
    #         if dataset != 'VS':
    #             tmp_mask *= valid_mask
    #         for h in range(H_fg):
    #             for w in range(W_fg):
    #                 if tmp_mask[h, w] != 0:
    #                     pseudo_fg[h, w] = tmp_mask[h, w]
    # pseudo_mask[x_min:x_max, y_min:y_max] = pseudo_fg
    # su_mask[x_min:x_max, y_min:y_max] = su_fg


    # for b in range(B):
    #     # 找前景区域，只在前景区域寻找pseudo label
    #     x, y = np.where(scribble[b, :, :] != 0)
    #     if x.size == 0:
    #         continue
    #     x_min, x_max, y_min, y_max = max((x.min() - 10), 0), min((x.max() + 10), scribble.shape[1]), \
    #                                                max((y.min() - 10), 0), min((y.max() + 10), scribble.shape[2])
    #
    #     img_fg = image[b, 0, x_min:x_max, y_min:y_max]
    #     scr_fg = scribble[b, x_min:x_max, y_min:y_max]
    #     H_fg, W_fg = img_fg.shape
    #     pseudo_fg = np.zeros(img_fg.shape)
    #     su_fg = np.zeros(img_fg.shape)
    #
    #     for d in range(1):
    #         img = img_fg[:, :]
    #         scr = scr_fg[:, :]
    #         su = felzenszwalb(img, scale=50, sigma=0.5, min_size=30)
    #
    #         su_fg[:, :] = su
    #         scribble_value_list = np.unique(scr)
    #         scribble_value_ignore = 0
    #         for scribble_value in scribble_value_list:
    #             if scribble_value != scribble_value_ignore:
    #                 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #                 tmp = scr.copy()
    #                 tmp[scr == scribble_value] = 1
    #                 tmp[scr != scribble_value] = 0
    #                 if dataset == 'ACDC':
    #                     valid_mask = cv2.dilate(tmp, kernel, iterations=1)
    #                 if 'CHAOS' in dataset:
    #                     valid_mask = cv2.dilate(tmp, kernel, iterations=5)
    #                 if dataset == 'VS':
    #                     valid_mask = cv2.dilate(tmp, kernel, iterations=1)
    #                 if dataset == 'RUIJIN':
    #                     valid_mask = cv2.dilate(tmp, kernel, iterations=1)
    #                 supervoxel_under_scribble_marking = np.unique(su[scr == scribble_value])
    #                 tmp_mask = np.zeros(img.shape)
    #                 for i in supervoxel_under_scribble_marking:
    #                     tmp_mask[su == i] = scribble_value
    #                 if dataset != 'VS':
    #                     tmp_mask *= valid_mask
    #                 for h in range(H_fg):
    #                     for w in range(W_fg):
    #                         if tmp_mask[h, w] != 0:
    #                             pseudo_fg[h, w] = tmp_mask[h, w]
    #     pseudo_mask[b, 0, x_min:x_max, y_min:y_max] = pseudo_fg
    #     su_mask[b, 0, x_min:x_max, y_min:y_max] = su_fg
    #
    # return pseudo_mask, su_mask


class Trunc_and_Normalize(object):
    '''
    truncate gray scale and normalize to [0,1]
    '''
    def __init__(self, scale, channels):
        self.scale = scale
        self.channels = channels
        # assert len(self.scale) == 2, 'scale error'

    def __call__(self, sample):
        image = sample['image']
        
        # gray truncation
        if self.scale is not None:
            assert len(self.scale) == 2, 'scale error'
            if np.max(image) > 1.0 or np.min(image) != 0: # if un-normalized
                image = image - self.scale[0]
                gray_range = self.scale[1] - self.scale[0]
                image[image < 0] = 0
                image[image > gray_range] = gray_range
                image = image / gray_range
        else:
        # min-max normalization
            if np.max(image) > 1.0 and (np.max(image) > np.min(image)): # un-normalized during data preprocessing
                if self.channels == 1:
                    image = (image - np.min(image)) / (np.max(image) - np.min(image))
                else:
                    for i in range(self.channels):
                        image[i] = (image[i] - np.min(image[i])) / (np.max(image[i]) - np.min(image[i]))
        
        sample['image'] = image
        return sample


class CropResize(object):
    '''
    Data preprocessing.
    Adjust the size of input data to fixed size by cropping and resize
    Args:
    - dim: tuple of integer, fixed size
    - crop: single integer, factor of cropping, H/W ->[:,crop:-crop,crop:-crop]
    '''
    def __init__(self, dim=None, num_class=2, crop=0, channels=1):
        self.dim = dim
        self.num_class = num_class
        self.crop = crop
        self.channels = channels

    def __call__(self, sample):

        # image: numpy array
        # label: numpy array
        image = sample['image']
        label = sample['label']

        mm = 1 if self.channels > 1 else 0
        # crop
        if self.crop != 0:
            if mm:
                image = image[:, self.crop:-self.crop, self.crop:-self.crop]
                label = label[:, self.crop:-self.crop, self.crop:-self.crop]
            else:
                image = image[self.crop:-self.crop, self.crop:-self.crop]
                label = label[self.crop:-self.crop, self.crop:-self.crop]
        # resize
        if self.dim is not None and label.shape != self.dim:
            if mm:
                temp_image = np.empty((self.channels,) + self.dim, dtype=np.float32)
                for i in range(self.channels):
                    temp_image[i] = resize(image[i], self.dim, order=1, anti_aliasing=True)
                image = temp_image
            else:
                image = resize(image, self.dim, order=1, anti_aliasing=True)
            
            temp_label = np.zeros(self.dim,dtype=np.float32)
            for z in range(1, self.num_class):
                roi = resize((label == z).astype(np.float32),self.dim, order=0, mode='constant')
                temp_label[roi >= 0.5] = z
            label = temp_label

        sample['image'] = image
        sample['label'] = label
        return sample

def random_rot_flip(image, label=None, edge=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        # edge = np.rot90(edge, k)
        # edge = np.flip(edge, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    # edge = ndimage.rotate(edge, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        # image, label, edge = sample["image"], sample["label"], sample["edge"]
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            # image, label, edge = random_rot_flip(image, label, edge)
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            # image, label, edge = random_rotate(image, label, edge)
            image, labeL = random_rotate(image, label)
        # print(image.shape)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # edge = zoom(edge, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # label = torch.from_numpy(label.astype(np.uint8))

        # edge = torch.from_numpy(edge.astype(np.uint8))
        # sample = {"image": image, "label": label, "edge": edge}
        # sample = {"image": image, "label": label}
        sample['image'] = image
        sample['label'] = label
        return sample


class To_Tensor0(object):
    '''
    Convert the data in sample to torch Tensor.
    Args:
    - n_class: the number of class
    '''

    def __init__(self, num_class=2, channels=1):
        self.num_class = num_class
        self.channels = channels

    def __call__(self, sample):

        image = sample['image']
        label = sample['label']
        label0 = sample['label']

        mm = 1 if self.channels > 1 else 0

        if mm:
            new_image = image[:self.channels, ...]
        else:
            new_image = np.expand_dims(image, axis=0)
        # expand dims
        new_label = np.empty((self.num_class,) + label.shape, dtype=np.float32)
        for z in range(1, self.num_class):
            temp = (label == z).astype(np.float32)
            new_label[z, ...] = temp
        new_label[0, ...] = np.amax(new_label[1:, ...], axis=0) == 0

        # convert to Tensor
        sample['image'] = torch.from_numpy(new_image.astype(np.float32))
        sample['label'] = torch.from_numpy(new_label)
        # sample['label'] = torch.from_numpy(label0)
        return sample


class To_Tensor1(object):
    '''
    Convert the data in sample to torch Tensor.
    Args:
    - n_class: the number of class
    '''
    def __init__(self,num_class=2, channels=1):
        self.num_class = num_class
        self.channels = channels

    def __call__(self,sample):

        image = sample['image']
        label = sample['label']
        label0 = sample['label']
        
        mm = 1 if self.channels > 1 else 0

        if mm:
            new_image = image[:self.channels,...]
        else:
            new_image = np.expand_dims(image, axis=0)
        # expand dims
        new_label = np.empty((self.num_class,) + label.shape, dtype=np.float32)
        for z in range(1, self.num_class):
            temp = (label==z).astype(np.float32)
            new_label[z,...] = temp
        new_label[0,...] = np.amax(new_label[1:,...],axis=0) == 0   
    
        # convert to Tensor
        sample['image'] = torch.from_numpy(new_image.astype(np.float32))
        sample['label'] = torch.from_numpy(new_label)
        sample['label_s'] = torch.from_numpy(label0.copy())
        return sample



class DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''
    def __init__(self, path_list, roi_number=None, num_class=2, transform=None, repeat_factor=1.0):

        self.path_list = path_list
        self.roi_number = roi_number
        self.num_class = num_class
        self.transform = transform
        self.repeat_factor = repeat_factor


    def __len__(self):
        # return len(self.path_list)
        return int(len(self.path_list)*self.repeat_factor)


    def __getitem__(self,index):
        # Get image and label
        # image: (D,H,W) or (H,W) 
        # label: same shape with image, integer, [0,1,...,num_class]
        index = index % len(self.path_list)
        image = hdf5_reader(self.path_list[index],'image')
        label = hdf5_reader(self.path_list[index], 'label')
        ## pseudo generate
        # label0 = hdf5_reader(self.path_list[index], 'label')
        # label, su_mask = label_propagation(image, label0, 'ACDC')
        ##

        if self.roi_number is not None:
            if isinstance(self.roi_number,list):
                tmp_mask = np.zeros_like(label,dtype=np.float32)
                assert self.num_class == len(self.roi_number) + 1
                for i, roi in enumerate(self.roi_number):
                    tmp_mask[label == roi] = i+1
                label = tmp_mask
            else:
                assert self.num_class == 2
                label = (label==self.roi_number).astype(np.float32) 
        # print('load_label.shape:', label.shape)
        sample = {'image':image, 'label':label}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class DataGeneratorval(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''
    def __init__(self, path_list, roi_number=None, num_class=2, transform=None, repeat_factor=1.0):

        self.path_list = path_list
        self.roi_number = roi_number
        self.num_class = num_class
        self.transform = transform
        self.repeat_factor = repeat_factor


    def __len__(self):
        # return len(self.path_list)
        return int(len(self.path_list)*self.repeat_factor)


    def __getitem__(self,index):
        # Get image and label
        # image: (D,H,W) or (H,W)
        # label: same shape with image, integer, [0,1,...,num_class]
        index = index % len(self.path_list)
        image = hdf5_reader(self.path_list[index],'image')
        label = hdf5_reader(self.path_list[index],'label')
        ## pseudo generate
        # label, su_mask = label_propagation(image, label0, 'ACDC')
        ##

        if self.roi_number is not None:
            if isinstance(self.roi_number,list):
                tmp_mask = np.zeros_like(label,dtype=np.float32)
                assert self.num_class == len(self.roi_number) + 1
                for i, roi in enumerate(self.roi_number):
                    tmp_mask[label == roi] = i+1
                label = tmp_mask
            else:
                assert self.num_class == 2
                label = (label==self.roi_number).astype(np.float32)
        # print('load_label.shape:', label.shape)
        sample = {'image':image, 'label':label}
        # Transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

class BalanceDataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of two lists, one includes positive samples, and the other includes negative samples
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''
    def __init__(self,
                 path_list=None,
                 roi_number=None,
                 num_class=2,
                 transform=None,
                 factor=0.3):

        self.path_list = path_list
        self.roi_number = roi_number
        self.num_class = num_class
        self.transform = transform
        self.factor = factor


    def __len__(self):
        assert isinstance(self.path_list,list)
        assert len(self.path_list) == 2
        return sum([len(case) for case in self.path_list])

    def __getitem__(self, index):
        # balance sampler
        item_path = random.choice(self.path_list[int(random.random() < self.factor)])
        # Get image and mask
        image = hdf5_reader(item_path,'image')
        label = hdf5_reader(item_path,'label')

        if self.roi_number is not None:
            if isinstance(self.roi_number,list):
                tmp_mask = np.zeros_like(label,dtype=np.float32)
                assert self.num_class == len(self.roi_number) + 1
                for i, roi in enumerate(self.roi_number):
                    tmp_mask[label == roi] = i+1
                label = tmp_mask
            else:
                assert self.num_class == 2
                label = (label == self.roi_number).astype(np.float32)

        sample = {'image': image, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)


        return sample