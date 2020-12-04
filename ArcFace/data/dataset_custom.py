import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
import matplotlib.pyplot as plt
from torch import nn

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(1, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape
        # root              : folder path
        # data_list_file    :
        # phase             : set(trian/ test/ val)
        # input shape       : model input shape

        if phase == 'test':
            imgs = data_list_file
            self.imgs = [os.path.join(root, img) for img in imgs]

        else:
            with open(os.path.join(data_list_file), 'r') as fd:
                imgs = fd.readlines()
                imgs = [os.path.join(root, img[:-1]) for img in imgs]  # img[:-1] = remove '\n'
                self.imgs = np.random.permutation(imgs)  # After imgs copy, shuffle imgs1`



        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        # normalize = T.Normalize(mean=[0.5], std=[0.5])
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.img_shape = cv2.imread(self.imgs[0].split()[0]).shape
        # print(self.img_shape)
        # print(self.input_shape[1:])

        if self.phase == 'train':
            self.transforms = T.Compose([
                # T.Resize(self.input_shape[1:]),
                T.RandomResizedCrop(self.input_shape[1:], scale=(0.75, 1), ratio=(1, 1)),
                T.ColorJitter(brightness=[0.5, 1]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                # T.Resize(self.input_shape[1:]) if self.img_shape < self.input_shape[1:]
                # else T.RandomCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path)
        if self.input_shape[0] == 1:    # channel = 1
            data = data.convert('L')
        data = self.transforms(data)

        # data_view = data.cpu().numpy().transpose(1, 2, 0)
        # plt.imshow(data_view)
        # # plt.axis('off')
        # plt.savefig(f'datasets/test/test_m{index}.jpg', bbox_inches='tight')
        # plt.show()

        if  self.phase == 'test':
            return data.float()

        else:
            label = np.int32(splits[1])
            return data.float(), label



    def __len__(self):
        return len(self.imgs)

'''
if __name__ == '__main__':
    dataset = Dataset(root='/data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1',
                      data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      phase='test',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
'''
