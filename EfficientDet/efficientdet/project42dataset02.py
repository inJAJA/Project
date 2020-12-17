import os
import torch
import numpy as np
import xml.etree.ElementTree as elemTree
import json

from torch.utils.data import Dataset
import cv2

'''
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))
project_name 
    |--> annotations
    |           |--> 00.json
    |           |--> ... .json
    |--> images
    |           |--> 00.jpg
    |           |--> ... .jpg
    |    
    |--> train.txt
    |--> test.txt


'''


class Project42Dataset(Dataset):
    def __init__(self, root_dir, params, set='trainval', transform=None):

        self.root_dir = root_dir  # root_dir = 'datasets/project/
        self.set_name = set
        self.params = params
        self.transform = transform

        self.label_option = 0   # 0 : {class}_{damagetype} / 1 : {class}_damage  / 2 : {damagetype}

        self.image_info()
        self.load_classes()

    def image_info(self):  # filename  # Done
        f = open(os.path.join(self.root_dir, self.set_name + '.txt'), 'r')
        self.filenames = f.readlines()

        self.image_ids = [i for i in range(len(self.filenames))]
        self.image_info = {}
        for i, name in zip(self.image_ids, self.filenames):
            self.image_info[i] = name.replace('\n', '')
        return self.image_info  # image_info : {0: filename, 1: ..., }

    def load_classes(self):  # Done
        # load class names (name -> label)
        categories = self.params.obj_list
        categories.sort()

        self.classes = {}
        for c in categories:
            self.classes[c] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    # image dataset number
    def __len__(self):
        return len(self.image_ids)
        # return 2

    def __getitem__(self, idx):
        filename = self.image_info[idx]
        img = self.load_image(filename)  # same index number
        annot = self.load_annotations(filename)
        sample = {'img': img, 'annot': annot}

        # rec = cv2.rectangle(img, (annot[:, 0], annot[:, 1]), (annot[:, 2], annot[:, 3]), (255, 0, 0), 3)
        # cv2.namedWindow(f'{filename}', cv2.WINDOW_NORMAL)
        # cv2.imshow(f'{filename}', rec)
        # cv2.waitKey(0)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, filename):  # image file root
        # filename = self.image_info[image_index]     # image_info = {0:'filename', ...}
        if os.path.exists(os.path.join(self.root_dir, 'JPEGImages')):   # image file path
            path = os.path.join(self.root_dir, 'JPEGImages', filename)
        else:
            path = os.path.join(self.root_dir, 'images', filename)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        return img.astype(np.float32) / 255.

    def load_annotations(self, filename):
        # get ground truth annotations
        # filename = self.image_info[image_index]
        # only use filename, not stuff
        if filename.endswith('.jpg'):
            filename = filename.replace('.jpg', '')
        elif filename.endswith('.png'):
            filename = filename.replace('.png', '')
        else:
            print('ERROR : Images not load ( we only load jpg, png) ')

        if os.path.exists(os.path.join(self.root_dir, 'Annotations')):
            path = os.path.join(self.root_dir, 'Annotations', filename + '.json')
        else:
            path = os.path.join(self.root_dir, 'annotations', filename + '.json')

        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if not os.path.exists(path):
            return annotations

# parse annotations
        with open(path, 'r') as file:
            f = json.load(file)
            objs = f['data']['labels']         # detection된 obj
            b = 0

            for i, obj in enumerate(objs):
                name = obj['bbox1']['class']      # class name
                bboxs = obj['bbox1']['bbox2']     # 파손 bbox data
                bbox_o = f['data']['shapes'][i+b]['bbox'] # detection된 obj의 bbox좌표

                for bbox in bboxs:
                    b += 1

                    damagetype = bbox['damagetype']
                    bbox_b = f['data']['shapes'][i+b]['bbox']

                    x_o = bbox_o['x']   # obj bbox (origin image에 대한)
                    y_o = bbox_o['y']

                    x_b = bbox_b['x']   # 파손 부분 bbox (origin image에 대한)
                    y_b = bbox_b['y']
                    w_b = bbox_b['w']
                    h_b = bbox_b['h']

                    x1 = x_b - x_o  # crop image에 대한 bbox 계
                    y1 = y_b - y_o
                    x2 = x1 + w_b
                    y2 = y1 + h_b

                    annotation = np.zeros((1, 5))
                    annotation[0, 0] = x1
                    annotation[0, 1] = y1
                    annotation[0, 2] = x2
                    annotation[0, 3] = y2

                    if self.label_option == 0:
                        annotation[0, 4] = self.classes[f'{name}_{damagetype}']    # label = [class_name]_[damagetype]
                    elif self.label_option == 1:
                        annotation[0, 4] = self.classes[f'{name}_damage']          # label = [class_name]_damage
                    elif self.label_option == 2:
                        annotation[0, 4] = self.classes[f'{damagetype}']           # label = [damagetype]

                    annotations = np.append(annotations, annotation, axis = 0)

        return annotations

# --------------------------------------------------------------------------------------------

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        # rec = cv2.rectangle(new_image, (annots[:, 0], annots[:, 1]), (annots[:, 2], annots[:, 3]), (255, 0, 0), 3)
        # cv2.namedWindow('rec', cv2.WINDOW_NORMAL)
        # cv2.imshow('rec', rec)
        # cv2.waitKey(0)

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots),
                'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

            # rec = cv2.rectangle(np.float32(image), (annots[:, 0], annots[:, 1]), (annots[:, 2], annots[:, 3]), (255, 0, 0), 3)
            # cv2.namedWindow('rec', cv2.WINDOW_NORMAL)
            # cv2.imshow('rec', rec)
            # cv2.waitKey(0)

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        # rec = cv2.rectangle(image, (annots[:, 0], annots[:, 1]), (annots[:, 2], annots[:, 3]), (255, 0, 0), 3)
        # cv2.namedWindow('rec', cv2.WINDOW_NORMAL)
        # cv2.imshow('rec', rec)
        # cv2.waitKey(0)

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}