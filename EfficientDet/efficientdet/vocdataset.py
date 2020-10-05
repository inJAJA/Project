import os
import torch
import numpy as np
import xml.etree.ElementTree as elemTree

from torch.utils.data import Dataset
import cv2

'''
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))
project_name 
    |--> Annotations
    |           |--> 00.xml
    |           |--> ... .xml
    |--> ImageSets
    |           |--> Main
    |                   |--> train.txt
    |                   |--> trainval.txt
    |--> JPEGImages
                |--> 00.png
                |--> ... .png
                                                      
'''
class VocDataset(Dataset):
    def __init__(self, root_dir, params, set='trainval',transform=None, stuff ='.jpg'):

        self.root_dir = root_dir        # root_dir = 'datasets/project/
        self.set_name = set
        self.stuff = stuff
        self.params = params
        self.transform = transform

        self.image_info()
        self.load_classes()

    def image_info(self):   # filename  # Done
        f = open(os.path.join(self.root_dir, 'ImageSets/Main/', self.set_name+'.txt'), 'r')
        self.filenames = f.readlines()

        self.image_ids = [i for i in range(len(self.filenames))]
        self.image_info = {}
        for i, name in zip(self.image_ids, self.filenames):
            self.image_info[i] = name
        return self.image_info   # image_info : {0: filename, 1: ..., }

    def load_classes(self): # Done
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

    def __len__(self):
        return len(self.image_ids)
        # return 320                      # image dataset number

    def __getitem__(self, idx):
        img = self.load_image(idx)              # same index number
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):  # Done
        filename = self.image_info[image_index].replace('\n', '')     # image_info = {0:'filename', ...}
        if os.path.exists(os.path.join(self.root_dir, 'JPEGImages')):
            path = os.path.join(self.root_dir, 'JPEGImages', filename + self.stuff)
        else:
            path = os.path.join(self.root_dir, 'images', filename + self.stuff)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        filename = self.image_info[image_index].replace('\n', '')
        if os.path.exists(os.path.join(self.root_dir, 'Annotations')):
            path = os.path.join(self.root_dir, 'Annotations', filename + '.xml')
        else:
            path = os.path.join(self.root_dir, 'annotations', filename+ '.xml')
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if not os.path.exists(path):
            return annotations

        # parse annotations
        tree = elemTree.parse(path)
        objs = tree.iter(tag='object')

        for obj in objs:
            cat = obj.find('./name').text
            x1 = int(obj.find('./bndbox/xmin').text)
            y1 = int(obj.find('./bndbox/ymin').text)
            x2 = int(obj.find('./bndbox/xmax').text)
            y2 = int(obj.find('./bndbox/ymax').text)

            bbox = [x1, y1, x2, y2]
            annotation = np.zeros((1, 5))
            annotation[0, :4] = bbox  # [[ x, y, w, h]]
            annotation[0, 4] = self.classes[cat]
            annotations = np.append(annotations, annotation, axis = 0)

        return annotations

#--------------------------------------------------------------------------------------------

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

        return sample

class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}



