# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
from models import *
from models.focal_loss import *
from models.metrics import *
from models.resnet import *
import torch
import numpy as np
import time
from config.config_DeepGlint import  Config
from torch.nn import DataParallel
from torchviz import make_dot

def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path, input_shape = (1, 128, 128)):
    if input_shape[0] == 1:
        image = cv2.imread(img_path, 0)
    else:
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

    # cv2.imshow('a', image)
    # cv2.waitKey(0)

    if image.shape[1] < input_shape[1]:
        image = cv2.resize(image, input_shape[1:], interpolation = cv2.INTER_CUBIC)      # test image resize
    elif image.shape[1] > input_shape[1]:
        image = cv2.resize(image, input_shape[1:], interpolation = cv2.INTER_AREA)
    else:
        pass

    if image is None:
        return None
    image = np.stack((image, np.fliplr(image)), axis = 0)
    image = image.transpose((0, 3, 1, 2))
    # image = image[ :, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5

    return image


def get_featurs(model, test_list, input_shape, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path, input_shape)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))

            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, input_shape, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, input_shape=input_shape, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)                     # key = file path / value = feature(1024, )
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc, th


if __name__ == '__main__':

    opt = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34(input_shape=opt.input_shape)
    elif opt.backbone == 'resnet50':
        model = resnet50(input_shape=opt.input_shape)
    elif opt.backbone == 'resnet_face50':
        model = resnet_face50(opt.input_shape, use_se=opt.use_se)

    opt.test_model_path = 'checkpoints/20201023-143633/resnet_face50_23.pth'
    # model = DataParallel(model)
    model = DataParallel(model)
    # with open(os.path.join('./weight0.txt'), 'w') as f:
    #     f.write(f'{torch.load(opt.test_model_path)}\n')

    # # load_model(model, opt.test_model_path)
    model.load_state_dict(torch.load(opt.test_model_path))
    # with open(os.path.join('./weight2.txt'), 'w') as f:
    #     f.write(f'{model.state_dict()}\n')

    model.to(torch.device("cuda"))

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    model.eval()
    lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.input_shape, opt.test_batch_size)




