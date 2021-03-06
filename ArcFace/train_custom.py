from __future__ import print_function
import os
from data.dataset_custom import Dataset
import torch
from torchviz import make_dot
from graphviz import Digraph
from torch.utils import data
import torch.nn.functional as F
from models.focal_loss import FocalLoss
from models.metrics import *
from models.resnet import *
import torchvision
from utils.visualizer import Visualizer
from utils import view_model
import torch
import numpy as np
import random
import time
from config.config_ms1m import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test_custom import *
from torchsummary import summary

from tensorboardX import SummaryWriter
import datetime
from tqdm.autonotebook import tqdm



def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':
    opt = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    os.makedirs(opt.checkpoints_path, exist_ok=True)  # checkpoint 폴더 없으면 생성

    project = opt.train_list.split('/')[1]
    os.makedirs(f'logs/{project}', exist_ok=True)

    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = opt.lfw_root

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet_face18':
        model = resnet_face18(opt.input_shape, use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34(input_shape=opt.input_shape)
    elif opt.backbone == 'resnet50':
        model = resnet50(input_shape=opt.input_shape)
    elif opt.backbone == 'resnet_face50':
        model = resnet_face50(opt.input_shape, use_se=opt.use_se)
    # elif opt.backbone == 'resnet_face100':
    #     model = resnet_face100(input_shape=opt.input_shape)
    elif opt.backbone == 'resnet101':
        model = resnet101(input_shape=opt.input_shape)
    elif opt.backbone == 'resnet152':
        model = resnet152(input_shape=opt.input_shape)

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # tensorboard
    save_time = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(f'logs/{project}/' + save_time +'/')


    # view_model(model, opt.input_shape)
    print('Backbone :', opt.backbone)
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()

    step = 0
    num_iter_per_epoch = len(trainloader)

    for epoch in range(opt.max_epoch):
        scheduler.step()

        model.train()

        progress_bar = tqdm(trainloader)  # progress bar
        for ii, data in enumerate(progress_bar):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)

            iters = epoch * len(trainloader) + ii

            if iters == 0:
                #     summary(model, input_size=(3, 128, 128))
                make_dot(feature).render(f'{opt.backbone}_pytorch')  # no multi-gpu!!!
                # make_dot(feature).render(f'{opt.backbone}_modify_pytorch2', format='png')
                # disgraph = make_dot(feature, params=dict(model.named_parameters()))
                # disgraph.view()

            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                # time_str = time.asctime(time.localtime(time.time()))
                # print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))

                progress_bar.set_description(
                    'Epoch: {}/{} Iter: {}/{} Loss: {:.5f} Acc: {:.5f} Speed: {:.4f}iters/s'.format(epoch, opt.max_epoch,
                                                                                                    ii,
                                                                                                    num_iter_per_epoch,
                                                                                                    loss.item(), acc,
                                                                                                    speed))

                writer.add_scalars('Loss', {'train': loss}, step)
                writer.add_scalars('Accuracy', {'train': acc}, step)
                # writer.add_graph(model, data_input)

                if opt.display:
                    visualizer.display_current_results(step, loss.item(), name='train_loss')
                    visualizer.display_current_results(step, acc, name='train_acc')

                start = time.time()
            step += 1

        os.makedirs(os.path.join(opt.checkpoints_path, save_time), exist_ok=True)
        if epoch % opt.save_interval == 0 or epoch == opt.max_epoch:
            save_model(model, os.path.join(opt.checkpoints_path, save_time), opt.backbone, epoch)

        '''
        model.eval()
        acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.input_shape, opt.test_batch_size)
        if opt.display:
            visualizer.display_current_results(epoch, acc, name='test_acc')
        '''
    writer.close()