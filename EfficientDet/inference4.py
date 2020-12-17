# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import os
import yaml

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

project1 = 'task42_1204'
project = 'task42_1209'
number = '199'

save_time1 = '20201204-142948'
save_time2 = '20201209-140944'
compound_coef = 4
force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.4
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

params_1 = Params(f'projects/{project1}.yml')
obj_list_1 = params_1.obj_list
obj_list_1.sort()
print(obj_list_1)

params_2 = Params(f'projects/{project}_crop.yml')
obj_list_2 = params_2.obj_list
obj_list_2.sort()
print(obj_list_2)

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# model 1
model_1 = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list_1),
                             ratios=anchor_ratios, scales=anchor_scales)
model_1.load_state_dict(torch.load(f'/data/efdet/logs/{project1}/weights/{save_time1}/efficientdet-d{compound_coef}_{number}.pth', map_location='cpu'))

# model 2
model_2 = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list_2),
                             ratios=anchor_ratios, scales=anchor_scales)
model_2.load_state_dict(torch.load(f'/data/efdet/logs/{project}/crop/weights/{save_time2}/efficientdet-d{compound_coef}_{number}.pth', map_location='cpu'))

model_1.requires_grad_(False)
model_1.eval()

model_2.requires_grad_(False)
model_2.eval()

if use_cuda:
    model_1 = model_1.cuda()
    model_2 = model_2.cuda()
if use_float16:
    model_1 = model_1.half()
    model_2 = model_2.half()

def display(out_1, out_2, imgs, imshow=True, showtime = 0, imwrite=False):
    # if len(preds[i]['rois']) == 0:                    # if model dosen't detect object, not show image
    #     continue

    for img, out_1 in zip(imgs, out_1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(out_1['rois'])):
            ox1, oy1, ox2, oy2 = out_1['rois'][i].astype(np.int)
            obj_1 = obj_list_1[out_1['class_ids'][i]]
            score = float(out_1['scores'][i])
            color = color_list[get_index_label(obj_1, obj_list_1)]
            plot_one_box(img, [ox1, oy1, ox2, oy2], label=obj_1,score=score,color=color)
            print(obj_1)
            print(f'obj {i}:', ox1, oy1, ox2, oy2)
            for j in range(len(out_2[i]['rois'])):
                dx1, dy1, dx2, dy2 = out_2[i]['rois'][j].astype(np.int)
                obj_2 = obj_list_2[out_2[i]['class_ids'][j]].split('_')[-1]
                score = float(out_2[i]['scores'][j])
                plot_one_box(img, [dx1+ox1, dy1+oy1, dx2+ox1, dy2+oy1], label=obj_2, score=score,
                             color=color)
                print(obj_2)
                print('damage :',dx1, dy1, dx2, dy2 )
                print('change :', dx1+ox1, dy1+oy1, dx2+ox1, dy2+oy1)

        if imshow:
            # print(f'{img_name}')
            cv2.namedWindow('__', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('__', 1500, 1000 )
            cv2.imshow('__', img)
            # cv2.waitKey(0)
            key = cv2.waitKey(showtime)
            if key == ord('p'):
                cv2.waitKey(-1)



'''
test.txt image loop
'''
with open(os.path.join('/data/data/ex_task42/',f'{project}', 'crop', 'test.txt')) as f:
    img_list = f.readlines()
    # img_list=['201029100414AO_00000508.jpg']

for img_path in img_list:
    img_path = img_path.replace('\n', '')
    img_path = os.path.join('/data/data/ex_task42/',f'{project}', 'images', f'{img_path}')  # image load

    ori_imgs_1, framed_imgs_1, framed_metas_1 = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x_1 = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs_1], 0)
    else:
        x_1 = torch.stack([torch.from_numpy(fi) for fi in framed_imgs_1], 0)

    x_1 = x_1.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    # img_name
    if type(img_path) == list:
        img_names = [i.split('/')[-1] for i in img_path]
    else:
        img_names = [img_path.split('/')[-1]]

    with torch.no_grad():
        features, regression, classification, anchors = model_1(x_1)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out_1 = postprocess(x_1,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        # print(out_1)
    out_1 = invert_affine(framed_metas_1, out_1)
    # image crop
    images = []
    for img, pred in zip(ori_imgs_1, out_1):
        for i in range(len(pred['rois'])):
            x1, y1, x2, y2 = pred['rois'][i].astype(int)
            img_crop = img[y1:y2, x1:x2]
            images.append(img_crop)


    print('ori_image:', ori_imgs_1[0].shape)
    print(img_names)

    # 2. Damage Detection
    out_2 = []
    if images != []:
        print(len(images))
        ori_imgs_2, framed_imgs_2, framed_metas_2 = preprocess(images, max_size=input_size, video = True)
        if use_cuda:
            x_2 = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs_2], 0)
        else:
            x_2 = torch.stack([torch.from_numpy(fi) for fi in framed_imgs_2], 0)

        x_2 = x_2.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)


        with torch.no_grad():
            features, regression, classification, anchors = model_2(x_2)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out_2 = postprocess(x_2,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                0.35, iou_threshold)
        # print(out_2)
        out_2 = invert_affine(framed_metas_2, out_2)
        # cv2.namedWindow('__', cv2.WINDOW_NORMAL)
        # cv2.imshow('__', ori_imgs_2[0])
        # cv2.waitKey(2000)
        print(out_2)
    display(out_1, out_2, ori_imgs_1, imshow=True, showtime = 3000)

'''
print('running speed test...')
with torch.no_grad():
    print('test1: model inferring and postprocessing')
    print('inferring image for 10 times...')
    t1 = time.time()
    for _ in range(10):
        _, regression, classification, anchors = model(x)

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        out = invert_affine(framed_metas, out)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')
'''
    # uncomment this if you want a extreme fps test
    # print('test2: model inferring only')
    # print('inferring images for batch_size 32 for 10 times...')
    # t1 = time.time()
    # x = torch.cat([x] * 32, 0)
    # for _ in range(10):
    #     _, regression, classification, anchors = model(x)
    #
    # t2 = time.time()
    # tact_time = (t2 - t1) / 10
    # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
