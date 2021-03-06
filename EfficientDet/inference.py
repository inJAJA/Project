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

project = 'task42_1130'
number = '199'
compound_coef = 4
force_input_size = None  # set None to use default size
# img_path = 'datasets/shape/val/900.jpg''
# img_path = os.path.join('datasets', project, 'test03.jpg')      # image load
img_path = os.path.join('/home/jain/Downloads','test', 'pavement1.jpg')      # image load
# img_path = os.path.join('/data/data',f'{project}','images', '201029100414AO_00002871.jpg')      # image load
# img_path = os.path.join('/data/data','ex_task42',f'{project}','images','55db7caa-8037-4d80-98a9-33d77346aee4.jpg')
print(img_path)

# img_name
if type(img_path) == list:
    img_names = [i.split('/')[-1] for i in img_path]
else:
    img_names = [img_path.split('/')[-1]]

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.3
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

params = Params(f'projects/{project}.yml')
obj_list = params.obj_list
obj_list.sort()
print(obj_list)


color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
# model.load_state_dict(torch.load(f'logs/{project}/efficientdet-d{compound_coef}_{number}.pth', map_location='cpu'))
# model.load_state_dict(torch.load(f'/data/efdet/logs/{project}/efficientdet-d{compound_coef}_{number}.pth', map_location='cpu'))
model.load_state_dict(torch.load(f'/data/efdet/logs/{project}/efficientdet-d{compound_coef}_{number}.pth', map_location='cpu'))

model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)
    print(out)

def display(preds, imgs, imshow=True, imwrite=True):
    for i, img_name in zip(range(len(imgs)), img_names):
        # if len(preds[i]['rois']) == 0:                    # if model dosen't detect object, not show image
        #     continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])
            print(obj)

        if imwrite:
            img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_{img_name}', img)

        if imshow:
            img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
            cv2.namedWindow(f'{img_name}', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow(f'{img_name}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


out = invert_affine(framed_metas, out)
display(out, ori_imgs, imshow=True, imwrite=False)
print("Done")
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
