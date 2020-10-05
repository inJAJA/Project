# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import torch
from torch.backends import cudnn
import argparse

from backbone import EfficientDetBackbone
import os
import yaml

from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, collater
from mAP.utils import mAP_score

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--project', type=str, default='shape', help = 'project name')
parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
parser.add_argument('-f', '--force_input_size', default=None, help='set None to use default size')
parser.add_argument('-w', '--weights', type=str, help='number of weights to use' )
parser.add_argument('--batch_size', type=int, default=16, help='The number of images per batch among all devices')
parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
opt = parser.parse_args()

opt.weights = '499_14000'

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

# project informations
params = Params(f'projects/{opt.project}.yml')

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['rectangle','circle']

val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[opt.compound_coef] if opt.force_input_size is None else opt.force_input_size

val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]))
val_generator = DataLoader(val_set, **val_params)


model = EfficientDetBackbone(compound_coef=opt.compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'logs/{opt.project}/efficientdet-d{opt.compound_coef}_{opt.weights}.pth', map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

result = []
for iter, data in enumerate(val_generator):
    with torch.no_grad():
        features, regression, classification, anchors = model(data['img'].cuda())   # cuda & cup type match
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(data['img'],
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        # print(out)
        result += out
        # print(result)
    print(len(result))

mAP = mAP_score(val_set, result, val_params)
print(mAP())
