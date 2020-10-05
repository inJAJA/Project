from efficientdet.dataset import CocoDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from  mAP import main
import numpy as np
import os

class mAP_score():
    def __init__(self, true, preds, param):
        self.log_path = 'input'  # txt save path
        self.true = true                    # ground truth ( CocoDataset )
        self.preds = preds                  # detection result ( model )
        self.img_ids = self.true.image_ids  # test dataset data numbers
        self.labels = self.true.labels      # search object name ( CocoDataset )
        self.param = param                  # evaluate param

        if self.param['drop_last']:
            self.using_ids = (len(self.img_ids) // self.param['batch_size']) * self.param['batch_size']    # iter * batch_size
        else:
            self.using_ids = len(self.img_ids)

        self.ground_truth()                 # save ground truth txt
        self.detection_result()             # save detection result txt

    def __call__(self):
        self.result = main.score()
        return self.result



    def ground_truth(self):
        os.makedirs(f'{self.log_path}/ground-truth', exist_ok=True)             # if don't have 'mAP/input/ground-truth', create folder

        for i in range(self.using_ids):                                         # each number to evaluate
            annotations = self.true.load_annotations(i)                         # ground truth : bbox ( CocoDataset )
            t = open(f'{self.log_path}/ground-truth/{i}.txt', 'w')              # save txt

            for ann in annotations:
                t.write(f'{self.labels[ann[-1]]} ')     # categories
                t.write(f'{ann[0]} ')                   # x1
                t.write(f'{ann[1]} ')                   # y1
                t.write(f'{ann[2]} ')                   # x2
                t.write(f'{ann[3]}\n')                  # y2
            t.close()
        print('[ Ground truth ] Save Done')


    def detection_result(self):
        os.makedirs(f'{self.log_path}/detection-results', exist_ok=True)    # if don't have 'mAP/input/detection-results', create folder

        for i in range(self.using_ids):
            pred = self.preds[i]                            # preds = [{'rois': ..., 'class_ids': ..., 'scores': ...}, {...}, {...}, ...] ( model )
            f = open(f'{self.log_path}/detection-results/{i}.txt', 'w')

            for j in range(len(pred['rois'])):
                obj = self.labels[pred['class_ids'][j]]     # categories
                x1, y1, x2, y2 = pred['rois'][j]            # detection result : bbox
                confidence = pred['scores'][j]              # categories interest / acc

                f.write(f'{obj} ')
                f.write(f'{confidence} ')
                f.write(f'{x1} ')
                f.write(f'{y1} ')
                f.write(f'{x2} ')
                f.write(f'{y2}\n')
            f.close()
        print('[ Detection Result ] Save Done')



