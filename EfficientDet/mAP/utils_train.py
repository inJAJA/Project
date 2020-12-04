
import numpy as np
import os
from mAP.main import score

class mAP_score():
    def __init__(self, trues, preds, labels):
        self.log_path = 'mAP/input'
        self.trues = trues.detach().cpu().numpy()    # annot = device('cuda')
        self.preds = preds
        self.labels = labels

        self.ground_truth()
        self.detection_result()
        try:
            self.results = score()
        except ZeroDivisionError:
            self.results = {'mAP': float(0)}


    def ground_truth(self):
        os.makedirs(f'{self.log_path}/ground-truth', exist_ok=True)

        for i, true in enumerate(self.trues):
            t = open(f'{self.log_path}/ground-truth/{i}.txt', 'w')

            for ann in true:
                if ann[-1] in self.labels:
                    t.write(f'{self.labels[ann[-1]]} ')     # categories
                    t.write(f'{ann[0]} ')                   # x1
                    t.write(f'{ann[1]} ')                   # y1
                    t.write(f'{ann[2]} ')                   # x2
                    t.write(f'{ann[3]}\n')                  # y2
            t.close()

    def detection_result(self):
        os.makedirs(f'{self.log_path}/detection-results', exist_ok=True)

        for i, pred in enumerate(self.preds):
            f = open(f'{self.log_path}/detection-results/{i}.txt', 'w')
            for j in range(len(pred['rois'])):
                obj = self.labels[pred['class_ids'][j]]  # categories
                x1, y1, x2, y2 = pred['rois'][j]  # detection result : bbox
                confidence = pred['scores'][j]  # categories interest / acc

                f.write(f'{obj} ')
                f.write(f'{confidence} ')
                f.write(f'{x1} ')
                f.write(f'{y1} ')
                f.write(f'{x2} ')
                f.write(f'{y2}\n')
            f.close()
