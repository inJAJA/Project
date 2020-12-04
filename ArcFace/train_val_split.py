import shutil
import os
import numpy as np
from tkinter import Tcl
import argparse

parser = argparse.ArgumentParser(description='make data_list_file [Train / Val]')
parser.add_argument('-p', '--project', default='DeepGlint', type=str, help='save name')
parser.add_argument('-r', '--ratio', default=0.2, type=float, help='How many images do you want to use in validation')
parser.add_argument('-s', '--set', default=['train', 'val'])
args = parser.parse_args()

root = os.path.join('/home/ubuntu/data/', args.project)
# root = os.path.join('datasets', args.project)

ids = os.listdir(root)
ids = Tcl().call('lsort', '-dict', ids)
print('Total_ids :', len(ids))

min_n = 0
max_n = 0
cn = 0
for id in ids:
    images = os.listdir(os.path.join(root, id))

    if min_n == 0:
        min_n = len(images)
    max_n = max([max_n, len(images)])
    min_n = min([min_n, len(images)])

    if len(images) < 11:
        # print('id :', id)
        cn += 1

print('max :', max_n)
print('min :', min_n)
print('smaller than 11 :', cn)
val_size = int(min_n * args.ratio)

if val_size == 0:
    val_size = 1
print('val_size :', val_size)

def create_txt(set, ids):
    path = os.path.join('datasets', args.project)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, f'{set}.txt'), 'w') as f:
        for label, id in enumerate(ids):
            if (val_size == 0) and (set == 'val'):
                break

            images = os.listdir(os.path.join(root, id))
            images = Tcl().call('lsort', '-dict', images)  # sort

            if val_size != 0:
                if set == 'train':
                    images = images[:-val_size]
                elif set == 'val':
                    images = images[-val_size:]

            for image in images:
                f.write(f'{id}/{image} {label}\n')

    return print(f'------{set}.txt created------')

for set in args.set:
    create_txt(set, ids)
