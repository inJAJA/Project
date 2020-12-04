import shutil
import os
import numpy as np
from tkinter import Tcl
import argparse

parser = argparse.ArgumentParser(description='make data_list_file [Train / Val]')
parser.add_argument('-p', '--project', default='VGG2', type=str, help='save name')
parser.add_argument('-r', '--ratio', default=0.2, type=float, help='How many images do you want to use in validation')
args = parser.parse_args()

root = os.path.join('datasets', args.project)
ids = os.listdir(root)
print(len(ids))

min_n = 0
max_n = 0
for id in ids:
    images = os.listdir(os.path.join(root, id))

    if min_n == 0:
        min_n = len(images)
    max_n = max([max_n, len(images)])
    min_n = min([min_n, len(images)])

print('max :', max_n)
print('min :', min_n)

val_size = int(min_n * args.ratio)
print(val_size)

for id in ids:
    os.makedirs(os.path.join(root, 'train', id))
    os.makedirs(os.path.join(root, 'val', id))

    # print(os.path.join(root, id))

    images = os.listdir(os.path.join(root, id))
    images = Tcl().call('lsort', '-dict', images)  # sort

    for image in images[:-val_size]:
        shutil.copy(os.path.join(root, id, image), os.path.join(root, 'train', id, image))
    for image in images[-val_size:]:
        shutil.copy(os.path.join(root, id, image), os.path.join(root, 'val', id, image))

print('------Done------')