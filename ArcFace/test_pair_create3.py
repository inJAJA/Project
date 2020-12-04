import os
import numpy as np
from tkinter import Tcl

project = 'MS1M-ArcFace'
root = os.path.join('/home/ubuntu/data/', project)   # image path

# names = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
ids = os.listdir(root)
ids = Tcl().call('lsort', '-dict', ids)

print('ids :', len(ids))

with open(f'datasets/{project}_test_pair3.txt', 'w') as f:
    for id in ids:
        images = os.listdir(os.path.join(root, id))
        sample = np.random.choice(images, 2, replace=False)
        label = 1
        index = f'{id}/{sample[0]} {id}/{sample[1]} {label}\n'
        f.write(index)

print(f'---------[ {project}_test_pair2.txt ] create done------------')