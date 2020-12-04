import os
import numpy as np
from tkinter import Tcl

project = 'MS1M-ArcFace'
root = os.path.join('/home/ubuntu/data/', project)   # image path

# names = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
ids = os.listdir(root)
ids = Tcl().call('lsort', '-dict', ids)

print('ids :', len(ids))

indexs = []
with open(f'datasets/{project}_test_pair.txt', 'w') as f:
    for id in ids:
        images = os.listdir(os.path.join(root, id))
        sample = np.random.choice(images, 2, replace=False)
        label = 1
        index = f'{id}/{sample[0]} {id}/{sample[1]} {label}\n'
        f.write(index)

    for _ in range(len(ids)):
        id_choice = np.random.choice(ids, 2, replace=False)
        id_1 = id_choice[0]
        id_2 = id_choice[1]

        id1_images = os.listdir(os.path.join(root, id_1))
        id2_images = os.listdir(os.path.join(root, id_2))

        id1_image = np.random.choice(id1_images, 1)
        id2_image = np.random.choice(id2_images, 1)

        label = 0

        index = f'{id_1}/{id1_image[0]} {id_2}/{id2_image[0]} {label}\n'
        index_flip = f'{id_2}/{id2_image[0]} {id_1}/{id1_image[0]} {label}\n'

        if (index not in indexs) and (index_flip not in indexs):
            indexs.append(index)
            f.write(index)

print('indexs :', indexs)
print(f'---------[ {project}_test_pair.txt ] create done------------')