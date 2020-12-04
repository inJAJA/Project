import os
import numpy as np

project = 'lfw-align-128'
root = os.path.join('/data', project)   # image path

names = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]   # only folder name
print(len(names))

falses = []

with open(f'{project}_test_pair.txt', 'w') as f:
    for name in names:
        images = os.listdir(os.path.join(root, name))
        # print('images :', images)
        # print(len(images))
        if len(images) > 1 :
            sample = np.random.choice(images, 2, replace =False)
            f.write(f'{name}/{sample[0]} {name}/{sample[1]} 1\n')

        elif len(images) == 1:
            false = f'{name}/{images[0]}'
            falses += [false]

    # print(falses)

    sample = np.random.permutation(falses)
    for i in range(int(len(sample) / 2)):
        n = i*2
        f.write(f'{sample[n]} {sample[n+1]} 0\n')

print(f'---------[ {project}_test_pair.txt ] create done------------')