import os
import numpy as np

import argparse

class create():
    def __init__(self, opt):
        self.project = opt.project
        self.set = opt.set
        self.root = os.path.join('datasets', self.project)

        self.txt()
        print('------ Done ------')

    def txt(self):
        path = os.path.join(self.root, self.set)         # train data path
        ids = os.listdir(path)

        with open(os.path.join(self.root, f'{self.set}.txt'), 'w') as f:
            # print(os.path.join(self.root, f'{self.set}.txt'))
            for label, id in enumerate(ids):
                images = os.listdir(os.path.join(path, id))
                for image in images:
                    f.write(f'{path}/{id}/{image} {label}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make data_list_file [Train / Val]')
    parser.add_argument('-p', '--project', default='test', type=str, help='save name')
    parser.add_argument('-s', '--set', default='train', type=str, help='[train / val /test]')

    args = parser.parse_args()

    create(args)


