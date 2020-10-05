import os
import numpy as np
import yaml

project = 'soccer_ball_data'
train_size = 0.8
random_seed = 13

def create_datasetfile(project, train_szie, random_seed):
    path = os.path.join('datasets', project, 'images')
    img_list = os.listdir(path)

    num = list(range(len(img_list)))

    np.random.seed(random_seed)
    # choice trainset : total_images * train_size
    train_index = np.random.choice(num, replace=False, size = int(len(img_list) * train_szie)).tolist()

    Set_num = set(num)
    Set_train = set(train_index)
    Set_test = Set_num.difference(Set_train)        # difference between 'num' and 'train index'
    test_index = list(Set_test)                     # extract 'test index' from 'num'


    with open(f'datasets/{project}/train.txt', 'w') as trainfile:
        for index in train_index:
            img_name = img_list[index].replace('.png', '')
            trainfile.write(img_name + '\n')

    with open(f'datasets/{project}/test.txt', 'w') as testfile:
        for index in test_index:
            img_name = img_list[index].replace('.png', '')
            testfile.write(img_name + '\n')

def create_yaml(project):
    data00 = {'project_name':f'{project}',
            'train_set': 'train',
            'val_set':'test',
            'num_gpus':1}

    data01 = {'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]}

    data02 ={'anchors_scales': '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]',
            'anchors_ratios': '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'}

    data03 = {'obj_list': ['ball']}


    # mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.


    # this is coco anchors, change it if necessary


    with open(f'projects/{project}.yml', 'w') as f:
        yaml.dump(data00, f)
        yaml.dump(data01, f, default_flow_style=None)
        yaml.dump(data02, f)
        yaml.dump(data03, f)


create_datasetfile(project, train_size, random_seed)
create_yaml(project)

