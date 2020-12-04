import os
import numpy as np
import yaml
import json

top = 'ex_task42'
project = 'task42_1204'
train_size = 0.8
random_seed = 13

# you need to Unnormal file ( Repair + Disposal )

obj_list = ['AirwalkMachine_discoloration', 'AirwalkMachine_surfacePeeling', 'BasketballStand_damage', 'BasketballStand_surfacePeeling',
            'BenchBack_damage', 'BenchBack_discoloration', 'BenchBack_surfacePeeling', 'Bench_damage', 'Bench_discoloration',
            'Bench_surfacePeeling', 'Bollard_damage', 'Bollard_surfacePeeling', 'BoundaryStone_damage', 'BoundaryStone_distortion',
            'BrailleBlock_damage', 'ConstructionCover_damage', 'ConstructionCover_distortion', 'DringkingFountain_damage',
            'DringkingFountain_discoloration', 'DringkingFountain_distortion', 'DringkingFountain_surfacePeeling', 'GoalPost_surfacePeeling',
            'HandicapZone_surfacePeeling', 'Manhole_damage', 'Manhole_surfacePeeling', 'MovableToilet_damage', 'MovableToilet_discoloration',
            'MovableToilet_surfacePeeling', 'Pagora_damage', 'Pagora_discoloration', 'Pagora_surfacePeeling', 'PavementBlock_damage',
            'PavementBlock_distortion', 'PavementBlock_surfacePeeling', 'PostBox_damage', 'ProtectionFence_distortion', 'PublicToilet_damage',
            'RoadSafetySign_damage', 'RoadSafetySign_surfacePeeling', 'RunningMachine_surfacePeeling', 'Seesaw_surfacePeeling', 'SignalController_damage',
            'SignalController_discoloration', 'SignalController_surfacePeeling', 'SitupNachine_surfacePeeling', 'StationShelter_surfacePeeling',
            'StationSign_damage', 'StationSign_discoloration', 'StationSign_surfacePeeling', 'StreetTreeCover_damage', 'StreetTreeCover_discoloration',
            'StreetTreeCover_distortion', 'StreetlampPole_damage', 'StreetlampPole_discoloration', 'StreetlampPole_distortion', 'TelephoneBooth_discoloration',
            'TelephoneBooth_surfacePeeling', 'TrashCan_damage', 'TrashCan_discoloration', 'TreeSupport_damage', 'Trench_damage',
            'Trench_surfacePeeling', 'TurnMachine_distortion', 'TurnMachine_surfacePeeling', 'WalkAcrossPreventionFacility_damage']

class create_datatxt:
    def __init__(self, project, train_size, random_seed):
        with open(f'/data/data/ex_task42/{project}/Unnormal.txt', 'r') as f:
            self.img_list = f.readlines()
        self.project = project
        self.train_size = train_size
        self.random_seed = random_seed

        self.dict_class()
        self.train_test_split()
        self.create_txt()

    def dict_class(self):
        path = os.path.join('/data/data/ex_task42', self.project, 'crop', 'annotations')
        names = self.img_list

        self.classes = {}
        for name in names:
            name = name.replace('.json\n', '')
            with open(os.path.join(path, name+'.json'), 'r') as file:
                annot = json.load(file)
                cat = annot['data']['labels'][0]['bbox1']['class']

                if cat not in list(self.classes.keys()):
                    self.classes[cat] = [name]
                else:
                    self.classes[cat].append(name)
        return self.classes

    def train_test_split(self):
        self.train_list = []
        self.test_list = []

        cats = sorted(list(self.classes.keys()))
        for i, cat in enumerate(cats):
            names = self.classes[cat]
            num = len(names)
            np.random.seed(random_seed)
            # choice trainset : total_images * train_size
            train = np.random.choice(names, replace=False, size = round(num * self.train_size)).tolist()  # train data choice, not overlap

            Set_total = set(names)
            Set_train = set(train)
            Set_test = Set_total.difference(Set_train)        # difference between 'num' and 'train index'
            test = list(Set_test)                     # extract 'test index' from 'num'

            self.train_list += train
            self.test_list += test
        print('Train :', len(self.train_list))
        print('Test :', len(self.test_list))

        return self.train_list, self.test_list

    def create_txt(self):
        with open(f'/data/data/ex_task42/{self.project}/crop/train.txt', 'w') as trainfile:
            for train in self.train_list:
                trainfile.write(train + '.jpg\n')

        with open(f'/data/data/ex_task42/{self.project}/crop/test.txt', 'w') as testfile:
            for test in self.test_list:
                testfile.write(test + '.jpg\n')

def create_yaml(project, obj_list):
    data00 = {'project_name':f'{project}',
            'train_set': 'train',
            'val_set':'test',
            'num_gpus':1}

    data01 = {'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]}

    data02 ={'anchors_scales': '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]',
            'anchors_ratios': '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'}

    data03 = {'obj_list': obj_list}


    # mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.


    # this is coco anchors, change it if necessary


    with open(f'projects/{project}_crop.yml', 'w') as f:
        yaml.dump(data00, f)
        yaml.dump(data01, f, default_flow_style=None)
        yaml.dump(data02, f)
        yaml.dump(data03, f)

create_datatxt(project, train_size, random_seed)
create_yaml(project, obj_list)


