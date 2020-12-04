import os
import json

"""
status check
"""


project = 'task42_1204'
top = 'ex_task42'
root = f"/data/data/{top}/{project}/annotations"
annots = os.listdir(root)
print(len(annots))   # 587

n = 0
r = 0
d = 0
u = 0

normal = open(f'/data/data/{top}/{project}/Normal.txt', 'w')
Repair = open(f'/data/data/{top}/{project}/Repair.txt', 'w')
Disposal = open(f'/data/data/{top}/{project}/Disposal.txt', 'w')
unnormal = open(f'/data/data/{top}/{project}/Unnormal.txt', 'w')

damagetype_list = []
for name in annots:
    path = os.path.join(root, name)
    with open(path, 'r') as annot:
        f = json.load(annot)
        status = f['data']['labels'][0]['bbox1']['status']
        if status == 'Normal':
            n += 1
            normal.write(f'{name}\n')
        elif status == 'Repair':
            r += 1
            Repair.write(f'{name}\n')
        else:
            d += 1
            Disposal.write(f'{name}\n')

        if status != 'Normal':
            u += 1
            unnormal.write(f'{name}\n')

        labels = f['data']['labels']
        for label in labels:
            bbox2 = label['bbox1']['bbox2']
            for bbox in bbox2:
                damagetype = bbox['damagetype']
                if damagetype not in damagetype_list:
                    damagetype_list += [damagetype]

normal.close()
Repair.close()
Disposal.close()
unnormal.close()

print('Normal : ', n)
print('Repair : ', r)
print('Disposal : ', d)
print('Unnormal : ', u)

print('damagetype_list :\n', damagetype_list)
# Normal :  524
# Repair :  43
# Broken :  20


"""
# Only CLASS Check #

Train or Test
data class check
"""


project = 'task42_1204'
root = '/data/data/ex_task42/'

check = 'test'

with open(f'{root}/{project}/{check}.txt') as f:
    list = f.readlines()

data_list = [name.replace('.jpg\n', '') for name in list]
# print(data_list)

obj_list = ['Pagora', 'BasketballStand', 'TreeSupport', 'StreetlampPole', 'SignalController', 'PavementBlock', 'BoundaryStone',
            'RoadSafetySign', 'TrashCan', 'Bollard', 'BrailleBlock', 'TelephoneBooth', 'HandicapZone', 'DringkingFountain',
            'SignalPole', 'RunningMachine', 'MovableToilet', 'ConstructionCover', 'TurnMachine', 'AirwalkMachine', 'PostBox',
            'StreetTreeCover', 'BenchBack', 'StationSign', 'SoundproofWalls', 'Bench', 'WalkAcrossPreventionFacility', 'Manhole',
            'PublicToilet', 'ProtectionFence', 'GoalPost', 'DirectionalSign', 'FlowerStand', 'StationShelter', 'ShadeCanopy', 'Slide',
            'BicycleRack', 'Trench', 'SubwayVentilation', 'Seesaw', 'Swing', 'SitupNachine']

classes = []
count = {}
for obj in obj_list:
    count[obj] = 0

for data in data_list:
    path = f'{root}/{project}/annotations/{data}.json'
    with open(path, 'r') as annot:
        f = json.load(annot)
        data_class = f['data']['labels'][0]['bbox1']['class']
        if data_class not in classes:
            classes.append(data_class)

        count[data_class] += 1

for num in sorted(count.items()):
    print(num)
print('-----------------------')

Set_train = set(obj_list)
Set_test = set(classes)
exception= Set_train.difference(Set_test)        # difference between 'num' and 'train index'

## Information
print(f'Information : {project}')
print(f'Data : {check}')
# print('Class :', sorted(classes))
print('Total :', len(data_list))
print('Class :', len(classes))
print('Exception :', sorted(exception))



"""
# Only DAMGE class Check #

Train or Test
data damge class check
"""
project = 'task42_1130'
root = '/data/data/ex_task42/'

check = 'train'

with open(f'{root}/{project}/crop/{check}.txt') as f:
    list = f.readlines()

data_list = [name.replace('.jpg\n', '') for name in list]
# print(data_list)

obj_list = ['BasketballStand_surfacePeeling', 'SignalController_discoloration', 'SignalController_surfacePeeling', 'PavementBlock_damage',
            'TelephoneBooth_discoloration', 'HandicapZone_surfacePeeling', 'BoundaryStone_damage', 'StreetTreeCover_discoloration',
            'StationSign_surfacePeeling', 'PavementBlock_surfacePeeling', 'Bench_surfacePeeling', 'StreetlampPole_distortion', 'ProtectionFence_distortion',
            'GoalPost_surfacePeeling', 'StationSign_damage', 'SignalController_damage', 'TreeSupport_damage', 'ConstructionCover_damage',
            'StreetTreeCover_distortion', 'Bollard_damage', 'AirwalkMachine_surfacePeeling', 'AirwalkMachine_discoloration', 'PavementBlock_distortion',
            'DringkingFountain_discoloration', 'BrailleBlock_damage', 'DringkingFountain_surfacePeeling', 'Bench_damage', 'WalkAcrossPreventionFacility_damage',
            'BenchBack_surfacePeeling', 'TurnMachine_surfacePeeling', 'MovableToilet_damage', 'Pagora_discoloration', 'StreetlampPole_damage',
            'StreetTreeCover_damage', 'Manhole_surfacePeeling', 'StreetlampPole_discoloration', 'MovableToilet_surfacePeeling', 'Manhole_damage',
            'TrashCan_damage', 'TelephoneBooth_surfacePeeling', 'MovableToilet_discoloration', 'Bench_discoloration', 'RoadSafetySign_damage',
            'BenchBack_discoloration', 'DringkingFountain_distortion', 'DringkingFountain_damage', 'StationSign_discoloration', 'Bollard_surfacePeeling',
            'BoundaryStone_distortion', 'ConstructionCover_distortion', 'RoadSafetySign_surfacePeeling', 'PostBox_damage', 'TrashCan_discoloration',
            'PublicToilet_damage', 'StationShelter_surfacePeeling', 'TurnMachine_distortion', 'SitupNachine_surfacePeeling', 'Pagora_damage']

classes = []
count = {}
damage_num = 0
for obj in obj_list:    # create FORM
    count[obj] = 0

for data in data_list:
    path = f'{root}/{project}/crop/annotations/{data}.json'

    with open(path, 'r') as annot:
        f = json.load(annot)
        data_class = f['data']['labels'][0]['bbox1']['class']
        damages = f['data']['labels'][0]['bbox1']['bbox2']

        for damage in damages:
            damagetype = damage['damagetype']
            label = f'{data_class}_{damagetype}'

            if label not in classes:
                classes.append(label)

            count[label] += 1
            damage_num += 1
for num in sorted(count.items()):
    print(num)
print('-----------------------')

Set_train = set(obj_list)
Set_test = set(classes)
exception= Set_train.difference(Set_test)        # difference between 'num' and 'train index'

## Information
print(f'Information : {project}')
print(f'Data        : {check}')
# print('Class :', sorted(classes))
print('Total data  :', len(data_list))
print('Damage data :', damage_num)
print('Obj list    :', len(obj_list))
print('Class       :', len(classes))
print('Exception   :', sorted(exception))
