import os
import json
from glob import glob
import cv2

'''
# 시설물 class check
'''


files = glob(r'/data/data/ex_task42/task42_1130/annotations'+'/*.json')

print('list :', files)
print('list length :', len(files))  # 380

# with open(files, 'r') as annot:
#     annotation = json.load(annot)
#     # print(annotation)
#     class_name = annotation['data']['labels'][0]['bbox1']['class']
#     bbox = annotation['data']['shapes'][0]['points']
#     print(class_name)
#     # print(bbox)
obj_list =[]
for file in files:
    with open(file, 'r') as annot:
        annotation = json.load(annot)
        objs = annotation['data']['labels']
        for obj in objs:
            class_name = obj['bbox1']['class']
            if class_name not in obj_list:
                obj_list += [class_name]
print('obj_list :', sorted(obj_list))


'''
# 파손 부위 data class check
'''
files = glob(r'/data/data/ex_task42/task42_1130/crop/annotations'+'/*.json')

label_list =[]
for file in files:
    with open(file, 'r') as annot:
        annotation = json.load(annot)
        objs = annotation['data']['labels']
        bboxs = annotation['data']['shapes']
        i = 0
        for obj in objs:
            i += 1
            class_name = obj['bbox1']['class']
            damagetypes = obj['bbox1']['bbox2']

            # image
            ture = 0
            image_name = file.split('/')[-1].replace('json', 'jpg')
            image = cv2.imread(os.path.join('/data/data/ex_task42/task42_1130/images', image_name))

            for damagetype in damagetypes:
                i += 1
                damage= damagetype['damagetype']
                bbox = bboxs[i-1]['points']
                label = f'{class_name}_{damage}'
                if label not in label_list:
                    label_list += [label]

                # 파손 라벨링 보기
                # print(image_name)
                image = cv2.rectangle(image, (int (bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255, 0, 0), 3)
                # ture = 1
                true = 0
            if ture:
                cv2.namedWindow(f'{image_name}', cv2.WINDOW_NORMAL)
                cv2.imshow(f'{image_name}', image)
                cv2.waitKey(2000)


print('obj_list :', sorted(label_list))
print('number :', len(label_list))
