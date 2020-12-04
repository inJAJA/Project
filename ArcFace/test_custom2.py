from test_custom import *
import cv2
import matplotlib.pyplot as plt
import pickle
from config.config_test import Config

opt = Config()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if opt.backbone == 'resnet18':
    model = resnet_face18(opt.use_se)
elif opt.backbone == 'resnet34':
    model = resnet34(input_shape=opt.input_shape)
elif opt.backbone == 'resnet50':
    model = resnet50(input_shape=opt.input_shape)
elif opt.backbone == 'resnet_face50':
    model = resnet_face50(opt.input_shape, use_se=opt.use_se)


model = DataParallel(model)
# load_model(model, opt.test_model_path)fmt
model.load_state_dict(torch.load(opt.test_model_path))
model.to(torch.device("cuda"))
print('model :', opt.test_model_path)

identity_list = get_lfw_list(opt.test_list)
with open('MS1M_identity_list.pickle', 'wb') as fi:
    pickle.dump(identity_list, fi)
# img_paths = [os.path.join(opt.test_root, each) for each in identity_list]
# with open('MS1M_img_paths.pickle', 'wb') as fw:
#     pickle.dump(img_paths, fw)

with open('MS1M_img_paths.pickle', 'rb') as fw:
    img_paths = pickle.load(fw)

print(img_paths)

model.eval()

s = time.time()
features, _ = get_featurs(model, img_paths , opt.input_shape, opt.test_batch_size)

t = time.time() - s
print('time is {}'.format(t))

fe_dict = get_feature_dict(identity_list, features)                     # key = file path / value = feature(1024, )

sims = []
labels = []

with open(opt.test_list, 'r') as fd:
    pairs = fd.readlines()

ths = np.arange(0.2, 1.02, 0.02)
Acc = []
with open('arcface_acc_1116.txt', 'w') as f:
    for th in ths:
        for pair in pairs:
            splits = pair.split()
            fe_1 = fe_dict[splits[0]]
            fe_2 = fe_dict[splits[1]]
            sim = cosin_metric(fe_1, fe_2)

            sims.append(sim)
            labels.append(int(splits[2]))

        print('sims :',len(sims))
        print('labels :',len(labels))

        y_score = np.asarray(sims)
        y_true = np.asarray(labels)

        y_test = (y_score >= th)
        print('y_test :', y_test, 'y_true :', y_true)
        acc = np.mean((y_test == y_true).astype(int))
        print(f'Threshold : {th}, Acc : {acc}')
        Acc.append(acc)
        f.write(f'Threshold : {th}, Acc : {acc}\n')

plt.plot(ths, Acc)
plt.title('ArcFace')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.show()
