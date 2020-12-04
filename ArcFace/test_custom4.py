from test_custom import *
import cv2
import matplotlib.pyplot as plt
import pickle
from config.config_test import Config
from tqdm import tqdm

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

# identity_list = get_lfw_list(opt.test_list)
# with open('MS1M_identity_list3.pickle', 'wb') as fi:
#     pickle.dump(identity_list, fi)
with open('MS1M_identity_list3.pickle', 'rb') as fi:
    identity_list = pickle.load(fi)
print('identity_list :', identity_list)
print('identity_list length: ', len(identity_list))

# img_paths = [os.path.join(opt.test_root, each) for each in identity_list]
# with open('MS1M_img_paths3.pickle', 'wb') as fw:
#     pickle.dump(img_paths, fw)
with open('MS1M_img_paths3.pickle', 'rb') as fw:
    img_paths = pickle.load(fw)

# print('img_paths :', img_paths)

model.eval()

s = time.time()
# features, _ = get_featurs(model, img_paths , opt.input_shape, opt.test_batch_size)
# with open('MS1M_features3.pickle', 'wb') as ff:
#     pickle.dump(features, ff)

with open('MS1M_features3.pickle', 'rb') as ff:
    features = pickle.load(ff)

t = time.time() - s
print('time is {}'.format(t))

fe_dict = get_feature_dict(identity_list, features)                     # key = file path / value = feature(1024, )

sims = []
labels = []

with open(opt.test_list, 'r') as fd:
    pairs = fd.readlines()

# ths = np.arange(0.2, 1.02, 0.02)
ths = [0.28]
Acc = []
pbar = tqdm(pairs)
with open('arcface_acc5.txt', 'w') as f:
    for th in ths:
        score = []
        for p, pair in enumerate(pbar):
            splits = pair.split()
            test = {}
            for i in range(len(pairs)):
                fe_1_name = identity_list[i*2]
                fe_1 = fe_dict[fe_1_name]
                fe_2 = fe_dict[splits[1]]
                sim = cosin_metric(fe_1, fe_2)

                if sim >= th:
                    test[fe_1_name] = sim

            same = sorted(test.items(), key=(lambda x:x[1]), reverse=True)  # type = tuple
            true_name = splits[1].split('/')[0]
            test_name = 0 if same == {} else same[0][0].split('/')[0]

            cnt = 1 if test_name == true_name else 0
            # print(f'{splits[1]} = {same[0][0]} : {cnt}')

            score.append(cnt)
            pbar.set_description("Threshold: {} Pair: {}/{}".format(th, p, len(pairs)))

        score = np.asarray(score)
        acc = np.mean(score)
        print('Threshold : {:.5f}, Acc : {:.5f}'.format(th, acc))
        Acc.append(acc)
        f.write('Threshold : {:.5f}, Acc : {:.5f}'.format(th, acc))

plt.plot(ths, Acc)
plt.title('ArcFace')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.show()
