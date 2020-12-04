from test_custom import *
import cv2
import matplotlib.pyplot as plt

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
# load_model(model, opt.test_model_path)
model.load_state_dict(torch.load(opt.test_model_path))
model.to(torch.device("cuda"))



identity_list = get_lfw_list(opt.test_list)
img_paths = [os.path.join(opt.test_root, each) for each in identity_list]
print(img_paths)

model.eval()

s = time.time()
features, _ = get_featurs(model, img_paths , opt.input_shape, 1)
t = time.time() - s
print('time is {}'.format(t))

fe_dict = get_feature_dict(identity_list, features)                     # key = file path / value = feature(1024, )

sims = []
labels = []

with open(opt.test_list, 'r') as fd:
    pairs = fd.readlines()


for pair in pairs:
    splits = pair.split()
    fe_1 = fe_dict[splits[0]]
    fe_2 = fe_dict[splits[1]]
    sim = cosin_metric(fe_1, fe_2)

    sims.append(sim)

    image1 = cv2.imread(os.path.join(opt.test_root, splits[0]))
    image2 = cv2.imread(os.path.join(opt.test_root, splits[1]))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    th = 0.5
    if sim > th:
        result = 'same'
    else:
        result = 'diffrent'

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image1)
    ax[1].imshow(image2)
    fig.suptitle(f'Similarity : { sim }  Result : {result}')
    plt.show()
