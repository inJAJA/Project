import datetime

today = datetime.date.today()

class Config(object):
    project = 'DeepGlint'
    backbone = 'resnet_face100'
    classify = 'softmax'
    num_classes = 180855
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = True               # python -m visdom.server
    finetune = False

    env = f'{backbone}_{today}_{project}'

    train_root = f'/home/ubuntu/data/{project}/'
    train_list = f'datasets/{project}/train.txt'
    val_list = f'datasets/{project}/val.txt'
    val_interval = 1

    # test_root = '/data/lfw-align-128'
    test_root = 'datasets/GS'
    test_list = f'datasets/GS_test_pair.txt'

    lfw_root = '/home/ubuntu/data/lfw-align-128'
    lfw_test_list = 'datasets/lfw-align-128_test_pair.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = ''
    test_model_path = 'checkpoints/20201019-134322/resnet_face50_30.pth'
    save_interval = 2

    train_batch_size = 128 # batch size
    test_batch_size = 64

    input_shape = (3, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '1'
    num_workers = 4  # how many workers for loading data
    print_freq = 1   # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 150
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
