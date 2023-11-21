import numpy as np


class YOLOV4DataSetConfig:
    root_path: str = 'F:/RSI/NWPU'
    image_net_dir: str = 'F:/RSI/NWPU/JPEGImages'
    # data set root dir
    image_size: tuple = (608, 608)
    image_shrink_rate: dict = {
        'for_s': (8, 8),
        'for_m': (16, 16),
        'for_l': (32, 32),
    }
    # pre_anchor_w_h_rate: dict = {
    #     'for_s': ((0.03522504866123199, 0.08666666597127914),
    #               (0.06860158592462540, 0.05555555596947670),
    #               (0.06151632964611053, 0.10228779911994934)), #for small
    #
    #     'for_m': ((0.04852320626378059, 0.06693989038467407),
    #               (0.06267280876636505, 0.09665427356958389),
    #               (0.08818010985851288, 0.13137558102607727)), #for middle
    #
    #     'for_l': ((0.36180421710014343, 0.27767693996429443),
    #               (0.21234521269798279, 0.49952015280723572),
    #               (0.33031991124153137, 0.52864521741867065)), #for large
    # }
    pre_anchor_w_h_rate: dict = {
        'for_s': ((0.02962206304073334, 0.05680119618773460),
                  (0.04334948956966400, 0.05478394031524658),
                  (0.05041152238845825, 0.07926829159259796)), #for small

        'for_m': ((0.04232283309102058, 0.10518933832645416),
                  (0.07361456006765366, 0.07351593673229218),
                  (0.06042834743857384, 0.13101160526275635)), #for middle

        'for_l': ((0.08780992031097412, 0.12673880159854889),
                  (0.13554966449737549, 0.16680042445659637),
                  (0.26678138971328735, 0.46437728404998779)), #for large
    }
    single_an: int = 3
    kinds_name: list = [
        'airplane', 'ship', 'storage tank', 'baseball diamond',
        'tennis court', 'basketball court', 'ground track field',
        'harbor', 'bridge','vehicle'
    ]

    class_colors: list = [
        (np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(len(kinds_name))
    ]
#R_mean is 0.330720, G_mean is 0.349062, B_mean is 0.310268 for NWPU VHR-10
    mean = [0.5, 0.5, 0.5]
#R_var is 0.197215, G_var is 0.185535, B_var is 0.187597 for NWPU VHR-10
    std = [0.5, 0.5, 0.5]



class YOLOV4TrainConfig:
    max_epoch_on_detector = 201
    num_workers: int = 0
    device: str = 'cuda:0'
    batch_size = 8
    lr: float = 1e-4
    warm_up_end_epoch: int = 5
    use_mosaic: bool = True
    use_mixup: bool = True

    weight_position: float = 1.0
    weight_conf_has_obj: float = 5.0
    weight_conf_no_obj: float = 1.0
    weight_cls_prob: float = 1.0


class YOLOV4EvalConfig:
    eval_frequency = 10
    conf_th_for_eval: float = 0.0
    prob_th_for_eval: float = 0.0
    score_th_for_eval: float = 0.001
    iou_th_for_eval: float = 0.5
    use_07_metric: bool = False


class YOLOV4VisualizationConfig:

    conf_th_for_show: float = 0.0
    prob_th_for_show: float = 0.0
    score_th_for_show: float = 0.3

    iou_th_for_show: float = 0.5

class YOLOV4AdversarialAttackConfig:
    # R_mean is 0.330720, G_mean is 0.349062, B_mean is 0.310268 for NWPU VHR-10
    # mean = (0.5, 0.5, 0.5)
    # mean = np.array(mean, dtype=np.float32)
    # # R_var is 0.197215, G_var is 0.185535, B_var is 0.187597 for NWPU VHR-10
    # std = (0.5, 0.5, 0.5)
    # std = np.array(std, dtype=np.float32)
    # i_max = (1.0-mean)/std
    # max_image = i_max
    # i_min = (0.0-mean)/std
    # min_image = i_min
    # distance = i_max - i_min
    lambda_has_obj = 1.0  #5.0
    lambda_no_obj = 0.0
    lambda_cls = 1.0  #1.0
    lambda_loc = 1.0  #5.0
    lambda_pos_obj = 0
    lambda_shape_rate = 0.0  #3.5
    abs_eps = 6.0 / 255
    # eps = (abs_eps * distance).sum()/3
    eps = abs_eps * 2
    max_iter = 5
    # eps_iter = (1.0 / 255 *distance).sum()/3
    eps_iter = 2.0 / 255
    momentum = 1.0
    pd= None

class YOLOV4Config:
    ABS_PATH: str = 'E:/My_ORSI_workdir/yolo_nwpu_attack/yolo_V4_tmp'
    LOG_PATH: str = 'E:/My_ORSI_workdir/yolo_nwpu_attack/logging'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    iou_th_for_make_target: float = 0.5
    multi_gt: bool = False
    data_config = YOLOV4DataSetConfig()
    train_config = YOLOV4TrainConfig()
    eval_config = YOLOV4EvalConfig()
    show_config = YOLOV4VisualizationConfig()
    attack_config = YOLOV4AdversarialAttackConfig()

class YOLOv4DIORConfig(YOLOV4Config):
    ABS_PATH: str = 'E:/My_ORSI_workdir/yolo_nwpu_attack/yolo_V4_tmp'
    LOG_PATH: str = 'E:/My_ORSI_workdir/yolo_nwpu_attack/logging'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    iou_th_for_make_target: float = 0.5
    multi_gt: bool = False
    data_config = YOLOV4DataSetConfig()
    train_config = YOLOV4TrainConfig()
    eval_config = YOLOV4EvalConfig()
    show_config = YOLOV4VisualizationConfig()
    attack_config = YOLOV4AdversarialAttackConfig()
    data_config.root_path: str = 'F:/RSI/DIOR'
    data_config.image_net_dir: str = 'F:/RSI/DIOR/JPEGImages'
    data_config.kinds_name: list = [
        'golffield', 'Expressway-toll-station', 'vehicle', 'trainstation',
        'chimney', 'storagetank', 'ship', 'tenniscourt',
        'harbor', 'airplane', 'groundtrackfield', 'dam',
        'Expressway-Service-area', 'stadium', 'airport', 'baseballfield',
        'overpass', 'windmill', 'basketballcourt', 'bridge'
    ]  # for DIOR
    data_config.class_colors: list = [
        (np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(len(data_config.kinds_name))
    ]