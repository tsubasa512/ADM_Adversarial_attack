from AYN_Package.DataSet.ForObjectDetection.VOC_nwpu import *
from AYN_Package.Task.ObjectDetection.D2.YOLO.V4 import *
from yolo_nwpu_attack.other.yolo_v4_config import *
from yolo_nwpu_attack.other.yolo_v4_helper import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import numpy as np
import random
import os
from utils.logging import init_log
from datetime import datetime
from yolo_nwpu_attack.other.attack import graph_IFGSM,graph_PGD,graph_PGD_eps



if __name__ == '__main__':

    manual_seed = 3407
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.manual_seed(manual_seed)
    torch.random.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.backends.cudnn.deteministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    iou_config=['iou', 'ciou', 'diou','giou']
    has_obj_loss_config=['iou', 'ciou', 'diou','giou']
    # lambda_has_obj = [0.01, 1.0 ,3.0, 5.0]
    lambda_has_obj = [5.0, 5.0, 5.0]
    #lambda_has_obj = [1.0]
    lambda_cls     = [1.0, 1.0, 1.0]
    #lambda_cls     = [1.0]
    lambda_loc     = [5.0, 5.0, 1.0]
    #lambda_loc     = [1.0]
    lambda_shape_rate = [0.0, 2.9, 0.0, 4.1, 0.0, 4.1]

    GPU_ID = 0

    attack_function = [graph_IFGSM, graph_PGD, graph_PGD_eps]


    config = YOLOV4Config()
    #config = YOLOv4DIORConfig()
    config.train_config.device = 'cuda:{}'.format(GPU_ID)

    csp_dark_net_53 = get_backbone_csp_darknet_53()

    net = YOLOV4Model(csp_dark_net_53, num_classes=len(config.data_config.kinds_name))

    net.to(config.train_config.device)
    if not os.path.exists(config.LOG_PATH):
        os.makedirs(config.LOG_PATH)
    logging = init_log(config.LOG_PATH)
    _print = logging.info

    """
            get data
    """

    voc_test_loader = get_nwpu_voc_data_loader(
        config.data_config.root_path,
        config.data_config.image_size,
        config.train_config.batch_size,
        train=False,
        num_workers=config.train_config.num_workers,
        mean=config.data_config.mean,
        std=config.data_config.std
    )

    helper = YOLOV4AttackHelper(
        net,
        config,
        restore_epoch=390
    )
    shape = (len(attack_function), len(lambda_has_obj), len(lambda_cls), len(lambda_loc))
    result_sum = np.zeros(shape=shape)
    # helper.eval_map(voc_test_loader)
    # result = helper.attack_and_evap(voc_test_loader)
    # result_sum_aaa = np.zeros(shape=(2,4))

    for i, attacker in enumerate(attack_function):
        helper.change_attack_func(attacker)
        alpha = lambda_has_obj[i]
        beta = lambda_cls[i]
        gamma = lambda_loc[i]
        config.attack_config.lambda_has_obj = alpha
        config.attack_config.lambda_cls = beta
        config.attack_config.lambda_loc = gamma
        for j in range(2):
            mu = lambda_shape_rate[2 * i + j]
            config.attack_config.lambda_shape_rate = mu
            helper.change_lambda_config(config)
            result_min = helper.attack_and_evap(voc_test_loader)
            _print('第{:1d}种攻击方法alpha = {:.2f}, beta = {:.1f}, gamma= {:.1f}, mu = {:.1f}:'.format(i + 1, alpha,beta, gamma, mu))



