import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from AYN_Package.Task.ObjectDetection.D2.YOLO.V4 import *
from AYN_Package.Optimizer.WarmUp import WarmUpOptimizer, WarmUpCosineAnnealOptimizer
from yolo_nwpu_attack.other.yolo_v4_config import YOLOV4Config
from .Loss_for_attack import LossforAttack
from .attack import ObjectionDetection_Attack,graph_MI_IFGSM,graph_FGSM,graph_IFGSM,graph_PGD,graph_PGD_eps,graph_IFGSM_eps,ObjectionDetection_Attack_AreaMask
from yolo_nwpu_attack.other.eval_for_attack import YOLOAttackEvaluator
from yolo_nwpu_attack.other.visualizer_for_attack import YOLOAttackVisualizer
from PIL import ImageFile
from AYN_Package.BaseDev import BaseTool
from AYN_Package.BaseDev import CV2
from yolo_nwpu_attack.other.attack_tools import generate_attack_area,generate_attack_area_enhance

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLOV4AttackHelper:
    def __init__(
            self,
            model: YOLOV4Model,
            config: YOLOV4Config,
            restore_epoch: int = -1,
            attacker = graph_IFGSM_eps,
            iou_loss_config = 'ciou',
            has_obj_loss_config = 'ciou'
    ):
        self.model = model  # type: nn.Module
        self.device = next(model.parameters()).device
        self.config = config
        self.attacker = attacker

        self.restore_epoch = restore_epoch
        self.yolo_mode = 'V4'
        self.iou_loss_config = iou_loss_config
        self.has_obj_loss_config = has_obj_loss_config

        if restore_epoch != -1:
            self.restore(restore_epoch)

        self.trainer = YOLOV4Trainer(
            model,
            self.config.data_config.pre_anchor_w_h_rate,
            self.config.data_config.image_size,
            self.config.data_config.image_shrink_rate,
            self.config.data_config.kinds_name,
            self.config.iou_th_for_make_target,
            multi_gt=self.config.multi_gt
        )

        self.predictor_for_show = YOLOV4Predictor(
            self.config.show_config.iou_th_for_show,
            self.config.show_config.prob_th_for_show,
            self.config.show_config.conf_th_for_show,
            self.config.show_config.score_th_for_show,
            self.config.data_config.pre_anchor_w_h_rate,
            self.config.data_config.kinds_name,
            self.config.data_config.image_size,
            self.config.data_config.image_shrink_rate,
            self.config.data_config.single_an
        )

        self.predictor_for_eval = YOLOV4Predictor(
            self.config.eval_config.iou_th_for_eval,
            self.config.eval_config.prob_th_for_eval,
            self.config.eval_config.conf_th_for_eval,
            self.config.eval_config.score_th_for_eval,
            self.config.data_config.pre_anchor_w_h_rate,
            self.config.data_config.kinds_name,
            self.config.data_config.image_size,
            self.config.data_config.image_shrink_rate,
            self.config.data_config.single_an
        )

        self.loss_lunc = LossforAttack(
            self.config.data_config.pre_anchor_w_h_rate,
            self.config.data_config.image_shrink_rate,
            self.config.data_config.single_an,
            self.config.attack_config.lambda_has_obj,
            self.config.attack_config.lambda_no_obj,
            self.config.attack_config.lambda_cls,
            self.config.attack_config.lambda_loc,
            self.config.attack_config.lambda_pos_obj,
            self.config.attack_config.lambda_shape_rate,
            image_size=self.config.data_config.image_size,
            yolo_mode=self.yolo_mode
        )

        self.attack_func = ObjectionDetection_Attack(
            model=self.model,
            loss_func= self.loss_lunc,
            eps = self.config.attack_config.eps,
            batch_size = self.config.train_config.batch_size,
            max_iter=self.config.attack_config.max_iter,
            eps_iter=self.config.attack_config.eps_iter,
            momentum=self.config.attack_config.momentum,
            pd= self.config.attack_config.pd,
            device=self.device,
            attacker=self.attacker
        )

        self.visualizer = YOLOAttackVisualizer(
            model,
            self.predictor_for_show,
            self.config.data_config.class_colors,
            self.config.iou_th_for_make_target,
            multi_gt=self.config.multi_gt,
            attack_func=self.attack_func,
            yolo_mode=self.yolo_mode
        )


        self.my_evaluator = YOLOAttackEvaluator(
            model,
            self.predictor_for_eval,
            self.config.iou_th_for_make_target,
            multi_gt=self.config.multi_gt,
            attack_func=self.attack_func,
            yolo_mode=self.yolo_mode
        )
    def change_attack_func(self,
                           attacker = graph_IFGSM,
                           ):
        self.attacker = attacker
        self.attack_func.attacker = attacker


    def change_iou_config(self,
                      iou_loss_config='iou',
                      has_obj_loss_config='iou',
                      ):
        self.iou_loss_config = iou_loss_config
        self.has_obj_loss_config = has_obj_loss_config
        self.loss_lunc.iou_loss_config =iou_loss_config
        self.loss_lunc.has_obj_loss_config = has_obj_loss_config

    def change_lambda_config(self,
                             config: YOLOV4Config,):
        self.config = config
        self.loss_lunc.lambda_has_obj = config.attack_config.lambda_has_obj
        self.loss_lunc.lambda_cls = config.attack_config.lambda_cls
        self.loss_lunc.lambda_loc = config.attack_config.lambda_loc
        self.loss_lunc.lambda_shape_rate = config.attack_config.lambda_shape_rate

    def restore(
            self,
            epoch: int
    ):
        self.restore_epoch = epoch
        saved_dir = self.config.ABS_PATH  + '/model_pth_detector/'
        saved_file_name = '{}/{}.pth'.format(saved_dir, epoch)
        self.model.load_state_dict(
            torch.load(saved_file_name)
        )

    def save(
            self,
            epoch: int
    ):
        # save model
        self.model.eval()
        saved_dir = self.config.ABS_PATH  + '/model_pth_detector/'
        os.makedirs(saved_dir, exist_ok=True)
        torch.save(self.model.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))

    def show_detect_results(
            self,
            data_loader_test: DataLoader,
            epoch: int
    ):
        with torch.no_grad():
            saved_dir = self.config.ABS_PATH + '/eval_images/{}/'.format(epoch)
            self.visualizer.show_detect_results(
                data_loader_test,
                saved_dir,
                desc='[show predict results]'
            )

    def show_detect_results_whole(
            self,
            data_loader_test: DataLoader,
    ):
        with torch.no_grad():
            saved_dir = 'E:/My_ORSI_workdir/backup' + '/eval_whole_images/'
            self.visualizer.show_detect_results_whole(
                data_loader_test,
                saved_dir,
                desc='[show whole predict results]'
            )

    def eval_map(
            self,
            data_loader_test: DataLoader,
    ):
        with torch.no_grad():
            self.my_evaluator.eval_map(
                data_loader_test,
                desc='[eval detector mAP]',
                use_07_metric=self.config.eval_config.use_07_metric
            )

    def eval_map_each_kind(
            self,
            data_loader_test: DataLoader,
    ):
        with torch.no_grad():
            self.my_evaluator.eval_map_each_kind(
                data_loader_test,
                desc='[eval detector mAP]',
                use_07_metric=self.config.eval_config.use_07_metric
            )

    def attack_and_evap(self,
        data_loader_adversarial: DataLoader,
                        mask = False
    ):
        return self.my_evaluator.eval_attack_map(
            data_loader_adversarial,
            mask = mask
        )

    def attack_and_visualizer(self,
        dataloader_adversarial: DataLoader,
                              mask=False,
                              save_path= 'attack_perdict_images'
    ):
        self.visualizer.show_detect_attack_results(
            dataloader_adversarial,
            saved_dir= self.config.ABS_PATH+'/'+save_path +'/',
            mask= mask
        )


    def attack_and_save(self,
                data_loader_adversarial: DataLoader,
                mask = False,
                save_path='Adv_image_output',
               ):

        index=0

        for batchid,(images,labels) in enumerate(data_loader_adversarial):
            targets = YOLOV4Tool.make_target(
                labels,
                self.trainer.pre_anchor_w_h,
                self.trainer.image_size,
                self.trainer.grid_number,
                self.trainer.kinds_name,
                self.trainer.iou_th_for_make_target,
                self.trainer.multi_gt)
            for anchor_key in list(targets):
                targets[anchor_key] =targets[anchor_key].to(self.device)

            if self.attack_func is not None and mask==False :
                images_adv = self.attack_func(images, targets)
                for i in range(len(images)):
                    x = BaseTool.image_tensor_to_np(images[i], self.config.data_config.mean,
                                                    self.config.data_config.std)

                    x_adv = BaseTool.image_tensor_to_np(images_adv[i], self.config.data_config.mean,
                                                        self.config.data_config.std)
                    save_dir = self.config.ABS_PATH + '/' + save_path + '/'
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    CV2.imwrite(save_dir + 'adv_' + str(index) + '.png', x_adv)
                    CV2.imwrite(save_dir + 'real_' + str(index) + '.png', x)
                    CV2.imwrite(save_dir + 'adv_per_' + str(index) + '.png', (x_adv - x))
                    index = index + 1
            elif self.attack_func is not None and mask==True:
                areas = generate_attack_area(images, labels)
                images_adv = self.attack_func(images,targets,areas)
                for i in range(len(images)):
                    x = BaseTool.image_tensor_to_np(images[i], self.config.data_config.mean,
                                                    self.config.data_config.std)

                    x_adv = BaseTool.image_tensor_to_np(images_adv[i], self.config.data_config.mean,
                                                        self.config.data_config.std)
                    areasnp = areas[i].numpy()
                    areasnp = np.transpose(areasnp, axes=(1, 2, 0))
                    save_dir = self.config.ABS_PATH + '/' + save_path + '/'
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    CV2.imwrite(save_dir + 'adv_' + str(index) + '.png', x_adv)
                    CV2.imwrite(save_dir + 'real_' + str(index) + '.png', x)
                    CV2.imwrite(save_dir + 'adv_per_' + str(index) + '.png', (x_adv - x) * areasnp)
                    index = index + 1
            else:
                images_adv = images






