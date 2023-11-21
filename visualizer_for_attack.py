

import torch
from torch.utils.data import DataLoader
from AYN_Package.Task.ObjectDetection.D2.YOLO.V4.Predictor import YOLOV4Predictor
from AYN_Package.Task.ObjectDetection.D2.YOLO.V5.Predictor import YOLOV5Predictor
from AYN_Package.Task.ObjectDetection.D2.YOLO.V4.Tools import YOLOV4Tool
from AYN_Package.Task.ObjectDetection.D2.YOLO.V5.Tools import YOLOV5Tool
from AYN_Package.Task.ObjectDetection.D2.YOLO.V4.Model import YOLOV4Model
from AYN_Package.Task.ObjectDetection.D2.YOLO.V5.Model import YOLOV5Model
from AYN_Package.Task.ObjectDetection.D2.Dev import DevVisualizer
from AYN_Package.Task.ObjectDetection.D2.Dev.predictor import DevPredictor
from AYN_Package.BaseDev import BaseVisualizer, CV2
from tqdm import tqdm
from yolo_nwpu_attack.other.attack_tools import generate_attack_area
import os


class YOLOAttackVisualizer(DevVisualizer):
    def __init__(
            self,
            model: YOLOV4Model or YOLOV5Model,
            predictor: YOLOV4Predictor or YOLOV5Predictor,
            class_colors: list,
            iou_th_for_make_target: float,
            multi_gt: bool,
            attack_func = None,
            yolo_mode = 'V4'
    ):
        self.yolo_mode = yolo_mode
        super().__init__(
            model,
            predictor,
            class_colors,
            iou_th_for_make_target
        )

        self.predictor = predictor
        self.anchor_keys = self.predictor.anchor_keys
        self.multi_gt = multi_gt
        if attack_func is not None:
            self.attack_func = attack_func
        else:
            self.attack_func = None


    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        if self.yolo_mode == 'V4':
            self.grid_number, self.pre_anchor_w_h = YOLOV4Tool.get_grid_number_and_pre_anchor_w_h(
                self.image_size,
                self.image_shrink_rate,
                self.pre_anchor_w_h_rate
            )
        elif self.yolo_mode=='V5':
            self.grid_number, self.pre_anchor_w_h = YOLOV5Tool.get_grid_number_and_pre_anchor_w_h(
                self.image_size,
                self.image_shrink_rate,
                self.pre_anchor_w_h_rate
            )
        else:
            print("Yolo_mode is not defined by the Project")


    def make_targets(
            self,
            labels,
    ):
        if self.yolo_mode == 'V4':
            targets = YOLOV4Tool.make_target(
                labels,
                self.pre_anchor_w_h,
                self.image_size,
                self.grid_number,
                self.kinds_name,
                self.iou_th_for_make_target,
                multi_gt=self.multi_gt
            )
        elif self.yolo_mode=='V5':
            targets = YOLOV5Tool.make_target(
                labels,
                self.pre_anchor_w_h,
                self.image_size,
                self.grid_number,
                self.kinds_name,
                self.iou_th_for_make_target,
                multi_gt=self.multi_gt
            )
        else:
            print("Yolo_mode is not defined by the Project")

        for anchor_key in self.anchor_keys:
            targets[anchor_key] = targets[anchor_key].to(self.device)
        return targets

    def show_detect_attack_results(
            self,
            data_loader_test: DataLoader,
            saved_dir: str,
            desc: str = 'show attack predict result',
            mask = False
    ):
        os.makedirs(saved_dir, exist_ok=True)
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            self.model.eval()
            images = images.to(self.device)
            targets = self.make_targets(labels)
            if self.attack_func.attacker!=None and mask==False :
                images_adv = self.attack_func(images, targets)
            elif self.attack_func.attacker!=None and mask==True:
                areas = generate_attack_area(images, labels)
                images_adv = self.attack_func(images,targets,areas)
            else:
                images_adv = images


            output = self.model(images_adv)

            gt_decode = self.predictor.decode_target(targets)
            pre_decode = self.predictor.decode_predict(output)

            for image_index in range(images.shape[0]):
                self.visualize(
                    images[image_index],
                    gt_decode[image_index],
                    saved_path='{}/{}_{}_gt.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )

                self.visualize(
                    images_adv[image_index],
                    pre_decode[image_index],
                    saved_path='{}/{}_{}_predict.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )



