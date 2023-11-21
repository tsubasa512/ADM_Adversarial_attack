from AYN_Package.Task.ObjectDetection.D2.YOLO.V4.Predictor import YOLOV4Predictor
from AYN_Package.Task.ObjectDetection.D2.YOLO.V5.Predictor import YOLOV5Predictor
from AYN_Package.Task.ObjectDetection.D2.YOLO.V4.Tools import YOLOV4Tool
from AYN_Package.Task.ObjectDetection.D2.YOLO.V5.Tools import YOLOV5Tool
from AYN_Package.Task.ObjectDetection.D2.YOLO.V4.Model import YOLOV4Model
from AYN_Package.Task.ObjectDetection.D2.YOLO.V5.Model import YOLOV5Model
from AYN_Package.Task.ObjectDetection.D2.Dev import DevEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from .attack_tools import generate_attack_area,generate_attack_area_enhance


class YOLOAttackEvaluator(DevEvaluator):
    def __init__(
            self,
            model: YOLOV4Model or YOLOV5Model,
            predictor: YOLOV4Predictor or YOLOV5Predictor,
            iou_th_for_make_target: float,
            multi_gt: bool,
            attack_func = None,
            yolo_mode = 'V4',
    ):
        super().__init__(
            model,
            predictor,
            iou_th_for_make_target
        )

        self.predictor = predictor
        self.anchor_keys = self.predictor.anchor_keys
        self.multi_gt = multi_gt
        self.yolo_mode = yolo_mode
        if attack_func is not None:
            self.attack_func = attack_func
        else:
            self.attack_func = None

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV4Tool.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def make_targets(
            self,
            labels
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
        elif self.yolo_mode =='V5':
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

    def eval_attack_map(
            self,
            data_loader_test: DataLoader,
            desc: str = 'eval detector mAP',
            use_07_metric: bool = True,
            mask = False
    ):
        print('')
        print('start eval Adversarial mAP(use_07_metric: {})...'.format(use_07_metric).center(50, '*'))
        print('be careful, we do not ignore the difficult objects, so mAP may decrease a little...')

        pre_info_vec = []
        gt_info_vec = []
        """
        info_vec = [info_0, info_1, ...]
        info = [
            class_id,       # 0
            abs_pos,        # 1-4 (x y x y)  
            score,          # 5
            image_id,       # 6
        ]
        """
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            self.model.eval()
            images = images.to(self.device)
            targets = self.make_targets(labels)
            if self.attack_func is not None and mask==False :
                images_adv = self.attack_func(images, targets)
            elif self.attack_func is not None and mask==True:
                areas = generate_attack_area(images, labels)
                # areas = generate_attack_area_enhance(images, labels)
                images_adv = self.attack_func(images,targets,areas)
            else:
                images_adv = images

            output = self.model(images_adv)

            gt_decode = self.predictor.decode_target(targets)  # kps_vec_s
            pre_decode = self.predictor.decode_predict(output)  # kps_vec_s

            gt_info_vec += self.get_info(
                gt_decode,
                batch_id,
                data_loader_test.batch_size
            )
            pre_info_vec += self.get_info(
                pre_decode,
                batch_id,
                data_loader_test.batch_size
            )

        pre_info_vec = np.array(pre_info_vec, dtype=np.float32)
        gt_info_vec = np.array(gt_info_vec, dtype=np.float32)
        ap_vec = []
        for kind_ind in range(len(self.kinds_name)):
            ap = self.compute_ap(
                pre_info_vec[pre_info_vec[:, 0] == kind_ind],
                gt_info_vec[gt_info_vec[:, 0] == kind_ind],
                use_07_metric=use_07_metric
            )
            ap_vec.append(ap)
        result = np.mean(ap_vec)
        print('\nmAP:{:.2%}'.format(result))
        return result
