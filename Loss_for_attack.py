import torch
import torch.nn as nn
from AYN_Package.Task.ObjectDetection.D2.Dev import DevLoss
from AYN_Package.Task.ObjectDetection.D2.YOLO.V4.Tools import YOLOV4Tool
from AYN_Package.Task.ObjectDetection.D2.YOLO.V5.Tools import YOLOV5Tool

class LossforAttack(DevLoss):
    def __init__(
            self,
            pre_anchor_w_h_rate: dict,
            image_shrink_rate: dict,
            each_size_anchor_number: int = 3,
            lambda_has_obj = 5.0,
            lambda_no_obj = 1.0,
            lambda_cls = 1.0,
            lambda_loc = 1.0,
            lambda_pos_obj = 1.0,
            lambda_shape_rate = 1.0,
            image_size: tuple = (416, 416),
            yolo_mode = 'V4',
            iou_loss_config = 'adviou', #iou、ciou、diou、giou、adviou
            has_obj_loss_config = 'adviou', #iou、ciou、diou、giou、adviou
    ):
        super().__init__()
        self.pre_anchor_w_h_rate = pre_anchor_w_h_rate
        self.pre_anchor_w_h = None  # type:dict

        self.image_shrink_rate = image_shrink_rate
        self.grid_number = None  # type:dict

        self.image_size = None
        self.yolo_mode = yolo_mode
        self.change_image_wh(image_size)

        self.each_size_anchor_number = each_size_anchor_number

        # self.weight_position = weight_position
        # self.weight_conf_has_obj = weight_conf_has_obj
        # self.weight_conf_no_obj = weight_conf_no_obj
        # self.weight_cls_prob = weight_cls_prob
        self.lambda_has_obj = lambda_has_obj
        self.lambda_no_obj = lambda_no_obj
        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc
        self.lambda_pos_obj = lambda_pos_obj
        self.lambda_shape_rate = lambda_shape_rate
        self.anchor_keys = list(pre_anchor_w_h_rate.keys())
        self.mse = nn.MSELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.bce_l = nn.BCEWithLogitsLoss(reduction='none')

        self.iou_loss_config = iou_loss_config
        self.has_obj_loss_config = has_obj_loss_config


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
        elif self.yolo_mode == 'V5':
            self.grid_number, self.pre_anchor_w_h = YOLOV5Tool.get_grid_number_and_pre_anchor_w_h(
                self.image_size,
                self.image_shrink_rate,
                self.pre_anchor_w_h_rate
            )
        else:
            print("Yolo_mode is not defined by the Project")

    def forward(
            self,
            out_put: dict,
            target: dict,
    ):
        if self.yolo_mode== 'V4':
            res_out = YOLOV4Tool.split_predict(
                out_put,
                self.each_size_anchor_number
            )
            res_target = YOLOV4Tool.split_target(
                target,
                self.each_size_anchor_number
            )
        elif self.yolo_mode=='V5':
            res_out = YOLOV4Tool.split_predict(
                out_put,
                self.each_size_anchor_number
            )
            res_target = YOLOV4Tool.split_target(
                target,
                self.each_size_anchor_number
            )
        else:
            print("Yolo_mode is not defined by the Project")
        loss_dict = {
            'total_loss': 0.0,
            'position_loss': 0.0,
            'has_obj_loss': 0.0,
            'no_obj_loss': 0.0,
            'cls_prob_loss': 0.0,
            'iou_loss': 0.0,
            'shape_rate_loss' : 0.0
        }

        for anchor_key in self.anchor_keys:
            pre_res_dict = res_out[anchor_key]
            gt_res_dict = res_target[anchor_key]

            N = out_put[anchor_key].shape[0]
            # -------------------------------------------------------------------
            # split output
            pre_txtytwth = pre_res_dict.get('position')[0]  # (N, H, W, a_n, 4)

            pre_txty = pre_txtytwth[..., 0:2]  # (N, H, W, a_n, 2)
            # be careful, not use sigmoid on pre_txty
            pre_twth = pre_txtytwth[..., 2:4]  # (N, H, W, a_n, 2)
            if self.yolo_mode == 'V4':
                pre_xyxy = YOLOV4Tool.txtytwth_to_xyxy(
                    pre_txtytwth,
                    self.pre_anchor_w_h[anchor_key],
                    self.grid_number[anchor_key]
                )
            elif self.yolo_mode == 'V5':
                pre_xyxy = YOLOV5Tool.txtytwth_to_xyxy(
                    pre_txtytwth,
                    self.pre_anchor_w_h[anchor_key],
                    self.grid_number[anchor_key]
                )
            else:
                print("Yolo_mode is not defined by the Project")

            # scaled in [0, 1]

            pre_conf = torch.sigmoid(pre_res_dict.get('conf'))  # (N, H, W, a_n)

            pre_cls_prob = pre_res_dict.get('cls_prob')  # (N, H, W, a_n, kinds_num)
            # be careful, if you use mse --> please softmax(pre_cls_prob)
            # otherwise not (softmax already used in CrossEntropy of PyTorch)
            # pre_cls_prob = torch.softmax(pre_res_dict.get('cls_prob'), dim=-1)  # (N, H, W, a_n, kinds_number)

            # -------------------------------------------------------------------
            # split target

            gt_xyxy = gt_res_dict.get('position')[1]
            # (N, H, W, a_n, 4) scaled in [0, 1]
            if self.yolo_mode == 'V4':
                gt_txty_s_twth = YOLOV4Tool.xyxy_to_txty_sigmoid_twth(
                    gt_xyxy,
                    self.pre_anchor_w_h[anchor_key],
                    self.grid_number[anchor_key]
                )
            elif self.yolo_mode == 'V5':
                gt_txty_s_twth = YOLOV5Tool.xyxy_to_txty_sigmoid_twth(
                    gt_xyxy,
                    self.pre_anchor_w_h[anchor_key],
                    self.grid_number[anchor_key]
                )
            else:
                print("Yolo_mode is not defined by the Project")

            gt_txty_s = gt_txty_s_twth[..., 0:2]  # (N, H, W, a_n, 2)
            gt_twth = gt_txty_s_twth[..., 2:4]  # (N, H, W, a_n, 2)

            gt_conf_and_weight = gt_res_dict.get('conf')  # (N, H, W, a_n)
            # gt_conf = (gt_conf_and_weight > 0).float()
            gt_weight = gt_conf_and_weight

            gt_cls_prob = gt_res_dict.get('cls_prob').argmax(dim=-1)  # (N, H, W, a_n)
            # be careful, if you use CrossEntropy of PyTorch, please argmax gt_res_dict.get('cls_prob')
            # because gt_res_dict.get('cls_prob') is one-hot code
            # gt_cls_prob = gt_res_dict.get('cls_prob')

            # -------------------------------------------------------------------
            # compute mask
            positive = (gt_weight > 0).float()
            ignore = (gt_weight == -1.0).float()
            negative = 1.0 - positive - ignore

            # -------------------------------------------------------------------
            # compute loss

            # position loss
            temp = (self.bce_l(pre_txty, gt_txty_s) + self.mse(pre_twth, gt_twth)).sum(dim=-1)
            loss_dict['position_loss'] += torch.sum(
                temp * positive * gt_weight
            ) / N

            # shape loss
            shape_rate_loss = YOLOV4Tool.shape_rate_loss(pre_xyxy,gt_xyxy)
            temp = 1.0 - shape_rate_loss
            loss_dict['shape_rate_loss'] += torch.sum(
                temp * positive * gt_weight
            ) / N

            # conf loss
            # compute iou
            if self.yolo_mode == 'V4':
                if self.iou_loss_config =='iou':
                    iou_loss = YOLOV4Tool.iou(pre_xyxy, gt_xyxy)
                elif self.iou_loss_config == 'ciou':
                    iou_loss = YOLOV4Tool.c_iou(pre_xyxy, gt_xyxy)  # (-1.0, 1.0)
                elif self.iou_loss_config == 'diou':
                    iou_loss = YOLOV4Tool.d_iou(pre_xyxy, gt_xyxy)
                elif self.iou_loss_config == 'giou':
                    iou_loss = YOLOV4Tool.g_iou(pre_xyxy, gt_xyxy)
                elif self.iou_loss_config == 'adviou':
                    iou_loss = YOLOV4Tool.adv_iou(pre_xyxy, gt_xyxy)
                else:
                    print("iou loss is not defined.")
                if self.has_obj_loss_config =='iou':
                    has_obj_iouloss = YOLOV4Tool.iou(pre_xyxy, gt_xyxy)
                elif self.has_obj_loss_config == 'ciou':
                    has_obj_iouloss = YOLOV4Tool.c_iou(pre_xyxy, gt_xyxy)  # (-1.0, 1.0)
                elif self.has_obj_loss_config == 'diou':
                    has_obj_iouloss = YOLOV4Tool.d_iou(pre_xyxy, gt_xyxy)
                elif self.has_obj_loss_config == 'giou':
                    has_obj_iouloss = YOLOV4Tool.g_iou(pre_xyxy, gt_xyxy)
                elif self.has_obj_loss_config == 'adviou':
                    has_obj_iouloss = YOLOV4Tool.adv_iou(pre_xyxy, gt_xyxy)
                else:
                    print("has obj iou_loss is not defined")
            elif self.yolo_mode == 'V5':
                iou_loss = YOLOV5Tool.c_iou(pre_xyxy, gt_xyxy)  # (-1.0, 1.0)
                has_obj_iouloss = iou_loss
                # g_iou = YOLOV5Tool.g_iou(pre_xyxy, gt_xyxy)  # (-1.0, 1.0)
                # d_iou = YOLOV5Tool.d_iou(pre_xyxy, gt_xyxy)  # (-1.0, 1.0)
                # iou = YOLOV5Tool.iou(pre_xyxy, gt_xyxy)  # (-1.0, 1.0)
            else:
                print("Yolo_mode is not defined by the Project")

            # iou = YOLOV4Tools.iou(pre_xyxy, gt_xyxy)  # (0.0, 1.0)

            # iou_loss
            temp = 1.0 - iou_loss
            loss_dict['iou_loss'] += torch.sum(
                temp * positive * gt_weight
            ) / N

            # has obj/positive loss
            iou_detach = 0.5 * has_obj_iouloss + 0.5  # (N, H, W, a_n) and no grad! (0.0, 1.0)
            # iou_detach = iou.detach()  # (0.0, 1.0)
            temp = self.mse(pre_conf, iou_detach)
            loss_dict['has_obj_loss'] += torch.sum(
                temp * positive
            ) / N

            # no obj/negative loss
            temp = self.mse(pre_conf, torch.zeros_like(pre_conf).to(pre_conf.device))
            loss_dict['no_obj_loss'] += torch.sum(
                temp * negative
            ) / N

            # cls prob loss
            temp = self.ce(
                pre_cls_prob.contiguous().view(-1, pre_cls_prob.shape[-1]),
                gt_cls_prob.contiguous().view(-1, )
            )

            loss_dict['cls_prob_loss'] += torch.sum(
                temp * positive.contiguous().view(-1, )
            ) / N

        loss_dict['total_loss'] = self.lambda_loc * loss_dict['iou_loss'] + \
                                  self.lambda_cls * loss_dict['cls_prob_loss'] + \
                                  self.lambda_has_obj * loss_dict['has_obj_loss'] + \
                                  self.lambda_no_obj * loss_dict['no_obj_loss'] + \
                                  self.lambda_pos_obj * loss_dict['position_loss'] + \
                                  self.lambda_shape_rate * loss_dict['shape_rate_loss']


        return loss_dict
