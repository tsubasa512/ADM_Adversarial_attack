import torch.nn as nn
import torch.nn.functional as F
from AYN_Package.DataSet.ForObjectDetection.VOC_nwpu import *
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader
from AYN_Package.Task.ObjectDetection.D2.YOLO.V4 import *
from AYN_Package.Optimizer.WarmUp import WarmUpOptimizer, WarmUpCosineAnnealOptimizer
from yolo_nwpu_attack.other.yolo_v4_config import YOLOV4Config
from .Loss_for_attack import LossforAttack
from PIL import ImageFile
from torchattacks.attack import Attack
import torch


def graph_IFGSM(model,image,target,attack_config,x_max,x_min,loss_func,device,areas):
    imageval = torch.tensor(image,requires_grad=True).to(device)
    adv_images = imageval
    model.eval()
    for i in range(attack_config.max_iter):
        adv_images = torch.tensor(adv_images, requires_grad= True).to(device)
        adv_images.retain_grad()
        model.zero_grad()
        outputs = model(adv_images)
        adv_loss = loss_func(outputs, target)
        adv_loss['total_loss'].backward()
        grad = adv_images.grad
        grad = areas * grad
        # print(torch.sum(torch.sum(grad,dim=0),dim=1))
        adv_images = adv_images + attack_config.eps_iter * torch.sign(grad)
        x_min = x_min.to(device)
        x_max = x_max.to(device)
        adv_images = adv_images.clamp(x_min, x_max).detach()
    return adv_images.detach()


def graph_PGD_eps(model,image,target,attack_config,x_max,x_min,loss_func,device,areas):
    size = image.shape
    noise = torch.rand(size ).to(device)
    imageval = torch.tensor(image,requires_grad=True).to(device)
    temp = imageval + 0.1 * noise
    temp = temp.clamp(imageval-attack_config.eps,imageval +  attack_config.eps)
    temp = temp.clamp(x_min,x_max)
    adv_images = torch.tensor(temp, requires_grad=True).to(device)
    eps_steps = np.linspace(attack_config.eps, attack_config.eps/2, attack_config.max_iter)
    model.eval()
    for i in range(attack_config.max_iter):
        adv_images = torch.tensor(adv_images, requires_grad= True)
        adv_images.retain_grad()
        model.zero_grad()
        outputs = model(adv_images)
        adv_loss = loss_func(outputs, target)
        adv_loss['total_loss'].backward()
        grad = adv_images.grad
        grad = areas * grad
        adv_images = adv_images + eps_steps[i] * torch.sign(grad)
        x_min = x_min.to(device)
        x_max = x_max.to(device)
        adv_images = adv_images.clamp(x_min, x_max).detach()
    return adv_images


def graph_PGD(model,image,target,attack_config,x_max,x_min,loss_func,device,areas):
    size = image.shape
    noise = torch.rand(size ).to(device)
    imageval = torch.tensor(image,requires_grad=True).to(device)
    temp = imageval + 0.1 * noise
    temp = temp.clamp(x_min,x_max)
    adv_images = torch.tensor(temp, requires_grad=True).to(device)
    model.eval()
    for i in range(attack_config.max_iter):
        adv_images = torch.tensor(adv_images, requires_grad= True)
        adv_images.retain_grad()
        model.zero_grad()
        outputs = model(adv_images)
        adv_loss = loss_func(outputs, target)
        adv_loss['total_loss'].backward()
        grad = adv_images.grad
        grad = areas * grad
        adv_images = adv_images + attack_config.eps_iter * torch.sign(grad)
        x_min = x_min.to(device)
        x_max = x_max.to(device)
        adv_images = adv_images.clamp(x_min, x_max).detach()
    return adv_images.detach()

class attack_config:
    def __init__(self,
                 max_image,
                 min_image,
                 eps,
                 max_iter,
                 eps_iter,
                 momentum,
                 pd
                 ):
        self.max_image = max_image
        self.min_image = min_image
        self.eps = eps
        self.max_iter = int(min(eps*255+4,1.25*max_iter))
        self.eps_iter = eps_iter
        self.momentum = momentum
        self.pd = pd


class ObjectionDetection_Attack(Attack):
    def __init__(self,
                 model,
                 loss_func,
                 max_image = 1.0,
                 min_image = -1.0,
                 eps=10.0/255,
                 batch_size = 1,
                 max_iter=20,
                 eps_iter = 1.0/255,
                 momentum = 1.0,
                 attack_mode=1.0,
                 pd = None,
                 device='cpu',
                 attacker = graph_FGSM
                 ):
        super().__init__("ObjectionDetection_Attack",model)
        self.attack_config = attack_config(max_image, min_image, eps,max_iter,eps_iter,momentum,pd)
        self.batch_size = batch_size
        self.attack_mode = attack_mode
        self._supported_mode = ['identical', 'different']
        self.loss_func = loss_func
        self.device = device
        self.attacker = attacker


    def forward(self, image, target):
        if self.batch_size==1:
            image_ex = image.unsqueeze(0)
            target_ex = target.unsqueeze(0)
        else:
            image_ex = image
            target_ex = target
        x_max = (image_ex+self.attack_config.eps).clamp(min=None, max=self.attack_config.max_image).to(self.device)
        x_min = (image_ex-self.attack_config.eps).clamp(min=self.attack_config.min_image, max=None).to(self.device)
        areas = torch.ones(image_ex.shape).to(self.device)
        adv_images = self.attacker(self.model,image_ex,target_ex, self.attack_config ,x_max,x_min,self.loss_func,self.device,areas)

        return adv_images





