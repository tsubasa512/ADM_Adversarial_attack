import os
import torch
import numpy as np
from AYN_Package.DataSet.ForObjectDetection.VOC_nwpu import *
from AYN_Package.BaseDev.cv2_ import CV2
from tqdm import tqdm
from AYN_Package.Task.ObjectDetection.D2.Dev.tools import DevTool

def cwloss(output : torch.Tensor,
           target : torch.Tensor,
           confidence = 50):
    _, num_classes = output.shape
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)

    target_var = target_onehot.clone().detach().requires_grad_(False)

    real = (target_var * output).sum(1)
    other = ((1. -  target_var) * output - target_var *10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min = 0.)
    loss = torch.sum(loss)
    return loss


def generate_attack_area(images, label):
    area = torch.zeros(images.shape)
    #area[...] = 0.01
    rate = 0.5
    imagew = images.shape[2]
    imageh = images.shape[3]
    batch_size = len(label)
    for i in range(batch_size):
        obj_number = len(label[i])
        for k in range(obj_number):
            bbox = np.round(label[i][k][1:5],0)
            w = bbox[3] - bbox[1]
            h = bbox[2] - bbox[0]
            xmin = np.clip(np.int(bbox[1]-w*rate),0,np.int(imagew))
            xmax = np.clip(np.int(bbox[3]+w*rate),0,np.int(imagew))
            ymin = np.clip(np.int(bbox[0]-h*rate),0,np.int(imageh))
            ymax = np.clip(np.int(bbox[2]+h*rate),0,np.int(imageh))
            #
            # h = bbox[3] - bbox[1]
            # temp = torch.ones(size=(w,h),dtype=float)
            area[i,:,xmin:xmax,ymin:ymax]=1.0

    return area

def generate_attack_area_enhance(images, label):
    area = torch.zeros(images.shape)
    #area[...] = 0.01
    rate = 0.5
    imagew = images.shape[2]
    imageh = images.shape[3]
    batch_size = len(label)
    for i in range(batch_size):
        obj_number = len(label[i])
        for k in range(obj_number):
            bbox = np.round(label[i][k][1:5],0)
            w = bbox[3] - bbox[1]
            h = bbox[2] - bbox[0]
            xmin1 = np.clip(np.int(bbox[1]-w*rate),0,np.int(imagew))
            xmax1 = np.clip(np.int(bbox[3]+w*rate),0,np.int(imagew))
            ymin1 = np.clip(np.int(bbox[0]-h*rate),0,np.int(imageh))
            ymax1 = np.clip(np.int(bbox[2]+h*rate),0,np.int(imageh))
            xmin0 = np.clip(np.int(bbox[1]),0,np.int(imagew))
            xmax0 = np.clip(np.int(bbox[3]),0,np.int(imagew))
            ymin0 = np.clip(np.int(bbox[0]),0,np.int(imageh))
            ymax0 = np.clip(np.int(bbox[2]),0,np.int(imageh))
            #
            # h = bbox[3] - bbox[1]
            # temp = torch.ones(size=(w,h),dtype=float)
            area[i,:,xmin1:xmax1,ymin1:ymax1]=1.0
            area[i,:,xmin0:xmax0,ymin0:ymax0]=2.5

    return area


if __name__ == '__main__':
    GPU_ID = 0
    root_path = 'F:/RSI/NWPU'
    image_size = (608, 608)
    batch_size = 8

    voc_test_loader = get_nwpu_voc_data_loader(
        root_path,
        image_size,
        batch_size,
        train=False,
        num_workers=0,
        mean=[0.330720, 0.349062, 0.310268],
        std=[0.197215, 0.185535, 0.187597]
    )

    for batch_id, (images, labels) in enumerate(tqdm(voc_test_loader,
                                                 desc="test only",
                                                 position=0)):
        if batch_id == 5:
            area = generate_attack_area(images,labels)
            multi_image = images * area
            area1 = DevTool.image_tensor_to_np(area[5,...])
            image1 = DevTool.image_tensor_to_np(images[5,...])
            multi_image1 = DevTool.image_tensor_to_np(multi_image[5,...])

            CV2.imwrite('areas.jpg',area1)
            CV2.imwrite('images.jpg',image1)
            CV2.imwrite('multi_images.jpg',multi_image1)
        else:
            pass



