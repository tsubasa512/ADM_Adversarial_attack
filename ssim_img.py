import cv2
import cv2
import math
from utils.logging import init_log
import numpy as np
import math
import sys
from scipy import signal
from scipy import ndimage

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2+1 , -size//2+1:size//2+1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def cal_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':

    pth = "E:/My_ORSI_workdir/yolo_nwpu_attack/yolo_V4_tmp/attack_nwpu_images_save3/"
    log_pth = "E:/My_ORSI_workdir/yolo_nwpu_attack/yolo_V4_tmp/logging"
    logging = init_log(log_pth)
    _print = logging.info
    img1_name = "adv_"
    img2_name = "real_"
    for i in range(159):
        img1 = cv2.imread(pth+img1_name+str(i)+".png")
        img2 = cv2.imread(pth+img2_name+str(i)+".png")
        img1 = np.array(img1)
        img2 = np.array(img2)
        res1 = cal_ssim(img1, img2)
        print("res1:",res1)
        _print("第{:1d}张图图片质量:{:.10f}".format(i,res1))


