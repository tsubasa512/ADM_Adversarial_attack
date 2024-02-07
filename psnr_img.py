import cv2
import math
import numpy
from utils.logging import init_log

def psnr1(img1,img2):
    mse = numpy.mean((img1/1.0 - img2 / 1.0)**2)
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(255/math.sqrt(mse))
    return psnr1


def psnr2(img1,img2):
    mse = numpy.mean((img1 / 255.0 - img2 / 255.0)**2)
    if mse <1e-10:
        return 100
    psnr2 = 20 * math.log10(1/math.sqrt(mse))
    return psnr2

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
        res1 = psnr1(img1,img2)
        print("res1:",res1)
        _print("{:1d} image quality-PSNR:{:.10f}".format(i,res1))
