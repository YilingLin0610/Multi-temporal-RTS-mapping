"""
Convert RS TIFF files into jpg files
author:Yiling Lin
"""
import os
import numpy as np
import cv2
import io


def tif_jpg_transform(file_path_name,bgr_savepath_name):
    """
    Convert one RS TIFF file into jpg file

    Parameters:
        file_path_name: The file path of TIFF file
        bgr_savepath_name: The file path where the converted jpg file will be stored

    Returns:
        None
    """
    img = io.imread(file_path_name)
    img = img / img.max()
    img = img * 255 - 0.001
    img = img.astype(np.uint8)
    b = img[:, :, 0]  # Read the blue channel
    g = img[:, :, 1]  # Read the green channel
    r = img[:, :, 2]  # Read the red channel
    nir = img[:, :, 3]  # Read the NIR channel
    bgr = cv2.merge([b, g, r])
    cv2.imwrite(bgr_savepath_name+".jpg", bgr)


def batch_processing(file_path, bgr_savepath):
    """
    Batch convert RS TIFF files to JPG format.

    Parameters:
        file_path: The file path where
        bgr_savepath: The file path where the converted jpg file will be stored
    Returns:
        None


    """
    filelist = [x for x in os.listdir(Origin_JPEGImages_path) if x.endswith(".tif")]
    for name in filelist:
        file_path_name = file_path + "\\" + name
        tif_jpg_transform(file_path_name, name[0:-4])  # 图像转换







if __name__ == "__main__":
    # The original path where the TIFF files stored
    Origin_JPEGImages_path = r'C:\Users\凌凌七\Desktop\学习文件夹\研一\ISPRS_special_issue_mars\file_for_picture\images_resample'
    # The file path where the converted JPG files will be stored
    Out_JPEGImages_path = r'C:\Users\凌凌七\Desktop\学习文件夹\研一\ISPRS_special_issue_mars\file_for_picture\images_resample_jpgs'

    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)
    batch_processing(Origin_JPEGImages_path,Out_JPEGImages_path)





