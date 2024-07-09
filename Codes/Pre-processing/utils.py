
"""
Frequently used functions to process TIFF file and shapefile
Author: Yiling Lin
"""
import os
import sys
import ogr
from osgeo import gdal
import numpy as np



def readTif(fileName):
    """
    Read a TIFF file.

    Parameters:
        fileName (str): Name of the TIFF file.

    Returns:
        dataset: Data contained in the TIFF file.
    """
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "file cannot be opened")
    return dataset


def writeTiff(im_data, im_geotrans, im_proj, path):
    """
    Save a TIFF file

    Parameters:
        im_data: An array containing the TIFF file data
        im_geotrans: The geographic location of the TIFF file
        im_proj: The projection information of the TIFF file
        path: The file path where the processed output will be saved.

    Returns:
        None
    """
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset
