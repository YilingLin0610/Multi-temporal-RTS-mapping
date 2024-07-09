"""
Generating TIFF file used as testing images
author:Yiling Lin
"""
# coding=gbk
import os
import numpy as np
import shutil
from osgeo import gdal
from utils import readTif,writeTiff


def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    """
    Generate testing images.
    Using sliding windows to clip testing images from entire image covering the study area.

    Parameters:
        TifPath: The file path of the entire image covering the study area
        SavePath: The file path where store the generated testing images
        CropSize: The height and width of the clipped testing images
        RepetitionRate: The repetition rate between adjacent sliding windows.
    """

    file_basenames = os.path.basename(TifPath)
    file_basename=file_basenames[0:-4]
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    origi_x=geotrans[0]
    origi_y=geotrans[3]
    try:
        os.mkdir(SavePath)
    except:
        pass
    if(height>CropSize and width>CropSize):
        for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            print("row_number:%d" % i)
            for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
                filename = file_basename + "_" + str(i) + "_" + str(j) + ".tif"
                cropped = dataset_img.ReadAsArray(int(j * CropSize * (1 - RepetitionRate)), \
                                                  (int(i * CropSize * (1 - RepetitionRate))), CropSize, CropSize)
                geo_x = origi_x + int(j * CropSize * (1 - RepetitionRate)) * 3
                geo_y = origi_y - int(i * CropSize * (1 - RepetitionRate)) * 3
                geos = (geo_x, geotrans[1], geotrans[2], geo_y, geotrans[4], geotrans[5])
                writeTiff(cropped, geos, proj, SavePath + "/" + filename)
        # Clipping the last column
        last_list = int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))
        last_row = int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))

        for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            filename = file_basename + "_" + str(i) + "_" + str(last_list) + ".tif"
            cropped = dataset_img.ReadAsArray((width - CropSize), \
                                              int(i * CropSize * (1 - RepetitionRate)), CropSize, CropSize)
            geo_x = origi_x + (width - CropSize) * 3
            geo_y = origi_y - int(i * CropSize * (1 - RepetitionRate)) * 3
            geos = (geo_x, geotrans[1], geotrans[2], geo_y, geotrans[4], geotrans[5])
            writeTiff(cropped, geos, proj, SavePath + "/" + filename)
        # Clipping the last row
        last_row = int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            filename = file_basename + "_" + str(last_row) + "_" + str(j) + ".tif"
            cropped = dataset_img.ReadAsArray((int(j * CropSize * (1 - RepetitionRate))), height - CropSize, CropSize,
                                              CropSize)
            geo_x = origi_x + int(j * CropSize * (1 - RepetitionRate)) * 3
            geo_y = origi_y - (height - CropSize) * 3
            geos = (geo_x, geotrans[1], geotrans[2], geo_y, geotrans[4], geotrans[5])
            writeTiff(cropped, geos, proj, SavePath + "/" + filename)
        #  ≤√ºÙ”“œ¬Ω«
        filename = file_basename + "_" + str(last_row) + "_" + str(last_list) + ".tif"
        cropped = dataset_img.ReadAsArray((width - CropSize), (height - CropSize), CropSize, CropSize)
        geo_x = origi_x + (width - CropSize) * 3
        geo_y = origi_y - (height - CropSize) * 3
        geos = (geo_x, geotrans[1], geotrans[2], geo_y, geotrans[4], geotrans[5])
        writeTiff(cropped, geos, proj, SavePath + "/" + filename)
    else:
        outname=SavePath + "/" + file_basename+"_1_1.tif"
        shutil.copy(TifPath,outname)




if __name__ == '__main__':

    # Clip the testing images
    TifCrop(r"/media/dell/disk/Yiling/RTS_project_2/datasets/Planet_images/merge_2022.tif",
            r"/media/dell/disk/Yiling/RTS_project_2/datasets/testing_data/tifs", 300, 0.4)
    # Clip the testing labels
    TifCrop(r"F:\graduate-project\data\SAR_AOI\\RTS_SAR_raster.tif",
          r"F:\graduate-project\data\testing_data\testing_data_2019_overlap_0.3_correct\label_patches", 300, 0.4)


