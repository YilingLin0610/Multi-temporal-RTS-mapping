"""
Generating training images and corresponding label images
author:Yiling Lin
"""

# coding=gbk
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


def export_RTS_individual(filePath,RTS_all):
    """
    Export each feature (RTS) in the shapefile as an individual shapefile file.
    The saved path is filePath+RTS_individual

    Parameters:
        filePath: The file path where intermediate and result data in the
                 training data generation process.
        RTS_all: Filename of original RTS shapefile.
    """
    outpath=filePath+'RTS_individual\\'
    try:
        os.mkdir(filePath+'RTS_individual')
    except:
        pass
    with arcpy.da.SearchCursor(RTS_all,["SHAPE@",'FID']) as cursor:
        for row in cursor:
            try:
                print(row[1])
                arcpy.FeatureClassToFeatureClass_conversion(row[0], outpath, row[1])
            except:
                print(str(row[1]),"falis")
        print('Finished RTS_individual!')

def export_RTS_buffer_individual(RTS_all,filePath,buffer_meter):
    """
     Exports the buffers of RTSs as individual polygon features,
     The saved path is filePath+RTS_buffers_individual.

     Parameters:
     filePath: The file path where intermediate and result data in the
               training data generation process.
     RTS_all : Filename of original RTS shapefile.
     buffer_meter : The length of the buffer zone in meters.

     """
    # Generate buffer shapefile
    try:
        os.mkdir(filePath+'RTS_buffers_'+str(buffer_meter))
    except:
        pass
    out_feature_class=filePath+'RTS_buffers_'+str(buffer_meter)+"\\AOI_RTS_buffer_"+str(buffer_meter)+".shp"
    buffer_distance_or_field=str(buffer_meter)+" Meters"
    arcpy.Buffer_analysis(in_features=RTS_all,
                          out_feature_class=out_feature_class,
                          buffer_distance_or_field=buffer_distance_or_field, line_side="FULL", line_end_type="ROUND",
                          dissolve_option="NONE", dissolve_field="", method="PLANAR")
    try:
        os.mkdir(filePath+'RTS_buffers_individual')
    except:
        pass
    outpath=filePath+'RTS_buffers_individual'
    num=0
    with arcpy.da.SearchCursor(out_feature_class, ["SHAPE@", 'FID']) as cursor:
        for row in cursor:
            # print('Clipping with'+str(row[1])+'row element to new raster')#打印正在裁剪中的栅格文件
            try:
                arcpy.FeatureClassToFeatureClass_conversion(row[0], outpath, row[1])
                num = num + 1
                print(num)
            except:
                print(str(num), str(row[1]), 'fails')

def Clip(shp,TifPath,outname):
    """
    Clip the TIFF file using the extent of shapefile

    Parameters:
        shp: The shapefile used to offer cutting extent
        TifPath: The TIFF file path to be clipped
        outname: The file path where the clipped TIFF file will be saved.
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shp, 1)

    layer = dataSource.GetLayer(0)

    extent = layer.GetExtent()
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    origi_x = geotrans[0]
    origi_y = geotrans[3]
    width_size = int((extent[1] - extent[0]) // 3)
    height_size = int((extent[3] - extent[2]) // 3)
    begin_width = int((extent[0] - origi_x) // 3)
    begin_height = int((origi_y - extent[3]) // 3)
    geo_x = begin_width * 3 + origi_x
    geo_y = origi_y - begin_height * 3
    cropped = dataset_img.ReadAsArray(begin_width, begin_height, width_size, height_size)

    geos = (geo_x, geotrans[1], geotrans[2], geo_y, geotrans[4], geotrans[5])
    writeTiff(cropped, geos, proj, outname)


def Clip_tif(smallTifPath,TifPath,outname):
    """
    Clip the TIFF file using the extent of another TIFF file

    Parameters:
        smallTifPath: The TIFF file used to offer cutting extent
        TifPath: The TIFF file path to be clipped
        outname: The file path where the clipped TIFF file will be saved.
    """
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    origi_x = geotrans[0]
    origi_y = geotrans[3]

    dataset_img_2 = readTif(smallTifPath)
    width_2 = dataset_img_2.RasterXSize
    height_2 = dataset_img_2.RasterYSize
    proj_2 = dataset_img_2.GetProjection()
    geotrans_2 = dataset_img_2.GetGeoTransform()
    origi_x_2 = geotrans_2[0]
    origi_y_2 = geotrans_2[3]
    width_size = width_2
    height_size = height_2
    begin_width = int((origi_x_2 - origi_x) // 3)
    begin_height = int((origi_y - origi_y_2) // 3)
    cropped = dataset_img.ReadAsArray(begin_width, begin_height, width_size, height_size)
    geos = (origi_x_2, geotrans[1], geotrans[2], origi_y_2, geotrans[4], geotrans[5])
    writeTiff(cropped, geos, proj, outname)


def clip_RS_image_as_traing_data(filePath,RS_path):
    """
    Generate the training images. The output filepath is filePath+training_data

    Parameters:
        filePath: The file path where intermediate and result data in the
                  training data generation process.
        RS_path: The remote sensing TIFF file path to be clipped
    """

    maskPath=filePath+'RTS_buffers_individual\\'
    files = os.listdir(maskPath)  #Getting all filename of the masks
    files_shp = []
    for i in range(len(files)):
        if (files[i][-4:] == '.shp'):
            files_shp.append(files[i])
    try:
        os.mkdir(filePath + 'training_data')
    except:
        pass
    out_path = filePath + 'training_data\\'
    num=0
    for i in range(len(files_shp)):
        mask_name = maskPath + files_shp[i]
        out_name = out_path + files_shp[i][0:-4] + '.tif'
        try:
            Clip(mask_name,RS_path,out_name)
            num = num + 1
            print(num, "       ", files_shp[i][0:-4], 'done')
        except:
            print('fails')
            num=num+1


def generate_label_data(filePath,TifPath):
    """
    Generate the corresponding label data referring to the training images.
    To ensure the training images and their corresponding label images share the same extent and size,
    the training image TIFF files are used as a reference to clip the label images.
    The output filepath is filePath+RTS_label

    Parameters:
        filePath: The file path where intermediate and result data in the
                  training data generation process.
        TifPath: The Label TIFF file path to be clipped
    """
    tifsPath = filePath+"training_data\\"
    files = os.listdir(tifsPath )
    files_tif = []
    for i in range(len(files)):
        if (files[i][-4:] == '.tif'):
            files_tif.append(files[i])
    try:
        os.mkdir(filePath+"RTS_label")
    except:
       pass
    outputPath = filePath+"RTS_label//"
    num=1
    for i in range(len(files_tif)):
        smallTifPath = tifsPath + files_tif[i]
        outputname = outputPath + files_tif[i][0:-4] + '.tif'
        Clip_tif(smallTifPath=smallTifPath,TifPath=TifPath,outname=outputname)
        num = num + 1
        print(num, "       ", files_tif[i][0:-4], 'done')
    print('end!')





if __name__ == "__main__":
    # The work directory used to store generated training iamges and label images as well as some intermediate data.
    filePath = r"F:\graduate-project\domain adversarial training\2019_new_datasets\\"
    # The shapefile containing all RTS groundtruth features.
    RTS_all = r'F:\graduate-project\model improvement\shapefile_RTS\2019\2019_nonRTS.shp'
    # The remote sensing images
    RS_path=r'F:\graduate-project\data\Planet data\merge\merge_2019.tif'
    # The label images
    Tif_Path=r"F:\graduate-project\model improvement\shapefile_RTS\2019\label_AOI.tif"
    print("Generate separate RTS masks")
    export_RTS_individual(filePath,RTS_all)
    print("Generate separate RTS masks with buffer zones")
    export_RTS_buffer_individual(RTS_all,filePath,300) # The buffer zone length is 300 meters
    print("Generate training images")
    clip_RS_image_as_traing_data(filePath,RS_path)
    print("Generate corresponding label images")
    generate_label_data(filePath,TifPath=Tif_Path)




