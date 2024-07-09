#coding=gbk
"""
Geometric characteristics of RTSs in Beiluhe region.
Author: Yiling Lin
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from osgeo import gdal
import numpy as np
mpl.rcParams['font.family'] = ['Helvetica']
mpl.rcParams['font.size'] = 40
import os

def calculate_image_pixels(image_name):
    """
    Given a tif image, read all image pixel data as a 1*M dimension array

    Parameter:
        image_name: path of the image
    return:
        data: a 1*M dimension array
    """
    dataset = gdal.Open(image_name)
    if dataset == None:
        print(image_name + "invalid")
    data=dataset.ReadAsArray()
    data=np.reshape(data,(-1,1))
    data=data[data>-1000]

    return data

def density(data,xlabel,xspan,colors):
    """
    Draw a distribution plot of RTS arributes

    Parameter:
        data: input data, an array with all RTS's geometric data
        xlabel: xlabel
        xspan: The extent of x axes
        color: The color of line and histogram
    return:
        None
    """
    plt.rcParams["font.size"] = 20 #Set the fontsize
    fig, ax = plt.subplots(figsize=(9, 6))
    ax2 = ax.twinx()
    # Plot the histogram and KDE
    sns.distplot(
        data ,
        bins=50,
        kde=False,
        color=colors[0],
        norm_hist=False,
        hist_kws={"edgecolor": colors[1], "alpha": 0.6},
        ax=ax
    )
    sns.distplot(
        data,
        hist=False,
        bins=50,
        kde=True,
        color=colors[1],
        norm_hist=False,
        kde_kws={"color": colors[2], "linewidth":2},
        ax=ax2
    )


    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.xlim(xspan)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax2.set_ylabel("Probability density")

    fig.savefig(r'C:\Users\凌凌七\Desktop\学习文件夹\研究生零年级\毕业设计\RSE\images\\'+"soil moisture"+".png", bbox_inches='tight')

    plt.show()


def density_with_gt(data,data_gt,xlabel,xspan,colors):
    """
   Draw a distribution plot of RTS attributes and background attributes

    Parameter:
        data: input data, an array with all RTS's attribute
        data_gt: input data, an array with all background attribute
        xlabel: xlabel
        xspan: The extent of x axes
        colors: The color of line and histogram
    return:
        None
    """
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots(figsize=(9, 6))
    ax2 = ax.twinx()
    # Plot the histogram and KDE
    sns.distplot(
        data ,
        #hist=True,
        bins=50,
        kde=False,
        color=colors[0],
        norm_hist=False,
        hist_kws={"edgecolor": colors[1], "alpha": 0.6},
        #kde_kws={"color": colors[2], "linewidth":2},
        ax=ax
        # Set the x-axis limits
    )
    sns.distplot(
        data,

        hist=False,
        bins=50,
        kde=True,
        color=colors[1],
        norm_hist=False,
        #hist_kws={"edgecolor": colors[2], "alpha": 0.6},
        kde_kws={"color": colors[2], "linewidth":2},
        ax=ax2
        # Set the x-axis limits
    )
    sns.distplot(
        data_gt,
        hist=False,
        bins=50,
        kde=True,
        color="#7F7F7F",
        norm_hist=False,
        kde_kws={"color": "#7F7F7F", "linewidth": 2,"linestyle":'--'},
        ax=ax2
    )


    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.xlim(xspan)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax2.set_ylabel("Probability density")
    try:
        fig.savefig(r'C:\Users\凌凌七\Desktop\学习文件夹\研究生零年级\毕业设计\RSE\images\\'+xlabel+".png", bbox_inches='tight')
    except:
        fig.savefig(r'C:\Users\凌凌七\Desktop\学习文件夹\研究生零年级\毕业设计\RSE\images\\' +"Soil_moisture" + ".png",
                    bbox_inches='tight')
    plt.show()










if __name__ == '__main__':
    # Read the attribute data
    Slope_landscape = calculate_image_pixels(r"F:\graduate-project\data\rasters_COP30\Slope.tif")
    altitude_landscape = calculate_image_pixels(r"F:\graduate-project\data\rasters_COP30\fill.tif")
    path_all = r"F:\graduate-project\statistics\TADA_statistics\\"
    soil_moisture_landscape = calculate_image_pixels(os.path.join(path_all, "datasets\soil_moisture_20200801\soil_moisture.tif"))
    df_2022_area = pd.read_csv(r'F:\graduate-project\statistics\Insar\Areas\2022.csv')
    areas_2022 = df_2022_area["area_ha"]
    colors = [(242 / 255, 207 / 255, 211 / 255, 1),
              (231 / 255, 169 / 255, 177 / 255, 1),
              "#E2979F"]
    density(areas_2022, "Area (ha)", [0, 16], colors)

    df_2022_area = pd.read_csv(r'F:\graduate-project\statistics\Insar\Areas\2022_presoon.csv')
    slope_2022 = df_2022_area["Slope"]

    colors = ["#DEF0F3", "#C4E4E9", "#739190"]
    density_with_gt(slope_2022,Slope_landscape ,"Slope (°)", [0, 16], colors)

    TWI_2022 = df_2022_area["DEM"]
    colors = [(194 / 255, 233 / 255, 255 / 255, 1),
              (165 / 255, 200 / 255, 225 / 255, 1),
              (55 / 255, 135 / 255, 192 / 255, 1)
              ]

    # Plot the Altitude distribution picture
    density_with_gt(TWI_2022,altitude_landscape, "Altitude (m)", [4500, 5000], colors)



    # Read the geometric attribute
    df_2022_area = pd.read_csv(r'F:\graduate-project\statistics\Insar\Areas\for_shapes.csv')
    slope_2022 = df_2022_area["Ratio"]
    colors = ["#D8CECB", "#C8BAB5","#9E7E7B" ]
    density(slope_2022, "Aspect ratio", [0, 9], colors)

    slope_2022 = df_2022_area["circularit"]
    colors = ["#BBAFC7", "#A08FB1", "#6F657E"]
    density(slope_2022, "Circularity", [0, 1], colors)

    slope_2022 = df_2022_area["perimeter"]
    slope_2022=slope_2022/1000
    colors = ["#ECF2D1", "#E4EAB9","#D6DF98" ]
    density(slope_2022, "Perimeter (km)", [0, 4], colors)

    df_2022_area = pd.read_csv(r'F:\graduate-project\statistics\Insar\soil\soil_moisture.csv')
    slope_2022 = df_2022_area["RASTERVALU"]
    colors = [(242 / 255, 207 / 255, 211 / 255, 1),
                         (231 / 255, 169 / 255, 177 / 255, 1),
                         "#E2979F"]

    slope_2022=slope_2022*0.0001
    density_with_gt(slope_2022,soil_moisture_landscape*0.0001, "Soil moisture ($\mathregular{m^3}$ /$\mathregular{m^3}$)", [0.15, 0.35], colors)