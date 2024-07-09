"""
Convert label TIFF files into png files
Modified from: bubbliiiing
Source: https://github.com/bubbliiiing

"""

import os

import numpy as np
from PIL import Image
from tqdm import tqdm

#The original path where the label TIFF files stored
Origin_SegmentationClass_path   = r"F:\graduate-project\domain adversarial training\2019_new_datasets\non_RTS_label_patches"
# The file path where the converted PNG files will be stored
Out_SegmentationClass_path       = r"F:\graduate-project\domain adversarial training\2019_new_datasets\non_RTS_label_patches_pngs"
Origin_Point_Value              = np.array([0, 1])
Out_Point_Value                 = np.array([0, 1])

if __name__ == "__main__":
    if not os.path.exists(Out_SegmentationClass_path):
        os.makedirs(Out_SegmentationClass_path)
    png_names = [x for x in os.listdir(Origin_SegmentationClass_path) if x.endswith(".tif")]
    for png_name in tqdm(png_names):
        png     = Image.open(os.path.join(Origin_SegmentationClass_path, png_name))
        w, h    = png.size
        png     = np.array(png)
        out_png = np.zeros([h, w])
        for i in range(len(Origin_Point_Value)):
            mask = png[:, :] == Origin_Point_Value[i]
            if len(np.shape(mask)) > 2:
                mask = mask.all(-1)
            out_png[mask] = Out_Point_Value[i]
        out_png = Image.fromarray(np.array(out_png, np.uint8))
        out_png_name = Out_SegmentationClass_path + '/' + png_name[0:-4] + '.png'
        out_png.save(out_png_name)
