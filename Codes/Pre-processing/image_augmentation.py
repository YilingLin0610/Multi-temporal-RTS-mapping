# coding=gbk

"""
Augment the training images and their corresponding label images
Modified from: Huang Lingcao
Source: https://github.com/yghlc

"""

import sys,os


from imgaug import augmenters as iaa
from skimage import io


num_classes = 0
code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)



def Flip(image_np, save_dir, input_filename):
    """
    旋转
    Flip image horizontally and vertically
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        file_basename: File base name (e.g basename.tif)

    Returns: True if successful, False otherwise
    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    flipper = iaa.Fliplr(1.0)  # always horizontally flip each input image; Fliplr(P) Horizontally flips images with probability P.
    images_lr = flipper.augment_image(image_np)  # horizontally flip image 0
    save_path = os.path.join(save_dir,  basename + '_fliplr' + ext)
    io.imsave(save_path, images_lr)
    #
    vflipper = iaa.Flipud(1.0)  # vertically flip each input image with 90% probability
    images_ud = vflipper.augment_image(image_np)  # probably vertically flip image 1
    save_path = os.path.join(save_dir, basename + '_flipud' + ext)
    io.imsave(save_path, images_ud)

    return True

def blurer(image_np, save_dir, input_filename,sigma=[1,2]):
    """
    Blur the original images
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        sigma: sigma value for blurring

    Returns: True if successful, False otherwise

    """

    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for value in sigma:
        save_path = os.path.join(save_dir, basename + '_B' + str(value) + ext)
        blurer = iaa.GaussianBlur(value)
        images_b = blurer.augment_image(image_np)
        io.imsave(save_path, images_b)

    return True

def Crop(image_np, save_dir, input_filename,px = [10,30] ):
    """
    Crop the original images
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        px:
        is_groud_true

    Returns: True if successful, False otherwise

    """

    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for value in px:
        crop = iaa.Crop(px=value)
        images_s = crop.augment_image(image_np)
        save_path = os.path.join(save_dir, basename + '_C'+str(value) + ext)
        io.imsave(save_path, images_s)

    return True


def scale(image_np, save_dir, input_filename,scale=[0.5,0.75,1.25,1.5]):
    """
    scale image with 90, 180, 270 degree
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        scale: the scale list for zoom in or zoom out
        is_groud_true:

    Returns: True is successful, False otherwise

    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for value in scale:
        scale = iaa.Affine(scale=value)
        images_s = scale.augment_image(image_np)
        save_path = os.path.join(save_dir, basename + '_S'+str(value).replace('.','') + ext)
        io.imsave(save_path, images_s)

    return True

def batch_operate(input_Path,output_Path):
    """
    Batch process of data augmentation

    Parameters:
        input_Path: The path where the original images are stored
        output_Path: The path where the augmented images will be stored

    Returns:
        None
    """
    try:
        os.mkdir(output_Path)
    except:
        pass
    files = os.listdir(input_Path)
    for i in range(len(files)):
        filename=input_Path+files[i]
        image_np=io.imread(filename)
        basename=os.path.basename(filename)
        scale(image_np,output_Path,basename)
        Crop(image_np,output_Path,basename)
        blurer(image_np,output_Path,basename)



if __name__ == "__main__":
    # The file path where the original training images are stored
    training_path=r'F:\graduate-project\model improvement\2021_model_performance\training data\training_data_jpg\\'
    # The file path where the augmented training images will be stored
    training_path_aug=r"F:\graduate-project\model improvement\2021_model_performance\training data\training_data_jpg_aug"
    batch_operate(training_path, training_path_aug)
    # The file path where the original label images are stored
    label_path=r'F:\graduate-project\model improvement\2021_model_performance\training data\RTS_label_png\\'
    # The file path where the augmented label images will be stored
    label_path_aug=r'F:\graduate-project\model improvement\2021_model_performance\training data\RTS_label_png_aug'
    batch_operate(label_path, label_path_aug)

