import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class DeeplabDataset(Dataset):
    def __init__(self, input_shape, num_classes, train, dataset_path_src, dataset_path_tgt, annotation_lines=None):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines = annotation_lines
        # self.length             = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path_src = dataset_path_src
        self.dataset_path_tgt = dataset_path_tgt

        self.img_path_src = os.path.join(self.dataset_path_src, 'images')
        self.img_list_src = os.listdir(self.img_path_src)
        self.mask_path_src = os.path.join(self.dataset_path_src, 'masks')
        self.mask_list_src = os.listdir(self.mask_path_src)
        self.label_path_src = os.path.join(self.dataset_path_src, 'labels')
        self.label_list_src = os.listdir(self.label_path_src)

        self.img_path_tgt = os.path.join(self.dataset_path_tgt, 'images')
        self.img_list_tgt = os.listdir(self.img_path_tgt)
        self.mask_path_tgt = os.path.join(self.dataset_path_tgt, 'masks')
        self.mask_list_tgt = os.listdir(self.mask_path_tgt)
        self.label_path_tgt = os.path.join(self.dataset_path_tgt, 'labels')
        self.label_list_tgt = os.listdir(self.label_path_tgt)

    # 考虑到source和target数据集可能大小不同，暂时与两者中最小值对齐
    #  并通过align_dataset成员判断与两者哪一个对其
    def __len__(self):
        self.length = min(len(self.img_list_src), len(self.img_list_tgt))
        if self.length == len(self.img_list_src):
            self.align_dataset = 'source'
        elif self.length == len(self.img_list_tgt):
            self.align_dataset = 'target'
        return self.length

    def __getitem__(self, index):
        # annotation_line = self.annotation_lines[index]
        # name = annotation_line.split()[0]
        img_list = []
        mask_list = []
        seg_label_list = []
        domain_label_list = [0, 1]  # 默认source在前，target在后

        # -------------------------------#
        #   从文件中读取图像
        # -------------------------------#
        # jpg = Image.open(os.path.join(os.path.join(self.dataset_path_src, "VOC2007/JPEGImages"), name + ".jpg"))
        # png = Image.open(os.path.join(os.path.join(self.dataset_path_src, "VOC2007/SegmentationClass"), name + ".png"))
        jpg_src = Image.open(os.path.join(self.img_path_src, self.img_list_src[index]))
        png_src = Image.open(os.path.join(self.mask_path_src, self.mask_list_src[index]))
        # -------------------------------#
        #   数据增强
        # -------------------------------#
        jpg_src, png_src = self.get_random_data(jpg_src, png_src, self.input_shape, random=self.train)

        # jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        jpg_src = np.transpose(preprocess_input(np.array(jpg_src, dtype=np.float32)), [2, 0, 1])  # 可能不需要转置
        png_src = np.array(png_src, dtype=np.int64)
        png_src[png_src >= self.num_classes] = self.num_classes
        img_list.append(jpg_src)
        mask_list.append(png_src)
        # -------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        # -------------------------------------------------------#
        seg_labels_src = np.eye(self.num_classes + 1)[png_src.reshape([-1])]
        seg_labels_src = seg_labels_src.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        jpg_tgt = Image.open(os.path.join(self.img_path_tgt, self.img_list_tgt[index]))
        png_tgt = Image.open(os.path.join(self.mask_path_tgt, self.mask_list_tgt[index]))
        # -------------------------------#
        #   数据增强
        # -------------------------------#
        jpg_tgt, png_tgt = self.get_random_data(jpg_tgt, png_tgt, self.input_shape, random=self.train)

        # jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        jpg_tgt = np.transpose(preprocess_input(np.array(jpg_tgt, dtype=np.float32)), [2, 0, 1])  # 可能不需要转置
        png_tgt = np.array(png_tgt, dtype=np.int64)
        png_tgt[png_tgt >= self.num_classes] = self.num_classes
        img_list.append(jpg_tgt)
        mask_list.append(png_tgt)

        seg_labels_tgt = np.eye(self.num_classes + 1)[png_tgt.reshape([-1])]
        seg_labels_tgt = seg_labels_tgt.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        # seg_labels_src = np.load(os.path.join(self.label_path_src, self.label_list_src[index]))
        seg_label_list.append(seg_labels_src)
        # seg_labels_tgt = np.load(os.path.join(self.label_path_tgt, self.label_list_tgt[index]))
        seg_label_list.append(seg_labels_tgt)
        # domain_labels = np.zeros(1)
        # if self.dataset_path_src[11:15] == '2022':
        #     domain_labels[0] = 1

        # return jpg, png, seg_labels, self.img_list_src[index], domain_labels
        dir_src = os.path.join(self.img_path_src, self.img_list_src[index])
        dir_tgt = os.path.join(self.img_path_tgt, self.img_list_tgt[index])
        return img_list, mask_list, seg_label_list, domain_label_list, dir_src, dir_tgt

    def get_img_list_src(self):
        return self.img_list_src

    def rand(self, a: float = 0, b: float = 1) -> float:
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        # ------------------------------------------#
        #   高斯模糊
        # ------------------------------------------#
        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        # ------------------------------------------#
        #   旋转
        # ------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


# DataLoader中collate_fn使用
# 为了实现batch前半和后半分离，实际的batch会是参数设置的两倍
def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    domain_labels = []
    for img, png, seg_label, domain_label, dir_src, dir_tgt in batch:
        # images.append(img)
        # pngs.append(png)
        # seg_labels.append(labels)
        # domain_labels.append(domain_label)
        list_insert(img, images)
        list_insert(png, pngs)
        list_insert(seg_label, seg_labels)
        list_insert(domain_label, domain_labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    domain_labels = torch.from_numpy(np.array(domain_labels)).type(torch.FloatTensor).unsqueeze(1)
    return images, pngs, seg_labels, domain_labels


# l1 -> l2
def list_insert(l1: list, l2: list):
    if len(l2) == 0:
        for item in l1:
            l2.append(item)
    else:  # 目前只考虑一个source和一个target
        l2.insert(len(l2)//2, l1[0])
        l2.append(l1[1])
