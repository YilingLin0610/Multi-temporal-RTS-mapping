# Multi-temporal-RTS-mapping
This repository contains data and codes used in manuscript "Domain Adaptive Semantic Segmentation of  Multi-annual Retrogressive Thaw Slumps"
## TADA Model


 Generally, we implemented a DeepLab V3+ segmentation model with Gradient Reversal Layer and a domain discriminator, in order to achieve transfer learning across remote sensing samples from multiple years.

### Quick Start

For environment configuration, refer to file `env.yaml`

```sh
conda env create -f env.yaml
```

>   Note that there may be excessive requirements that are not needed and may slow down the configuration process. We may fix that later.

In a configured environment, one can simply run

```sh
python train.py
```

to begin training. For testing scripts, some related files are available upon request.

### File Structure

The `datasets` folder should be placed outside this folder. During each time of training, our codes will create a folder outside this name `logs`, and create a sub-folder named with the time training starts, where weights, loss and accuracy plots are saved.

#### Pretrained Model

We adopted mobilenetv2 as the backbone. You can download it via https://pan.bnu.edu.cn/l/X1ns1x, and put it inside sub-folder `model_data`.

```sh
- datasets
-- 2019
--- images
--- labels
--- masks
-- 2020
-- 
 
The multi-annual RTS 
- DANN RTS Segmentation
-- model_data
--- deeplab_mobilenetv2.pth

- logs
```

## Multi-temporal RTS ground truth
The multi-temporal RTS ground truth as well as corresponding non-RTS negative shapefile from 2019 to 2022 are released in

https://github.com/YilingLin0610/Multi-temporal-RTS-mapping/tree/main/Data/Ground-truth-data




