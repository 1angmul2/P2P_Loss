## Installation

### Requirements

- Linux
- Python 3.7
- PyTorch 1.7.1
- CUDA 11.0 or higher
- NCCL 2
- GCC(G++) **7.5** or higher
- [mmcv](https://github.com/open-mmlab/mmcv)==**0.2.14**

Note some cuda extensions, e.g., ```box_iou_rotated``` and ```nms_rotated``` require pytorch>=1.3 and gcc>=4.9.

We have tested the following versions of OS and softwares:

- OS:  ubuntu18.04
- CUDA: 11.0
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC(G++): 7.5
- pytorch: 1.7.1

### Install

a. Create a conda virtual environment and activate it.

```shell
conda create -n p2ploss python=3.7 -y
conda activate p2ploss
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0 -c pytorch
```

c. Clone the s2anet repository.

```shell
git clone https://github.com/SZUYangY/P2P_Loss.git
cd P2P_Loss
```

d. Install s2anet

```shell
pip install -r requirements.txt

python setup.py develop
# or "pip install -v -e ."
```

### Install DOTA_devkit
```
sudo apt-get install swig
cd DOTA_devkit/polyiou
swig -c++ -python csrc/polyiou.i
python setup.py build_ext --inplace
```

### Prepare datasets

For DOTA, we provide scripts to split the original images into chip images (e.g., 1024*1024), and convert annotations to mmdet's format. Please refer to [DOTA_devkit/prepare_dota1_ms.py](../DOTA_devkit/prepare_dota1_ms.py).

It is recommended to symlink the dataset root to `$MMDETECTION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── DOTA
│   │   ├── dota_1024
│   │   │    ├── trainval_split
│   │   │    │    │─── images
│   │   │    │    │─── labelTxt
│   │   │    │    │─── trainval1024.pkl
│   │   │    ├── test_split
│   │   │    │    │─── images
│   │   │    │    │─── test1024.pkl
│   ├── HRSC2016 (optional)
│   │   ├── Train
│   │   │    │─── AllImages
│   │   │    │─── Annotations
│   │   │    │─── train.txt
│   │   ├── Test
│   │   │    │─── AllImages
│   │   │    │─── Annotations
│   │   │    │─── test.txt
```

Note `train.txt` and `test.txt` in HRSC2016 are `.txt` files recording image names without extension.

For example:
```
P00001
P00002
...
```
