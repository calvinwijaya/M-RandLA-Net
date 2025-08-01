# M-RandLA-Net
This repository contain personal modification of RandLA-Net implemented in Pytorch from [here](https://github.com/aRI0U/RandLA-Net-pytorch). Please check the [original repository](https://github.com/QingyongHu/RandLA-Net) and the [paper](https://arxiv.org/abs/1911.11236) by Hu et al. (2019) for complete and details explanation of RandLA-Net. RandLA-Net is a simple and efficient neural architecture for semantic segmentation of large-scale 3D point clouds. This modification of RandLA-Net used to learn how the code structured and works, by changing it into binary problem to segment ground and off-ground point cloud. The modification that has been made into the script in this repository:

1. Change from S3DIS as input file into our sample dataset
2. Change how data input prepared (changing folder order): Before in > `datasets/s3dis/S3DIS_v1.2` into `data` only
3. Change `prepare_s3dis.py` into `prepare_data.py` which change how data is read (check the file)
4. Change class into off-ground (0) and ground (1)
5. Change data input and output into LAS file with UTM compatibility
6. Add several parser argument for user as input

## Prerequisites & Instalations
This implementation has been tested on Windows 11. The original repo implemented in Tensorflow 1, while this repo is modification from [here](https://github.com/aRI0U/RandLA-Net-pytorch) which use Pytorch. Check the Installation guide in [this repository](https://github.com/aRI0U/RandLA-Net-pytorch) for details.

1. Clone this repository
2. Install all Python dependencies
   ```
   cd RandLA-Net-pytorch
   pip install -r requirements.txt
   ```
   **Common issue**: the setup file from torch-points-kernels package needs PyTorch to be previously installed. You may thus need to install PyTorch first and then `torch-points-kernels`. Check [this](https://pytorch.org/get-started/locally/) to install pytorch, while for `torch-point-kernels` use:
   ```
   pip install torch-points-kernels
   ```
4. Install Microsoft Visual Basic for C++
5. Install `cpp_subsampling`
   ```
   cd utils/cpp_wrappers/cpp_subsampling
   python setup.py build_ext --inplace
   ```
7. You should now be able to train RandLA-Net models

# How to use
This repository only try Semantic Segmentation to train S3DIS data type on scene segmentation task. For other use please check the original documentation.

## Data
1. Place LAS point cloud data into `./data` folder. Now, you can use LAS point cloud with UTM compatibility. To help determine which area used for test, change the prefix of filename into number_{filename}. Example:
   ```
   1_Data train 1.las
   2_Data train 2.las
   3_Data test.las
   ```
   ![image](https://github.com/calvinwijaya/M-RandLA-Net/assets/88726143/6b204d73-3b15-44d3-8fe3-04d34b4a07fe)

2. After las data is inside `data` folder, run
   ```
   python prepare_data.py
   ```
   It will create a new folder called `train` which contains *.NPY files as much as las data in `data` folder.
   ![image](https://github.com/calvinwijaya/M-RandLA-Net/assets/88726143/c9cf7eab-37ed-4836-8236-e70c2be54704)

4. Finally, in order to subsample the point clouds using a grid subsampling, run:
   ```
   python subsample_data.py
   ```
   It will create a new folder called `subsampled/train` which contains subsample result.
   ![image](https://github.com/calvinwijaya/M-RandLA-Net/assets/88726143/99a0a900-0c1f-40e7-b9eb-77a145a70449)


## Training
Simply run the following script to start the training:
```
python train.py
```
Several parser arguments that user can define:

- `dataset`, Location of `subsampled` folder, default: `data/subsampled`
- `epochs`, Number of epochs, default: 100
- `log_dir`, Log directory to save or store the model
- `load`, Path to load trained model

To load and retrain previous model:
```
python train.py --load previous_model
```

**Temporary** To change the test dataset, change in `utils/data.py` in line 74:
![image](https://github.com/calvinwijaya/M-RandLA-Net/assets/88726143/4bc7c058-5be9-4828-affb-27f50064b101)

Training process:

![image](https://github.com/calvinwijaya/M-RandLA-Net/assets/88726143/db1845df-f894-4d79-85f4-defc46617749)


## Test
Simply run the following script to start the test:
```
python test.py
```
Several parser arguments that user can define:

(will be updated)

Testing process:

![image](https://github.com/calvinwijaya/M-RandLA-Net/assets/88726143/431445ba-3f79-45b0-8058-c1c1f545b97e)


# Citation
I do not own any of these code, please cite the original paper:
```
@article{hu2019randla,
  title={RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds},
  author={Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}

@article{hu2021learning,
  title={Learning Semantic Segmentation of Large-Scale Point Clouds with Random Sampling},
  author={Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```
