# Bidirectional Projection-Based Multi-Modal Fusion Transformer for Early Detection of Cerebral Palsy in Infants

Code release is forthcoming.



##  Abstract




## Architecture

![Image text](architure2.png)



##  Training

The file path should be changed.


Download weights of SegFormer (MiT-B1) pre-trained on ImageNet-1K, and put them in a folder

https://github.com/NVlabs/SegFormer?tab=readme-ov-file



##  Brain Tumor Segmentation: BraTS challenge 2021

Please put the BraTS2021 dataset into dataset/ folder and it's structure should be like below:
```
├── dataset/
│   ├── brats2021
│   │   ├── train
│   │   │     ├── BraTS2021_00000
│   │   │	  │		    ├──BraTS2021_00000_t1.nii.gz
│   │   │	  │		    ├──BraTS2021_00000_t1ce.nii.gz
│   │   │	  │		    ├──BraTS2021_00000_t2.nii.gz
│   │   │	  │		    ├──BraTS2021_00000_flair.nii.gz
│   │   │	  │		    └──BraTS2021_00000_seg.nii.gz
│   │   │     ├── BraTS2021_00001   
│   │   │     └── ...
│   │   │        
│   │   ├── val
│   │   |     ├── BraTS2021_00800
│   │   |     ├── BraTS2021_00801
│   │   |     └── ...
│   │   |     
│   │   └── test
│   │         ├── BraTS2021_01000        
│   |         ├── BraTS2021_01001
│   |         └── ...
```






## Requirements

```
torch==1.10.0+cu113
mmcv==1.6.1
mmcv_full==1.6.1
numpy==1.24.4
opencv_python==4.7.0.72
Pillow==8.2.0
scikit_learn==0.24.1
scipy==1.13.1
```


## Citation





## Acknowledgement

Our model is based on:

E. Xie et al. “SegFormer: Simple and efficient design for semantic segmentation with transformers”. NeurIPS 34 (2021), pp. 12077–12090

Perera S, Navard P, Yilmaz A. SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation. CVPR 2024: 4981-4988.








