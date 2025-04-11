# Bidirectional Projection-Based Multi-Modal Fusion Transformer for Early Detection of Cerebral Palsy in Infants



##  Abstract
Periventricular white matter injury (PWMI) is the most frequent magnetic resonance imaging (MRI) ﬁnding in infants with Cerebral Palsy (CP).
We aim to detect CP and identify subtle, sparse PWMI lesions in infants under two years of age with immature brain structures.
Based on the characteristic that the responsible lesions are located within five target regions,
we first construct a multi-modal dataset including 310 cases with the mask annotations of five target regions for delineating anatomical structures on T1-Weighted Imaging (T1WI) images, masks for lesions on T2-Weighted Imaging (T2WI) images, and categories (CP or Non-CP).
Furthermore,
we develop a bidirectional projection-based multi-modal
fusion transformer (BiP-MFT), incorporating a ***Bidirectional Projection Fusion Module*** (BPFM) for integrating the features between five target regions on T1WI images and lesions on T2WI images.
Our BiP-MFT achieves subject-level classification
accuracy of 0.90, specificity of 0.87, and sensitivity of
0.94. It surpasses the best results of nine comparative
methods, with 0.10, 0.08, and 0.09 improvements in classification accuracy, specificity and sensitivity respectively. Our BPFM outperforms eight compared feature fusion strategies using Transformer and U-Net backbones on our dataset. Ablation studies on the dataset annotations and model components justify the effectiveness of our annotation method and the model rationality.


## Architecture
![Image text](visualization.png)


## Our Dataset: PWMI-CP MRI Dataset

The PWMI-CP MRI dataset focuses on the study of periventricular white matter injury (PWMI) and its role in predicting the risk of cerebral palsy (CP) in infants. It consists of 310 MRI scan cases, including 122 infants diagnosed with PWMI (90 CP cases and 32 non-CP cases) and 188 infants with normal MRI scans as controls. The dataset includes multi-modal MRI scans, such as T1-weighted and T2-weighted imaging, acquired using 3.0T and 1.5T MRI scanners. The dataset provides annotated lesion regions using expert radiologist segmentation, making it a valuable resource for studying PWMI-related brain abnormalities and developing automated diagnostic models for CP risk assessment.

![Image text](architure2.png)


##  Training on PWMI-CP MRI Dataset

The file path should be changed.


Download weights of SegFormer (mit-b1.pth) pre-trained on ImageNet-1K, and put them in a folder

[https://github.com/NVlabs/SegFormer?tab=readme-ov-file](https://connecthkuhk-my.sharepoint.com/personal/xieenze_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxieenze%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fsegformer%2Fpretrained%5Fmodels&ga=1)



##  Applying our model to Brain Tumor Segmentation (BraTS challenge 2021)

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

E. Xie et al. “SegFormer: Simple and efficient design for semantic segmentation with transformers”. NeurIPS 34 (2021), pp. 12077–12090

Perera S, Navard P, Yilmaz A. SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation. CVPR 2024: 4981-4988.








