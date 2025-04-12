# Bidirectional Projection-Based Multi-Modal Fusion Transformer for Early Detection of Cerebral Palsy in Infants

##  1. Abstract
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
methods, with 0.10, 0.08, and 0.09 improvements in classification accuracy, specificity, and sensitivity, respectively. Our BPFM outperforms eight compared feature fusion strategies using Transformer and U-Net backbones on our dataset. Ablation studies on the dataset annotations and model components justify the effectiveness of our annotation method and the model rationality.


## 2. Architecture
![Image text](architure2.png)


## 3. Our Dataset: Infant-PWMl-CP Dataset

The PWMI-CP MRI dataset focuses on the study of periventricular white matter injury (PWMI) and its role in predicting the risk of cerebral palsy (CP) in infants. It consists of 243 MRI scan cases, including 122 infants diagnosed with PWMI (90 CP cases and 32 non-CP cases) and 121 infants with normal MRI scans as controls. The dataset includes multi-modal MRI scans, such as T1-weighted and T2-weighted imaging, acquired using 3.0T and 1.5T MRI scanners. The dataset provides annotated lesion regions using expert radiologist segmentation, making it a valuable resource for studying PWMI-related brain abnormalities and developing automated diagnostic models for CP risk assessment. 
![Image text](visualization.png)

The dataset (Infant-PWMl-CP.zip, 2.86GB) and dataset documentation are available for download at [Google Drive](https://drive.google.com/drive/folders/1yBVICW9lcDANth-RlwJy1C9M6QNXJ0L2?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1XiwKp7Ayc81qefs3eu7pGg?pwd=fae8).


The case identifiers used for the training and validation sets in each fold of the five-fold cross-validation are in the `BiP-MFT-2D_Infant-PWML-CP/k-fold-ID/K-fold-ID.txt`.
The directory structure of Infant-PWML-CP is organized as follows:
```
Infant-PWML-CP/
|-- CP/
|   |-- 01001/
|   |   |-- case_01001_T1.nii.gz
|   |   |-- case_01001_T1_seg.nii.gz
|   |   |-- case_01001_T2.nii.gz
|   |   |-- case_01001_T2_seg.nii.gz
|   |-- 01002/
|   |-- 01003/
|   |-- ...
|   |-- 01090/
|
|-- N_CP/
|   |-- 02001/
|   |   |-- case_02001_T1.nii.gz
|   |   |-- case_02001_T1_seg.nii.gz
|   |   |-- case_02001_T2.nii.gz
|   |   |-- case_02001_T2_seg.nii.gz
|   |-- 02002/
|   |-- 02003/
|   |-- ...
|   |-- 02032/
|
|-- Normal/
|   |-- 03001/
|   |   |-- case_03001_T1.nii.gz
|   |   |-- case_03001_T1_seg.nii.gz
|   |   |-- case_03001_T2.nii.gz
|   |-- 03002/
|   |-- 03003/
|   |-- ...
|   |-- 03121/
```


##  4. Training on the Infant-PWMl-CP Dataset

### 🔧 Training

Before training, please modify the following file paths in `BiP-MFT-2D_Infant-PWML-CP/train.py`:

- **`total_path`**: The absolute path to the `BiP-MFT-2D_Infant-PWML-CP/` directory.
- **`pretrained_weight_path`**: The path to the SegFormer weights pretrained on ImageNet-1K (`mit_b1.pth`), which can be downloaded from  
  [Google Drive](https://drive.google.com/drive/folders/1yBVICW9lcDANth-RlwJy1C9M6QNXJ0L2?usp=sharing) or  [Baidu Netdisk](https://pan.baidu.com/s/1XiwKp7Ayc81qefs3eu7pGg?pwd=fae8).

- **`data_path`**: The path to the Infant-PWML-CP dataset archive `Infant-PWML-CP.zip` (2.86 GB), downloadable from the same links above.

**Example command for training on Fold 0:**

```b
CUDA_VISIBLE_DEVICES=0 python BiP-MFT-2D_Infant-PWML-CP/train.py \
  --w1 0.2 --w2 0.5 --w3 0.1 --w4 0.2 \
  --learn_rate 0.000015 --num_epochs 30 \
  --fold 0 --phi 'mit_b1' --batch_size 5
```


### 🧪 Evaluation

The trained model weights (`last_epoch_weights.pth`) from Fold 0 of the Infant-PWML-CP dataset are available for download:
[Google Drive](https://drive.google.com/drive/folders/1yBVICW9lcDANth-RlwJy1C9M6QNXJ0L2?usp=sharing)  or [Baidu Netdisk](https://pan.baidu.com/s/1XiwKp7Ayc81qefs3eu7pGg?pwd=fae8).

---


## 5. Applying Our Model to Brain Tumor Segmentation (BraTS 2021)

### 🧪 Training

Before training on the BraTS 2021 dataset, please update the following paths in `BiP-MFT-3D_Brain2021/main.py`:

- **`path`**: The absolute path to the `BiP-MFT-3D_Brain2021/` directory.
- **`--dataset-folder`**: The path to the BraTS 2021 dataset.

**Example command for training on BraTS 2021:**

```
CUDA_VISIBLE_DEVICES=0 python BiP-MFT-3D_Brain2021/main.py \
  --workers 4 --val 1 --learn_rate2 0.000001 --lr 0.000001 \
  --eta_min 0.0000001 --clip 60 --batch-size 1 \
  --drop_path_rate 0.1 --drop 0.2 --end-epoch 500
```


### 🧪 Evaluation

The pretrained model weights (`best_model.pkl`) trained on the BraTS 2021 dataset can be downloaded from:
[Google Drive](https://drive.google.com/drive/folders/1yBVICW9lcDANth-RlwJy1C9M6QNXJ0L2?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1XiwKp7Ayc81qefs3eu7pGg?pwd=fae8).

### 📁 BraTS2021 dataset

The expected directory structure of the BraTS 2021 dataset is as follows:
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






## 6. Requirements

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


## 7. Citation
```
Kai Qi, Tingting Huang, Chao Jin, Yizhe Yang, Shihui Ying, Jian Sun*, and Jian Yang*. Bidirectional Projection-Based Multi-Modal
Fusion Transformer for Early Detection of Cerebral Palsy in Infants, IEEE Transactions on Medical Imaging, Accept with minor revision, 2025.
```




## 8. Acknowledgement

E. Xie et al. “SegFormer: Simple and efficient design for semantic segmentation with transformers”. NeurIPS 34 (2021), pp. 12077–12090

Perera S, Navard P, Yilmaz A. SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation. CVPR 2024: 4981-4988.

J. Lin et al. “CKD-TransBTS: clinical knowledge-driven hybrid transformer with modality-correlated cross-attention for brain tumor segmentation”. IEEE Trans. Med. Imag. 42.8 (2023), pp. 2451–2461.







