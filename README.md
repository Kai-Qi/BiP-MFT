# Bidirectional Projection-Based Multi-Modal Fusion Transformer for Early Detection of Cerebral Palsy in Infants

##  🧾 1.  Abstract


Periventricular white matter injury (PWMI) is the most common magnetic resonance imaging (MRI) finding in infants with cerebral palsy (CP). This work aims to detect CP and identify subtle, sparse PWMI lesions in infants under two years old with immature brain structures.

To this end, we construct a multi-modal dataset consisting of 243 cases, each with:
- Region masks of five anatomically defined target areas on T1-weighted imaging (T1WI),
- Lesion annotations on T2-weighted imaging (T2WI),
- Diagnostic labels (CP or Non-CP).

We further propose a **Bidirectional Projection-Based Multi-Modal Fusion Transformer (BiP-MFT)**, which integrates cross-modal features using a novel **Bidirectional Projection Fusion Module (BPFM)** to align anatomical regions (T1WI) with lesion patterns (T2WI).

Our BiP-MFT achieves subject-level classification accuracy of **0.90**, specificity of **0.87**, and sensitivity of **0.94**, outperforming nine competing methods by **0.10**, **0.08**, and **0.09**, respectively. Additionally, BPFM surpasses eight alternative fusion strategies based on Transformer and U-Net backbones on our dataset.

Comprehensive ablation studies demonstrate the effectiveness of the proposed annotation strategy and validate the design of the model components.


## 🧠 2.  Architecture
![Image text](architure2.png)


## 🧬 3.  Our Dataset: Infant-PWMl-CP Dataset

The **PWMI-CP MRI dataset** is designed to support research on periventricular white matter injury (PWMI) and its role in predicting the risk of cerebral palsy (CP) in infants. It consists of **243 infant MRI cases**, including:

- 122 infants diagnosed with PWMI:
  - 90 CP cases  
  - 32 non-CP cases  
- 121 healthy controls with normal MRI scans.

The dataset includes **multi-modal MRI sequences** such as T1-weighted and T2-weighted scans, acquired from both 3.0T and 1.5T MRI systems. Expert-annotated lesion segmentations are provided for PWMI cases, making this dataset a valuable resource for studying brain abnormalities and developing automated diagnostic models for early CP risk assessment. The following figure shows the five target regions and typical CP-associated PWMI lesions.

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


##  🏋️‍♂️ 4.  Training on the Infant-PWMl-CP Dataset

### 🔧 Training

Before training, please modify the following file paths in `BiP-MFT-2D_Infant-PWML-CP/train.py`:

- **`total_path`**: The absolute path to the `BiP-MFT-2D_Infant-PWML-CP/` directory.
- **`pretrained_weight_path`**: The path to the SegFormer weights pretrained on ImageNet-1K (`mit_b1.pth`), which can be downloaded from  
  [Google Drive](https://drive.google.com/drive/folders/1yBVICW9lcDANth-RlwJy1C9M6QNXJ0L2?usp=sharing) or  [Baidu Netdisk](https://pan.baidu.com/s/1XiwKp7Ayc81qefs3eu7pGg?pwd=fae8).

- **`data_path`**: The path to the Infant-PWML-CP dataset archive `Infant-PWML-CP.zip` (2.86 GB), downloadable from the same links above.

**Example command for training on Fold 0:**

```
CUDA_VISIBLE_DEVICES=0 python BiP-MFT-2D_Infant-PWML-CP/train.py \
  --w1 0.2 --w2 0.5 --w3 0.1 --w4 0.2 \
  --learn_rate 0.000015 --num_epochs 30 \
  --fold 0 --phi 'mit_b1' --batch_size 5
```


### 🧪 Evaluation

The trained model weights (`last_epoch_weights.pth`) from Fold 0 of the Infant-PWML-CP dataset are available for download:
[Google Drive](https://drive.google.com/drive/folders/1yBVICW9lcDANth-RlwJy1C9M6QNXJ0L2?usp=sharing)  or [Baidu Netdisk](https://pan.baidu.com/s/1XiwKp7Ayc81qefs3eu7pGg?pwd=fae8).

---


## 🚀 5.  Applying Our Model to Brain Tumor Segmentation (BraTS 2021)

### 🔧 Training

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
│   └── brats2021/
│       ├── train/
│       │   ├── BraTS2021_00000/
│       │   │   ├── BraTS2021_00000_t1.nii.gz
│       │   │   ├── BraTS2021_00000_t1ce.nii.gz
│       │   │   ├── BraTS2021_00000_t2.nii.gz
│       │   │   ├── BraTS2021_00000_flair.nii.gz
│       │   │   └── BraTS2021_00000_seg.nii.gz
│       │   ├── BraTS2021_00001/
│       │   └── ...
│
│       ├── val/
│       │   ├── BraTS2021_00800/
│       │   ├── BraTS2021_00801/
│       │   └── ...
│
│       └── test/
│           ├── BraTS2021_01000/
│           ├── BraTS2021_01001/
│           └── ...
```






## 🛠️ 6. Requirements
The following Python packages are required to run our code. We recommend using Python ≥ 3.8 and setting up a virtual environment (e.g., via `conda` or `venv`) for installation.
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


## 📚 7. Citation
If you find this work useful in your research, please cite our paper:

```
@article{qi2025bipmft,
  title     = {Bidirectional Projection-Based Multi-Modal Fusion Transformer for Early Detection of Cerebral Palsy in Infants},
  author    = {Kai Qi and Tingting Huang and Chao Jin and Yizhe Yang and Shihui Ying and Jian Sun and Jian Yang},
  journal   = {IEEE Transactions on Medical Imaging},
  year      = {2025},
  note      = {Accepted with minor revision}
}
```




## 🙏 8. Acknowledgement

We would like to acknowledge the contributions of the following works, which inspired and supported our research:

- Xie, E., Wang, W., Yu, Z., et al. **SegFormer: Simple and efficient design for semantic segmentation with transformers**. *NeurIPS*, 34 (2021), pp. 12077-12090.
- Perera, S., Navard, P., Yilmaz, A. **SegFormer3D: An Efficient Transformer for 3D Medical Image Segmentation**. *CVPR*, 2024, pp. 4981-4988.
- Lin, J., Chen, C., Xie, W., et al. **CKD-TransBTS: Clinical knowledge-driven hybrid transformer with modality-correlated cross-attention for brain tumor segmentation**. *IEEE Transactions on Medical Imaging*, 42(8), 2023, pp. 2451-2461.






