# PTB-XL — Deep Learning for 12-lead ECG Classification and Representation

Multi-label classification of 12-lead ECGs from the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset into the five diagnostic superclasses (**NORM**, **MI**, **STTC**, **CD**, **HYP**), and exploration of learned ECG representations:

1. **Baselines** — 1D CNN vs 1D ResNet trained directly on the raw 12-lead signals.
2. **Lead-to-lead autoencoder** — a 1D U-Net that takes a single (randomly chosen) lead and reconstructs the full 12-lead ECG; its bottleneck gives a per-lead embedding, analysed with PCA/UMAP.
3. **ECG as a point cloud** — each ECG becomes a set of points `(time, lead, voltage, embedding)` built from the signal and the U-Net embeddings, then classified with a PointNet (including a full version with input/feature transforms).

## Repository layout

```
├── data/ptb-xl-…-1.0.3/     PTB-XL v1.0.3 (100 Hz records + ptbxl_database.csv)
├── models/                  Architectures
│   ├── cnn1D.py                    1D CNN baseline
│   ├── resnet1D.py                 1D ResNet
│   ├── UNet1D.py                   lead-to-12-lead autoencoder
│   ├── pointnet.py                 PointNet
│   └── pointnet_full_ecg.py        PointNet with input/feature transforms
├── src/                     Data pipeline and training loops
│   ├── dataset.py                  PTB-XL loading, labels, splits
│   ├── pointcloud_ecg.py           signal + embeddings → point cloud
│   ├── train.py                    classifier training / evaluation
│   ├── train_autoencoder.py        U-Net training
│   └── train_pointnet.py, train_newpointnet.py
├── notebooks/               Analysis, in reading order
│   ├── 0_(initial) Data_loading_and_overview.ipynb
│   ├── 1_EDA_and_Visualizations.ipynb       demographics, class imbalance, co-occurrence
│   ├── 2_Basline1_CNN_VS_RESNET.ipynb       baselines
│   ├── 3_autoencoder_UNet1D.ipynb           U-Net, 12×12 lead reconstruction matrix, embeddings
│   ├── 3_Full_PointNet_ECG.ipynb            full PointNet on ECG point sets
│   ├── 4_pointcloud_classification_(try).ipynb
│   ├── best_model.pt, unetecg.pth           trained weights
├── results/                 Cleaned metadata with superclass labels
└── Methods_and_models_for_ECG_analysis_and_classification.pdf   report
```

## Results (test set, 5 superclasses, multi-label)

| Model | Macro AUC | Macro F1 | Macro precision | Macro recall |
|---|---|---|---|---|
| 1D CNN | **0.880** | 0.635 | 0.540 | 0.821 |
| 1D ResNet | 0.873 | **0.646** | 0.649 | 0.646 |
| PointNet on ECG point cloud (first try) | – | 0.533 | 0.710 | 0.464 |

U-Net lead-to-lead reconstruction: mean off-diagonal MSE 0.505, mean Pearson correlation 0.680 over the 12×12 source/target lead matrix.

HYP is the hardest class (strongly under-represented), and NORM dominates the dataset; see notebook 1 for the class imbalance analysis.

## Getting started

```bash
git clone https://github.com/somehcene/ptbxl5d.git
cd ptbxl5d
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

The PTB-XL dataset (v1.0.3) is included under `data/`. It can also be downloaded from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).

## Reference

Wagner, P., Strodthoff, N., Bousseljot, R.-D., et al. *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data 7, 154 (2020).
