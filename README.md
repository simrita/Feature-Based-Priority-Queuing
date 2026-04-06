# Feature-Driven Priority Queuing — Replication Code

**GitHub Repository:** https://github.com/simrita/Feature-Based-Priority-Queuing

This repository contains the code and pre-trained model weights to reproduce the results in:

> **Feature-Driven Priority Queuing**
> *Operations Research* (2026)

## Overview

We study feature-driven priority queuing, where job types are unobserved and must be inferred from observable features using a classifier. We compare two implementations: (1) **type-first**, which predicts type probabilities and then maps them to queues, and (2) **direct**, which maps features directly to queue probabilities in an end-to-end manner. The direct approach is trained using a novel *queuing loss* that minimizes empirical waiting cost.

## Repository Structure

```
.
├── README.md
├── 01_stylized_example/          # Example 3 (Section 4, Table 1)
│   └── Example3_Direct_vs_TypeFirst.Rmd
├── 02_theoretical_analysis/      # Propositions 4-6, Section 3.3
│   ├── Proposition4_5_LowerBound_2Queues.Rmd
│   ├── Proposition6_Optimal_3Types_2Queues.Rmd
│   ├── Feature_Overlap_Analysis.Rmd
│   └── Optimal_Configurations.Rmd
├── 03_chest_xray_case_study/     # Section 5: NIH ChestX-ray14 experiments
│   ├── training/
│   │   ├── 01_Train_Sequential_TypeFirst.ipynb   # Type-first MobileNet training
│   │   └── 02_Train_Direct.ipynb                 # Direct approach training
│   ├── comparison/
│   │   └── 03_Compare_Sequential_vs_Direct.ipynb # Tables 3, 4, Eq. 28
│   └── model_weights/                            # Pre-trained HDF5 weights
│       ├── sequential_weights.hdf5
│       ├── direct_linear_1.8_rho_0.9.hdf5
│       ├── direct_linear_10_rho_0.9.hdf5
│       ├── direct_convex_1.5_rho_0.9.hdf5
│       └── direct_convex_1.8_rho_0.9.hdf5
├── 04_knn_sensitivity/           # Section 5.3, Figure 3
│   ├── kNN_Sensitivity_Analysis.ipynb
│   └── Figure3_kNN_Waiting_Cost.ipynb
└── utils/                        # Shared R utility functions
    ├── cost_functions.R          # Queuing cost computations (Eq. 2-3)
    └── header.R                  # R package dependencies
```

## Mapping to Paper

| Paper Section | Table/Figure | Code File |
|---|---|---|
| Section 4 (Example 3) | Table 1 | `01_stylized_example/Example3_Direct_vs_TypeFirst.Rmd` |
| Section 3.3 | Figure 2, Propositions 4-5 | `02_theoretical_analysis/Proposition4_5_LowerBound_2Queues.Rmd` |
| Section 3.3.1 | Proposition 6 | `02_theoretical_analysis/Proposition6_Optimal_3Types_2Queues.Rmd` |
| Section 5.1 (Type-First) | Model training | `03_chest_xray_case_study/training/01_Train_Sequential_TypeFirst.ipynb` |
| Section 5.1 (Direct) | Model training | `03_chest_xray_case_study/training/02_Train_Direct.ipynb` |
| Section 5.2 | Tables 3, 4, Eq. 28 | `03_chest_xray_case_study/comparison/03_Compare_Sequential_vs_Direct.ipynb` |
| Section 5.3 | Figure 3 | `04_knn_sensitivity/kNN_Sensitivity_Analysis.ipynb` |
| Appendix 7.2 | Figure 5 (ROC curves) | `03_chest_xray_case_study/training/01_Train_Sequential_TypeFirst.ipynb` |

## Requirements

### R (for Sections 1-2: Stylized Example and Theoretical Analysis)

- R >= 4.0
- Packages: `ggplot2`, `dplyr`, `tidyr`, `zoo`, `reshape2`, `magrittr`, `corrplot`, `knitr`, `readxl`, `lubridate`, `gtools`, `lhs`, `MASS`, `distr`, `triangulr`

Install via:
```r
install.packages(c("ggplot2", "dplyr", "tidyr", "zoo", "reshape2",
                    "magrittr", "corrplot", "knitr", "readxl", "lubridate",
                    "gtools", "lhs", "MASS", "distr", "triangulr"))
```

### Python (for Section 5: Chest X-ray Case Study)

- Python >= 3.8
- TensorFlow >= 2.6
- NumPy, Pandas, Matplotlib, scikit-learn
- imbalanced-learn (for `RandomOverSampler`)
- SciPy (for optimization in type-first queue assignment)
- h5py (for loading model weights)
- seaborn (for visualization)

Install via:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn imbalanced-learn scipy h5py seaborn
```

## Data

### Chest X-ray14 Dataset (Section 5)

The chest X-ray experiments use the publicly available NIH ChestX-ray14 dataset (112,120 frontal chest X-ray images, 14 disease labels):

> Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM.
> *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.*
> IEEE CVPR, 2017.

**Option 1 — Download via Kaggle CLI (recommended):**

Install the Kaggle CLI if you don't have it, then download the dataset:
```bash
pip install kaggle
kaggle datasets download -d nih-chest-xrays/data
```
This downloads a single zip file (~42 GB). Unzip it into a `data/` directory one level above the code folder:
```bash
mkdir -p data
unzip data.zip -d data/
```

**Option 2 — Download from NIH Box:**

https://nihcc.app.box.com/v/ChestXray-NIHCC

After downloading (or unzipping), your directory structure should look like this:
```
├── data/
│   ├── Data_Entry_2017.csv
│   ├── images_001/images/*.png
│   ├── images_002/images/*.png
│   ├── ...
│   └── images_012/images/*.png
└── 03_chest_xray_case_study/
    ├── training/
    ├── comparison/
    └── model_weights/
```

The notebooks load images from `../data/`, so `data/` must sit as a sibling to the `03_chest_xray_case_study/` folder.

### Pre-trained Model Weights

Pre-trained MobileNet weights are provided in `03_chest_xray_case_study/model_weights/` for reproducing the comparison results (Tables 3-4) without retraining. The weights correspond to:

- `sequential_weights.hdf5`: Type-first disease classifier (binary cross-entropy)
- `direct_linear_1.8_rho_0.9.hdf5`: Direct classifier, c_i = 1.8(T-i)+1, rho=0.9
- `direct_linear_10_rho_0.9.hdf5`: Direct classifier, c_i = 10(T-i)+1, rho=0.9
- `direct_convex_1.5_rho_0.9.hdf5`: Direct classifier, c_i = 1.5^(T-i), rho=0.9
- `direct_convex_1.8_rho_0.9.hdf5`: Direct classifier, c_i = 1.8^(T-i), rho=0.9

## How to Reproduce Results

### Example 3 and Table 1 (Stylized Example)

```r
# Open in RStudio and knit, or run from command line:
Rscript -e "rmarkdown::render('01_stylized_example/Example3_Direct_vs_TypeFirst.Rmd')"
```

### Theoretical Analysis (Propositions 4-6, Figure 2)

```r
Rscript -e "rmarkdown::render('02_theoretical_analysis/Proposition4_5_LowerBound_2Queues.Rmd')"
Rscript -e "rmarkdown::render('02_theoretical_analysis/Proposition6_Optimal_3Types_2Queues.Rmd')"
```

### Chest X-ray Case Study (Tables 3-4)

To reproduce using pre-trained weights (recommended):
1. Download the ChestX-ray14 dataset and place in `data/`
2. Run `03_chest_xray_case_study/comparison/03_Compare_Sequential_vs_Direct.ipynb`

To retrain models from scratch:
1. Run `03_chest_xray_case_study/training/01_Train_Sequential_TypeFirst.ipynb`
2. Run `03_chest_xray_case_study/training/02_Train_Direct.ipynb`
3. Run `03_chest_xray_case_study/comparison/03_Compare_Sequential_vs_Direct.ipynb`

### k-NN Sensitivity Analysis (Figure 3)

Run `04_knn_sensitivity/kNN_Sensitivity_Analysis.ipynb`

## Notes

- All experiments use a single-server non-preemptive priority queue with T=14 image types and N=4 queues (Critical, Urgent, Important, Routine).
- Server utilization is set to rho=0.9 throughout.
- The MobileNet architecture follows Garyfallos et al. (2019) with 512×512 RGB input. The type-first model is initialized with ImageNet pre-trained weights; the direct models are trained from scratch (`weights=None`).
- Training uses RandomOverSampler to balance the dataset across types.
- Results in Table 3 report mean and standard deviation over 10 random samples of 2,000 test images each.
