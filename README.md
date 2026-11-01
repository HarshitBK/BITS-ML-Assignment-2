# Machine Learning Assignment 2: Breast Cancer Classification

**Course:** M.Tech (AIML) - Machine Learning  
**Institution:** BITS Pilani (Work Integrated Learning Programmes Division)   

---

## a. Problem Statement

Breast cancer is one of the most diagnosed life-threatening cancers globally. Early detection and precise classification of breast lesions into **Benign** (non-cancerous) and **Malignant** (cancerous) is vital for clinical intervention, personalized treatment planning, and significantly increasing survival rates.

The goal of this assignment is to:
1. Develop an end-to-end Machine Learning classification pipeline evaluating **5 distinct supervised learning algorithms** on the Wisconsin Breast Cancer Diagnostic dataset.
2. Evaluate each model using **6 standardized performance metrics**: Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).
3. Build and deploy an interactive **Streamlit web application** enabling dynamic model selection, custom test dataset upload, live metric calculations, confusion matrix visualization, ROC curves, and single-sample clinical simulations.

---

## b. Dataset Description

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:**  Kaggle (`usman44m/breast-cancer-tabular-dataset`). Also added in the repository as `data.csv`. 
- **Number of Instances:** **569** 
- **Number of Features:** **30 continuous numeric predictive attributes** 
- **Target Attribute:** `diagnosis` (Binary classification: `M` = Malignant, `B` = Benign)
- **Class Distribution:**
  - Benign (`B` / `0`): 357 instances (62.7%)
  - Malignant (`M` / `1`): 212 instances (37.3%)
- **Data Split:**
  - **Training Set (80%):** 455 instances (used for model fitting and scaler calibration)
  - **Testing Set (20%):** 114 instances (preserved in `test_data.csv` for independent evaluation and app upload)

### Attribute Details
Features are computed from digitized images of Fine Needle Aspirates (FNA) of breast masses, describing characteristics of the cell nuclei present:
1. **radius** (mean of distances from center to points on the perimeter)
2. **texture** (standard deviation of gray-scale values)
3. **perimeter**
4. **area**
5. **smoothness** (local variation in radius lengths)
6. **compactness** ($\text{perimeter}^2 / \text{area} - 1.0$)
7. **concavity** (severity of concave portions of the contour)
8. **concave points** (number of concave portions of the contour)
9. **symmetry**
10. **fractal dimension** ("coastline approximation" - 1)

For each image, the **Mean**, **Standard Error (SE)**, and **"Worst" (mean of the three largest values)** are recorded for each of the 10 characteristics, yielding $10 \times 3 = 30$ continuous attributes. Non-predictive identifiers (`id`) and empty trailing columns (`Unnamed: 32`) were removed during data preprocessing.

---


