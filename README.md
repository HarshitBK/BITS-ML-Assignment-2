# Machine Learning Assignment 2: Breast Cancer Classification (Malignant v/s Benign)

**Course:** M.Tech (AIML) - Machine Learning  
**Institution:** BITS Pilani (WILP)   

---

## a. Problem Statement

Breast cancer is one of the most diagnosed life-threatening cancers globally. Early detection and precise classification of breast lesions into **Benign** (non-cancerous) and **Malignant** (cancerous) is vital for clinical intervention, personalized treatment planning, and significantly increasing survival rates.

The goal of this assignment is to:
1. Develop an end-to-end Machine Learning classification pipeline evaluating **5 distinct supervised learning algorithms** on the Wisconsin Breast Cancer Diagnostic dataset.
2. Evaluate each model using **6 standardized performance metrics**: Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).
3. Build and deploy an interactive **Streamlit web application** `https://2026aj05631-bits-ml-assignment-2.streamlit.app/` enabling dynamic model selection, custom test dataset upload, live metric calculations, confusion matrix visualization, ROC curves, and single-sample clinical simulations.

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

## c. GitHub Repository Link

- **GitHub Repository:** `https://github.com/HarshitBK/BITS-ML-Assignment-2`  
 

### Repository Layout
```text
project-folder/
│-- app.py                  # Streamlit web application frontend and logic
│-- requirements.txt        # Deployment dependencies
│-- README.md               # Complete assignment documentation & report
│-- test_data.csv           # 20% held-out test data (114 samples with ground truth)
│-- data.csv                # Complete raw dataset (569 samples)
│-- train_models.py         # Full model training and evaluation script
│-- model/                  # Serialized artifacts directory
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn.joblib
    │-- naive_bayes.joblib
    │-- svm.joblib
    │-- scaler.joblib       # Fitted StandardScaler instance
    │-- features.json       # Canonical ordered list of 30 feature names
    │-- metrics_summary.json# Computed test evaluation metrics
    
```

---

## d. Models Used

Five supervised classification models were trained and evaluated on identical stratified train-test splits:
1. **Logistic Regression** (L2 penalty, maximum iterations = 1000, standardized features)
2. **Decision Tree Classifier** (Gini impurity, max depth = 5, min samples split = 4)
3. **K-Nearest Neighbors (KNN)** (k = 5, distance-weighted voting, standardized features)
4. **Naive Bayes Classifier** (GaussianNB, standardized features)
5. **Support Vector Machine (SVM)** (Radial Basis Function [RBF] kernel, $C=1.0$, probability enabled)

### Comparison Table

The evaluation metrics calculated on the independent held-out test set ($N=114$) are presented below:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9649 | **0.9960** | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| **Decision Tree** | 0.9211 | 0.9676 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| **KNN** | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| **Naive Bayes** | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| **Support Vector Machine** | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Highly effective on normalized cell data. Achieves the **highest overall AUC (0.9960)** and strong test accuracy (96.49%), proving strong linear separability among nuclear size, perimeter, and concavity features. Low false positive rate makes it reliable for clinical screening. |
| **Decision Tree** | Delivers fast and interpretable rule-based partitions, but achieves lower recall (83.33%) and accuracy (92.11%). Axis-aligned orthogonal splits struggle to capture oblique correlations present among physical dimensions (radius, area, perimeter), leading to slight overfitting compared to ensemble or margin-based methods. |
| **KNN** | Leverages distance-weighted voting in standardized feature space to capture non-linear local neighborhood similarities. Yields a competitive 95.61% accuracy and 0.9825 AUC, demonstrating that cell nuclei with similar morphological clusters share diagnostic outcomes. |
| **Naive Bayes** | GaussianNB achieves a solid 0.9891 AUC despite the strong conditional independence assumption being violated by highly correlated nuclear metrics (such as radius and perimeter). However, probability calibration slightly dampens raw accuracy (92.11%) and recall (85.71%). |
| **Support Vector Machine** | **Best overall performing model.** By projecting scaled features into an infinite-dimensional Hilbert space via the RBF kernel, SVM achieves the highest test Accuracy (**97.37%**), perfect Precision (**1.0000** — 0 false positives), highest F1 Score (**0.9630**), and highest Matthews Correlation Coefficient (**0.9442**). |

---

### Overall Winner for Your Dataset

> 🏆 **Overall Winner:** **Support Vector Machine (SVM)**

**Justification:**
1. **Superior Overall Accuracy & MCC:** SVM achieved the highest classification accuracy (**97.37%**) and the highest Matthews Correlation Coefficient (**0.9442**), proving balanced predictive strength across both positive (Malignant) and negative (Benign) classes.
2. **Zero False Positives (Precision = 1.0000):** In clinical oncology, false positive diagnoses lead to severe emotional trauma, unnecessary invasive biopsies, and patient distress. SVM classified all non-cancerous benign samples correctly without a single false alarm.
3. **High Recall (0.9286):** Minimizes false negatives (missed cancer diagnoses), correctly detecting 39 out of 42 malignant cases in the test set.
4. **Discriminative Margin:** Combined with Logistic Regression's 0.9960 AUC, SVM provides an optimal maximum-margin boundary that generalizes robustly to unseen patient data.

---

## e. Streamlit app Link

- **Streamlit app link:** `https://2026aj05631-bits-ml-assignment-2.streamlit.app/`  

---