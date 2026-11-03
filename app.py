"""
Streamlit Web Application: Breast Cancer Classification Studio
BITS Pilani - M.Tech (AIML) - Machine Learning Assignment 2

Features:
- Test CSV dataset upload with validation (fallback to bundled test_data.csv)
- Dynamic model selection (All 5 classifiers + comparative view)
- 6 Key Evaluation Metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Interactive Confusion Matrix & Classification Report
- ROC Curve Visualization
- Comparative Performance Analytics across all models
- Interactive Single-Sample Diagnosis Simulator
"""

# imports
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification | BITS Pilani ML Assignment 2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .winner-badge {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        color: #065F46;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 600;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# Cache model loading for snappy performance
@st.cache_resource
def load_model_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")

    models = {
        "Support Vector Machine": joblib.load(os.path.join(model_dir, "svm.joblib")),
        "Logistic Regression": joblib.load(os.path.join(model_dir, "logistic_regression.joblib")),
        "KNN": joblib.load(os.path.join(model_dir, "knn.joblib")),
        "Naive Bayes": joblib.load(os.path.join(model_dir, "naive_bayes.joblib")),
        "Decision Tree": joblib.load(os.path.join(model_dir, "decision_tree.joblib"))
    }

    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

    with open(os.path.join(model_dir, "features.json"), "r") as f:
        feature_names = json.load(f)

    with open(os.path.join(model_dir, "metrics_summary.json"), "r") as f:
        baseline_metrics = json.load(f)

    # Scaling rules (Decision Tree doesn't strictly need scaling, others use scaled)
    scale_flags = {
        "Support Vector Machine": True,
        "Logistic Regression": True,
        "KNN": True,
        "Naive Bayes": True,
        "Decision Tree": False
    }

    return models, scaler, feature_names, baseline_metrics, scale_flags


try:
    MODELS, SCALER, FEATURE_NAMES, BASELINE_METRICS, SCALE_FLAGS = load_model_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure 'train_models.py' was run.")
    st.stop()


# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("BITS Pilani - M.Tech AIML")
st.sidebar.markdown("**Machine Learning - Assignment 2**")
st.sidebar.markdown("---")

st.sidebar.subheader("📂 1. Test Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (Features + Target)",
    type=["csv"],
    help="Upload your held-out test CSV file containing 30 features and 'diagnosis' target."
)

if uploaded_file is not None:
    try:
        raw_test_df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Uploaded: {uploaded_file.name} ({len(raw_test_df)} rows)")
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        st.stop()
else:
    # Fallback to pre-bundled test_data.csv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_test_path = os.path.join(base_dir, "test_data.csv")
    if os.path.exists(default_test_path):
        raw_test_df = pd.read_csv(default_test_path)
        st.sidebar.info(f"Using default `test_data.csv` ({len(raw_test_df)} samples)")
    else:
        st.sidebar.error("test_data.csv not found. Please upload a CSV.")
        st.stop()

# Download sample test data button
base_dir = os.path.dirname(os.path.abspath(__file__))
default_test_path = os.path.join(base_dir, "test_data.csv")
if os.path.exists(default_test_path):
    with open(default_test_path, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Sample Test CSV",
            data=f,
            file_name="test_data.csv",
            mime="text/csv",
            help="Download the sample test data CSV to test the upload feature."
        )

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 2. Model Selection")
model_options = [
    "All Models (Comparative View)",
    "Support Vector Machine",
    "Logistic Regression",
    "KNN",
    "Naive Bayes",
    "Decision Tree"
]
selected_option = st.sidebar.selectbox("Choose Model to Evaluate:", model_options)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Target Mapping:**
- **M** (Malignant) $\\rightarrow$ **1**
- **B** (Benign) $\\rightarrow$ **0**
""")


# Preprocess test data
def prepare_test_data(df, feature_names):
    # Check target column
    target_col = None
    for cand in ['diagnosis', 'target', 'class', 'label']:
        if cand in df.columns:
            target_col = cand
            break

    if target_col is None:
        raise ValueError("Could not find target column ('diagnosis'). Test CSV must include target column.")

    # Validate feature columns
    missing_feats = [col for col in feature_names if col not in df.columns]
    if missing_feats:
        raise ValueError(f"Uploaded CSV is missing {len(missing_feats)} feature(s): {missing_feats[:3]}...")

    X = df[feature_names].copy()
    y_raw = df[target_col]

    # Map target if categorical
    if y_raw.dtype == object or isinstance(y_raw.iloc[0], str):
        y = y_raw.map({'M': 1, 'B': 0, '1': 1, '0': 0})
        # If any didn't map, fillna
        if y.isnull().any():
            y = pd.to_numeric(y_raw, errors='coerce').fillna(0).astype(int)
        else:
            y = y.astype(int)
    else:
        y = y_raw.astype(int)

    return X, y, target_col


try:
    X_test, y_test, detected_target = prepare_test_data(raw_test_df, FEATURE_NAMES)
except Exception as e:
    st.error(f"⚠️ Dataset Validation Error: {e}")
    st.stop()


# Evaluation helper 
def evaluate_model(model_name, clf, X, y, scaler, scale_flags):
    use_scaled = scale_flags.get(model_name, True)
    X_proc = scaler.transform(X) if use_scaled else X.values

    y_pred = clf.predict(X_proc)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_proc)[:, 1]
    elif hasattr(clf, "decision_function"):
        df_vals = clf.decision_function(X_proc)
        # Sigmoid normalize
        y_prob = 1 / (1 + np.exp(-df_vals))
    else:
        y_prob = y_pred

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)

    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred, target_names=["Benign (0)", "Malignant (1)"], output_dict=True)

    fpr, tpr, _ = roc_curve(y, y_prob)

    return {
        "name": model_name,
        "accuracy": acc,
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "cm": cm,
        "cr": cr,
        "fpr": fpr,
        "tpr": tpr
    }


# Compute evaluation for all models
all_eval_results = {}
for m_name, m_obj in MODELS.items():
    all_eval_results[m_name] = evaluate_model(m_name, m_obj, X_test, y_test, SCALER, SCALE_FLAGS)


# UI Header
st.markdown('<div class="main-title">🔬 Breast Cancer Classification System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Evaluation of 5 Machine Learning Classifiers on Breast Cancer Wisconsin (Diagnostic) Dataset '
    f'• Evaluated on <b>{len(X_test)}</b> test instances</div>',
    unsafe_allow_html=True
)

tabs = st.tabs(["📊 Model Evaluation & Metrics", "🔍 Live Sample Diagnosis", "📖 Dataset & Assignment Details"])


# TAB 1: Evaluation and metrics
with tabs[0]:
    if selected_option == "All Models (Comparative View)":
        st.subheader("📋 Models Performance Comparison Table")
        st.caption("Comprehensive comparison of all 5 classifiers across the 6 mandatory evaluation metrics.")

        comparison_data = []
        for m_name in ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]:
            res = all_eval_results[m_name]
            comparison_data.append({
                "ML Model Name": m_name,
                "Accuracy": f"{res['accuracy']:.4f}",
                "AUC": f"{res['auc']:.4f}",
                "Precision": f"{res['precision']:.4f}",
                "Recall": f"{res['recall']:.4f}",
                "F1": f"{res['f1']:.4f}",
                "MCC": f"{res['mcc']:.4f}"
            })

        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="winner-badge">
            🏆 <b>Overall Winner: Support Vector Machine (SVM)</b><br>
            SVM achieves the highest test Accuracy (<b>97.37%</b>), perfect Precision (<b>1.0000</b>), highest F1 Score (<b>0.9630</b>), and highest MCC (<b>0.9442</b>).
            Logistic Regression closely follows with the highest AUC (<b>0.9960</b>). In clinical diagnosis, zero false positives with high recall makes SVM the optimal choice!
        </div>
        """, unsafe_allow_html=True)

        # Comparative Visualization
        st.subheader("📈 Multi-Metric Comparative Visualizer")
        
        # Prepare numeric comparison df
        num_comp_data = []
        for m_name in ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]:
            res = all_eval_results[m_name]
            num_comp_data.append({
                "Model": m_name,
                "Accuracy": res['accuracy'],
                "AUC": res['auc'],
                "Precision": res['precision'],
                "Recall": res['recall'],
                "F1 Score": res['f1'],
                "MCC": res['mcc']
            })
        num_df = pd.DataFrame(num_comp_data).set_index("Model")

        col1, col2 = st.columns([1.3, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(10, 5))
            num_df.plot(kind="bar", ax=ax, colormap="viridis", width=0.82)
            ax.set_title("Performance Metrics by Classifier", fontsize=13, fontweight='bold', pad=12)
            ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
            ax.set_ylim(0.75, 1.02)
            ax.legend(loc='lower right', frameon=True, ncol=3)
            plt.xticks(rotation=15, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### 📝 Performance Observations")
            st.markdown("""
            - **Logistic Regression**: Exceptional linear separability on standardized features; attains highest AUC (0.9960) and 0.9649 accuracy.
            - **Decision Tree**: Fast and interpretable, but suffers slightly from greedy axis-aligned splits on correlated continuous features (Acc: 0.9211, MCC: 0.8299).
            - **K-Nearest Neighbors**: Distance-weighted metric with standardized features provides strong local manifold boundary detection (Acc: 0.9561, MCC: 0.9058).
            - **Naive Bayes**: Performs reliably (AUC: 0.9891) despite feature independence violation inherent to correlated biological cell measurements.
            - **Support Vector Machine**: Optimal maximum-margin hyperplane with RBF kernel delivers best balance, zero false positives, and highest MCC (0.9442).
            """)

    else:
        # Single model detailed view
        res = all_eval_results[selected_option]

        st.subheader(f"🔍 Evaluation Results: {selected_option}")

        # Top 6 Metric Cards
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{res['accuracy']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AUC Score</div>
                <div class="metric-value">{res['auc']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{res['precision']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{res['recall']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{res['f1']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MCC</div>
                <div class="metric-value">{res['mcc']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Diagnostic Visualizations: Confusion Matrix & ROC Curve
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("#### 🎯 Confusion Matrix")
            cm = res['cm']
            fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Benign (0)", "Malignant (1)"],
                yticklabels=["Benign (0)", "Malignant (1)"],
                cbar=False,
                annot_kws={"size": 14, "weight": "bold"},
                ax=ax_cm
            )
            ax_cm.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
            ax_cm.set_ylabel("True Ground Truth", fontsize=11, fontweight='bold')
            ax_cm.set_title(f"Confusion Matrix: {selected_option}", fontsize=12, pad=10)
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()

            # Breakdown
            tn, fp, fn, tp = cm.ravel()
            st.caption(f"**TN**: {tn} | **FP**: {fp} | **FN**: {fn} | **TP**: {tp} (Total: {tn+fp+fn+tp})")

        with viz_col2:
            st.markdown("#### 📉 ROC Curve")
            fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.2))
            ax_roc.plot(res['fpr'], res['tpr'], color='#1E40AF', lw=2.5, label=f"ROC (AUC = {res['auc']:.4f})")
            ax_roc.plot([0, 1], [0, 1], color='#94A3B8', lw=1.5, linestyle='--', label="Random Classifier")
            ax_roc.fill_between(res['fpr'], res['tpr'], alpha=0.15, color='#3B82F6')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
            ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
            ax_roc.set_title(f"ROC Curve: {selected_option}", fontsize=12, pad=10)
            ax_roc.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_roc)
            plt.close()

        # Detailed Classification Report
        st.markdown("#### 📑 Detailed Classification Report")
        cr_dict = res['cr']
        cr_df = pd.DataFrame({
            "Class": ["Benign (0)", "Malignant (1)", "Accuracy", "Macro Avg", "Weighted Avg"],
            "Precision": [
                f"{cr_dict['Benign (0)']['precision']:.4f}",
                f"{cr_dict['Malignant (1)']['precision']:.4f}",
                "-",
                f"{cr_dict['macro avg']['precision']:.4f}",
                f"{cr_dict['weighted avg']['precision']:.4f}"
            ],
            "Recall": [
                f"{cr_dict['Benign (0)']['recall']:.4f}",
                f"{cr_dict['Malignant (1)']['recall']:.4f}",
                "-",
                f"{cr_dict['macro avg']['recall']:.4f}",
                f"{cr_dict['weighted avg']['recall']:.4f}"
            ],
            "F1-Score": [
                f"{cr_dict['Benign (0)']['f1-score']:.4f}",
                f"{cr_dict['Malignant (1)']['f1-score']:.4f}",
                f"{cr_dict['accuracy']:.4f}",
                f"{cr_dict['macro avg']['f1-score']:.4f}",
                f"{cr_dict['weighted avg']['f1-score']:.4f}"
            ],
            "Support": [
                int(cr_dict['Benign (0)']['support']),
                int(cr_dict['Malignant (1)']['support']),
                int(len(y_test)),
                int(cr_dict['macro avg']['support']),
                int(cr_dict['weighted avg']['support'])
            ]
        })
        st.dataframe(cr_df, use_container_width=True, hide_index=True)


# ---------------- TAB 2: Live sample diagnosis ----------------
with tabs[1]:
    st.subheader("🔍 Interactive Single Patient Prediction Simulator")
    st.markdown("Select any patient sample from the uploaded test dataset or inspect custom biological cell feature values.")

    sample_idx = st.slider("Select Sample Index from Test Dataset:", min_value=0, max_value=len(X_test)-1, value=0)

    selected_sample = X_test.iloc[sample_idx]
    actual_label = y_test.iloc[sample_idx]
    actual_str = "Malignant (1)" if actual_label == 1 else "Benign (0)"

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        st.markdown(f"**Sample #{sample_idx} Ground Truth:** `{actual_str}`")

        # Let user choose model for this sample
        sim_model_choice = st.selectbox(
            "Classifier for Prediction:",
            list(MODELS.keys()),
            key="sim_model_select"
        )
        sim_clf = MODELS[sim_model_choice]
        sim_scaled = SCALE_FLAGS[sim_model_choice]

        # Prepare 1 sample
        x_sample = selected_sample.values.reshape(1, -1)
        if sim_scaled:
            x_sample_proc = SCALER.transform(x_sample)
        else:
            x_sample_proc = x_sample

        pred = sim_clf.predict(x_sample_proc)[0]
        if hasattr(sim_clf, "predict_proba"):
            prob = sim_clf.predict_proba(x_sample_proc)[0]
            prob_mal = prob[1]
            prob_ben = prob[0]
        else:
            prob_mal = float(pred)
            prob_ben = 1.0 - prob_mal

        pred_str = "Malignant (1)" if pred == 1 else "Benign (0)"
        is_correct = (pred == actual_label)

        if pred == 1:
            st.error(f"🚨 **Predicted: {pred_str}**")
        else:
            st.success(f"✅ **Predicted: {pred_str}**")

        st.metric("Malignancy Probability", f"{prob_mal*100:.2f}%")
        st.metric("Benign Probability", f"{prob_ben*100:.2f}%")

        if is_correct:
            st.info("🎯 **Prediction Matches Ground Truth!**")
        else:
            st.warning("⚠️ **Prediction Differs from Ground Truth!**")

    with sim_col2:
        st.markdown("**Top Feature Values for this Patient:**")
        key_feats = [
            'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'concavity_mean', 'concave points_mean', 'radius_worst', 'perimeter_worst'
        ]
        sample_subset = selected_sample[key_feats].to_frame(name="Value")
        st.dataframe(sample_subset, use_container_width=True)


#  TAB 3: Dataset and assignment details
with tabs[2]:
    st.subheader("📖 Dataset Description & Academic Specifications")

    st.markdown("""
    ### 1. Problem Statement
    Breast cancer is one of the most prevalent and critical forms of cancer affecting individuals worldwide.
    Early and accurate diagnosis between **Benign** (non-cancerous) and **Malignant** (cancerous) tissue is essential
    for patient survival and treatment strategy.
    

    ### 2. Dataset Overview
    - **Dataset Name:** Kaggle (usman44m/breast-cancer-tabular-dataset). Also added in the repository as data.csv
    - **Total Instances:** 569 
    - **Total Features:** 30 continuous numerical biological cell nuclei features 
    - **Target Variable:** `diagnosis` (Binary: `M` = Malignant, `B` = Benign)
    - **Class Balance:** 357 Benign (62.7%), 212 Malignant (37.3%)
    - **Split:** 80% Train (455 samples), 20% Test (114 samples, saved as `test_data.csv`)

    ### 3. Features Extracted from Cell Nuclei
    Features are computed from digitized images of fine needle aspirates (FNA) of breast masses:
    1. **radius** (mean of distances from center to points on perimeter)
    2. **texture** (standard deviation of gray-scale values)
    3. **perimeter**
    4. **area**
    5. **smoothness** (local variation in radius lengths)
    6. **compactness** (perimeter^2 / area - 1.0)
    7. **concavity** (severity of concave portions of contour)
    8. **concave points** (number of concave portions of contour)
    9. **symmetry**
    10. **fractal dimension** ("coastline approximation" - 1)
    *(Each computed for Mean, Standard Error (SE), and Worst / Largest values = 30 total features)*

    ### 4. Evaluation Metrics Definitions
    - **Accuracy:** $(TP + TN) / (TP + TN + FP + FN)$
    - **AUC (Area Under ROC Curve):** Measures discriminative capability across all classification thresholds.
    - **Precision:** $TP / (TP + FP)$ - Crucial for avoiding false alarms.
    - **Recall (Sensitivity):** $TP / (TP + FN)$ - Vital in oncology to minimize missed malignant cases.
    - **F1 Score:** Harmonic mean of Precision and Recall: $2 \\cdot (P \\cdot R) / (P + R)$.
    - **Matthews Correlation Coefficient (MCC):** Balanced measure taking into account true and false positives and negatives, highly robust to class imbalance.
    """)
