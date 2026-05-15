#  Iris Species Classification

**Universidad de la Costa · Data Mining · Final Project**

---

##  Team Members
<!-- Add your names here -->
- Student 1 Name
- Student 2 Name
- Student 3 Name
- Student 4 Name

---

##  Project Overview

An end-to-end Data Mining project that classifies Iris flowers (**setosa**, **versicolor**, **virginica**) using a **Random Forest** classifier trained on the classic [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) (150 samples, 4 features).

The results are communicated through a fully interactive **Streamlit dashboard** with real-time predictions and 3D visualizations.

---

## 🔬 Methodology

| Step | Description |
|------|-------------|
| **1. Data Understanding** | Load Iris dataset, explore distributions, check class balance (50 samples/class). |
| **2. Preprocessing** | `StandardScaler` normalization; 75/25 stratified train-test split (`random_state=42`). |
| **3. Modeling** | `RandomForestClassifier` (200 estimators) — robust to scale, provides feature importance, handles multi-class natively. |
| **4. Evaluation** | Accuracy, Precision, Recall, F1 (weighted) on test set + 5-fold cross-validation. Per-class confusion matrix. |
| **5. Deployment** | Interactive Streamlit dashboard with four pages: Overview, Data Explorer, Model Metrics, and live Prediction. |

### Why Random Forest?
- Handles small datasets without overfitting (via bagging + random feature subsets).
- No hyperparameter tuning required to reach high accuracy on Iris.
- Built-in Gini-based feature importance gives explainability.
- Robust against correlated features (sepal/petal measurements are correlated).

---

##  Dashboard Pages

| Page | Content |
|------|---------|
|  **Overview** | Global metrics, scatter matrix, class balance pie, feature importance |
|  **Data Explorer** | Per-feature histograms, violin plots, correlation heatmap, raw data table |
|  **Model Metrics** | Test-set scores, 5-fold CV bar chart, confusion matrix, per-class report |
|  **Predict Species** | Slider inputs → real-time prediction + confidence bars + interactive 3D scatter |

---

##  Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/iris-classification.git
cd iris-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run Proyect.py
```

The app opens automatically at `http://localhost:8501`.

---

##  Deploy to Streamlit Cloud (free)

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and file `Proyect.py`.
4. Click **Deploy** — done!

---

##  Repository Structure

```
iris-classification/
├── Proyect.py          # Streamlit dashboard (main script)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

##  Results Summary

| Metric | Score |
|--------|-------|
| Accuracy | ~97% |
| Precision (weighted) | ~97% |
| Recall (weighted) | ~97% |
| F1 Score (weighted) | ~97% |
| CV Mean Accuracy | ~96% |


---

##  License
Educational use only — Universidad de la Costa, Data Mining course.
