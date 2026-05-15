"""
Iris Species Classification — Final Project
Universidad de la Costa | Data Mining
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .page-title {
        font-size: 2rem; font-weight: 700;
        color: #1a1a2e; margin-bottom: 0.2rem; letter-spacing: -0.5px;
    }
    .page-subtitle { font-size: 0.9rem; color: #777; margin-bottom: 1.8rem; }

    .section-title {
        font-size: 1rem; font-weight: 600; color: #1a1a2e;
        border-bottom: 2px solid #e8e8f0;
        padding-bottom: 0.4rem; margin: 1.6rem 0 0.9rem;
    }

    .metric-card {
        background: #f7f7fb; border: 1px solid #e0e0ec;
        border-radius: 10px; padding: 1.1rem 1.4rem; text-align: center;
    }
    .metric-value { font-size: 1.9rem; font-weight: 700; color: #2d2d6b; letter-spacing: -1px; }
    .metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }

    .pred-box {
        background: #f2f2fb; border: 1.5px solid #c5c5e8;
        border-radius: 12px; padding: 1.4rem; text-align: center; margin-top: 0.8rem;
    }
    .pred-label { font-size: 0.75rem; color: #999; text-transform: uppercase; letter-spacing: 1px; }
    .pred-species { font-size: 1.7rem; font-weight: 700; color: #2d2d6b; margin-top: 4px; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ── Data & model (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = [
        "Sepal Length (cm)", "Sepal Width (cm)",
        "Petal Length (cm)", "Petal Width (cm)", "target",
    ]
    df["Species"] = df["target"].map(dict(enumerate(iris.target_names)))

    X = df[["Sepal Length (cm)", "Sepal Width (cm)",
            "Petal Length (cm)", "Petal Width (cm)"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall":    recall_score(y_test, y_pred, average="weighted"),
        "F1 Score":  f1_score(y_test, y_pred, average="weighted"),
    }
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="accuracy")
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(
        y_test, y_pred, target_names=iris.target_names, output_dict=True,
    )
    feature_imp = pd.Series(
        model.feature_importances_,
        index=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
    ).sort_values(ascending=True)

    return df, X, y, model, scaler, metrics, cv_scores, cm, report, feature_imp, iris.target_names


df, X, y, model, scaler, metrics, cv_scores, cm, report, feature_imp, target_names = load_and_train()

FEATURES = list(X.columns)
COLORS = {"setosa": "#5b8dee", "versicolor": "#3ecf8e", "virginica": "#c084fc"}
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#fafafa",
    font=dict(family="Inter", color="#333", size=12),
    margin=dict(t=24, b=24, l=10, r=10),
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Iris Classification")
    st.caption("Universidad de la Costa — Data Mining")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Data Explorer", "Model Metrics", "Predict"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Random Forest · scikit-learn · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<p class="page-title">Iris Species Classification</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Universidad de la Costa &nbsp;·&nbsp; Data Mining Final Project &nbsp;·&nbsp; Random Forest Classifier</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, (name, val) in zip(cols, metrics.items()):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val:.1%}</div>'
            f'<div class="metric-label">{name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-title">Scatter Matrix</p>', unsafe_allow_html=True)
    fig_sm = px.scatter_matrix(
        df, dimensions=FEATURES, color="Species",
        color_discrete_map=COLORS, opacity=0.68, height=460,
    )
    fig_sm.update_traces(marker=dict(size=4, line=dict(width=0)))
    fig_sm.update_layout(**LAYOUT)
    st.plotly_chart(fig_sm, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<p class="section-title">Class Balance</p>', unsafe_allow_html=True)
        counts = df["Species"].value_counts().reset_index()
        counts.columns = ["Species", "Count"]
        fig_pie = px.pie(
            counts, names="Species", values="Count",
            color="Species", color_discrete_map=COLORS,
            hole=0.45, height=280,
        )
        fig_pie.update_layout(**LAYOUT, legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown('<p class="section-title">Feature Importance</p>', unsafe_allow_html=True)
        fi_df = feature_imp.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            text_auto=".3f", height=280,
        )
        fig_fi.update_layout(**LAYOUT, coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<p class="section-title">Workflow</p>', unsafe_allow_html=True)
    st.markdown("""
| Step | Description |
|------|-------------|
| Data Understanding | Load Iris (150 samples, 4 features, 3 balanced classes). Explore distributions. |
| Preprocessing | StandardScaler normalization; 75/25 stratified train-test split. |
| Modeling | Random Forest (200 trees) — handles correlated features, built-in feature importance, no tuning needed on this dataset. |
| Evaluation | Accuracy, Precision, Recall, F1 on test set + 5-fold cross-validation. Confusion matrix per class. |
| Deployment | Interactive Streamlit dashboard with real-time prediction and 3D visualization. |
""")


# ══════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown('<p class="page-title">Data Explorer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">150 samples · 4 numerical features · 3 balanced classes (50 samples each)</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-title">Feature Distributions by Species</p>', unsafe_allow_html=True)
    sel_feat = st.selectbox("Feature", FEATURES, index=2)
    fig_hist = px.histogram(
        df, x=sel_feat, color="Species", color_discrete_map=COLORS,
        barmode="overlay", opacity=0.72, nbins=25,
        marginal="box", height=380,
    )
    fig_hist.update_layout(**LAYOUT)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<p class="section-title">Violin Plots — All Features</p>', unsafe_allow_html=True)
    fig_vio = go.Figure()
    for sp, col in COLORS.items():
        sub = df[df["Species"] == sp]
        for feat in FEATURES:
            fig_vio.add_trace(go.Violin(
                x=[feat] * len(sub), y=sub[feat],
                name=sp, legendgroup=sp,
                showlegend=(feat == FEATURES[0]),
                line_color=col, fillcolor=col,
                opacity=0.55, box_visible=True, meanline_visible=True,
            ))
    fig_vio.update_layout(
        violinmode="group", height=400,
        xaxis_title="Feature", yaxis_title="Value (cm)",
        legend=dict(orientation="h", y=-0.15),
        **LAYOUT,
    )
    st.plotly_chart(fig_vio, use_container_width=True)

    st.markdown('<p class="section-title">Correlation Heatmap</p>', unsafe_allow_html=True)
    fig_corr = px.imshow(
        df[FEATURES].corr(), text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=360,
    )
    fig_corr.update_layout(**LAYOUT)
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("View raw dataset"):
        st.dataframe(df.drop(columns=["target"]), use_container_width=True, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Metrics":
    st.markdown('<p class="page-title">Model Metrics</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Results on the held-out test set (25%) and 5-fold cross-validation.</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, f"{val:.4f}")

    st.markdown('<p class="section-title">5-Fold Cross-Validation Accuracy</p>', unsafe_allow_html=True)
    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)],
        "Accuracy": cv_scores,
    })
    fig_cv = px.bar(
        cv_df, x="Fold", y="Accuracy",
        color="Accuracy", color_continuous_scale="Blues",
        range_y=[0.85, 1.01], height=300, text_auto=".3f",
    )
    fig_cv.add_hline(
        y=cv_scores.mean(), line_dash="dash", line_color="#e55",
        annotation_text=f"Mean: {cv_scores.mean():.3f}",
    )
    fig_cv.update_layout(**LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig_cv, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=list(target_names), y=list(target_names),
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            height=360,
        )
        fig_cm.update_layout(**LAYOUT)
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        st.markdown('<p class="section-title">Per-Class Report</p>', unsafe_allow_html=True)
        rows = [{
            "Species":   sp,
            "Precision": round(report[sp]["precision"], 4),
            "Recall":    round(report[sp]["recall"],    4),
            "F1 Score":  round(report[sp]["f1-score"],  4),
        } for sp in target_names]
        rep_df = pd.DataFrame(rows)
        fig_rep = px.bar(
            rep_df.melt(id_vars="Species", value_vars=["Precision", "Recall", "F1 Score"]),
            x="Species", y="value", color="variable",
            barmode="group", range_y=[0, 1.05],
            color_discrete_sequence=["#5b8dee", "#3ecf8e", "#c084fc"],
            height=360, text_auto=".3f",
        )
        fig_rep.update_layout(**LAYOUT, legend_title="Metric", yaxis_title="Score")
        st.plotly_chart(fig_rep, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown('<p class="page-title">Predict Species</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Adjust the four measurements and the model classifies the flower in real time.</p>',
        unsafe_allow_html=True,
    )

    c_in, c_out = st.columns([1, 2])

    with c_in:
        st.markdown('<p class="section-title">Measurements</p>', unsafe_allow_html=True)
        sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sw = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
        pl = st.slider("Petal Length (cm)", 1.0, 7.0, 3.7, 0.1)
        pw = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

        input_arr    = np.array([[sl, sw, pl, pw]])
        input_scaled = scaler.transform(input_arr)
        pred_idx     = model.predict(input_scaled)[0]
        proba        = model.predict_proba(input_scaled)[0]
        species      = target_names[pred_idx]

        st.markdown(
            f'<div class="pred-box">'
            f'<div class="pred-label">Predicted species</div>'
            f'<div class="pred-species">Iris {species}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-title">Confidence</p>', unsafe_allow_html=True)
        for sp, p in zip(target_names, proba):
            st.markdown(f"<small><b>Iris {sp}</b> &nbsp; {p:.1%}</small>", unsafe_allow_html=True)
            st.progress(float(p))

    with c_out:
        st.markdown('<p class="section-title">3D Scatter — Your Sample in Context</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        ax1 = col1.selectbox("X axis", FEATURES, index=2, key="ax1")
        ax2 = col2.selectbox("Y axis", FEATURES, index=3, key="ax2")
        ax3 = col3.selectbox("Z axis", FEATURES, index=0, key="ax3")

        feat_vals = {
            "Sepal Length (cm)": sl, "Sepal Width (cm)": sw,
            "Petal Length (cm)": pl, "Petal Width (cm)": pw,
        }

        fig_3d = px.scatter_3d(
            df, x=ax1, y=ax2, z=ax3,
            color="Species", color_discrete_map=COLORS,
            opacity=0.55, height=520, symbol="Species",
        )
        fig_3d.add_trace(go.Scatter3d(
            x=[feat_vals[ax1]], y=[feat_vals[ax2]], z=[feat_vals[ax3]],
            mode="markers",
            marker=dict(size=12, color="#e55", symbol="diamond",
                        line=dict(color="white", width=2)),
            name="Your sample",
        ))
        fig_3d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis_title=ax1, yaxis_title=ax2, zaxis_title=ax3,
                bgcolor="#fafafa",
            ),
            font=dict(family="Inter", color="#333", size=12),
            legend=dict(orientation="h", y=-0.06),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_3d, use_container_width=True)
