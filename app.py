import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------- CONFIG ----------------
st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "dataset" / "creditcard.csv"
MODEL_PATH = BASE / "artifacts" / "fraud_model.joblib"
METRICS_PATH = BASE / "artifacts" / "metrics.json"

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#080d19,#101a2d 55%,#0b1220);color:#e8eefc}
.block-container {max-width:1400px;padding-top:2rem;padding-bottom:3rem}
section[data-testid="stSidebar"] {background:#0b1322;border-right:1px solid #24324a}
[data-testid="stMetric"] {background:linear-gradient(145deg,#17243a,#111c2e);border:1px solid #2a3b56;border-radius:14px;padding:16px}
[data-testid="stMetricLabel"] {color:#a9bad5}
[data-testid="stMetricValue"],h1,h2,h3 {color:#f4f7ff}
div.stButton>button {width:100%;min-height:44px;border-radius:9px;background:#2563eb;color:white;border:1px solid #3b82f6;font-weight:650}
[data-testid="stDataFrame"] {border:1px solid #2a3b56;border-radius:10px}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ FraudShield AI")
st.caption("INTELLIGENT TRANSACTION MONITORING • MACHINE LEARNING")

# ---------------- LOAD DATA ----------------
for path in (DATA_PATH, MODEL_PATH, METRICS_PATH):
    if not path.exists():
        st.error(f"Required file nahi mili: {path.relative_to(BASE)}")
        st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def load_metrics(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    df, metrics, model = load_data(str(DATA_PATH)), load_metrics(str(METRICS_PATH)), load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"File loading error: {error}")
    st.stop()

features = metrics.get("feature_columns", ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
if not hasattr(model, "predict") or not all(c in df.columns for c in features + ["Class"]):
    st.error("Saved model ya dataset mein required features valid nahi hain.")
    st.stop()

def show_plot(fig, key, height=350):
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=20, r=20, t=45, b=20))
    st.plotly_chart(fig, use_container_width=True, key=key)

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio("NAVIGATION", ["Overview", "Transaction Detection", "Model Performance", "About Project"])
st.sidebar.divider()
st.sidebar.caption("FraudShield AI | College Project")

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.subheader("📊 Dataset Overview")
    total = len(df)
    fraud = int((df["Class"] == 1).sum())
    legit = int((df["Class"] == 0).sum())
    rate = fraud / total * 100 if total else 0

    cols = st.columns(4)
    for col, label, value in zip(cols, ["Total Transactions", "Fraud Cases", "Legitimate Cases", "Fraud Rate"],
                                 [f"{total:,}", f"{fraud:,}", f"{legit:,}", f"{rate:.3f}%"]):
        col.metric(label, value)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Transaction Class Distribution")
        fig = px.bar(x=["Legitimate", "Fraud"], y=[legit, fraud], color=["Legitimate", "Fraud"],
                     color_discrete_map={"Legitimate": "#3B82F6", "Fraud": "#EF4444"},
                     labels={"x": "Transaction Type", "y": "Number of Transactions"})
        show_plot(fig, "transaction_class_distribution")
    with right:
        st.markdown("#### Transaction Amount Distribution")
        amount = df["Amount"].clip(upper=df["Amount"].quantile(.99))
        fig = px.histogram(x=amount, nbins=35, labels={"x": "Amount (upper 1% capped)", "y": "Transactions"})
        show_plot(fig, "amount_distribution")

    st.info("Dataset statistics actual Class labels se calculate kiye gaye hain, model predictions se nahi.")
    st.divider()
    st.subheader("📈 Advanced Fraud Analytics")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Average Transaction Amount")
        avg = df.groupby("Class")["Amount"].mean().reindex([0, 1], fill_value=0)
        fig = px.bar(x=["Legitimate", "Fraud"], y=avg.values, color=["Legitimate", "Fraud"],
                     color_discrete_map={"Legitimate": "#3B82F6", "Fraud": "#EF4444"},
                     labels={"x": "Transaction Type", "y": "Average Amount"})
        show_plot(fig, "average_amount_by_class")
    with right:
        st.markdown("#### Fraud Transactions Over Time")
        fraud_df = df[df["Class"] == 1].copy()
        if not fraud_df.empty:
            fraud_df["Dataset hour"] = (fraud_df["Time"] // 3600).astype(int)
            hourly = fraud_df.groupby("Dataset hour").size().reindex(range(48), fill_value=0)
            fig = px.line(x=hourly.index, y=hourly.values, markers=True,
                          labels={"x": "Dataset time bucket (hour)", "y": "Fraud transactions"})
            show_plot(fig, "fraud_over_time")
        else:
            st.info("No fraud-labelled transactions available.")

    st.divider()
    st.subheader("🔎 Transaction Data Explorer")
    class_filter = st.selectbox("Filter transactions", ["All", "Legitimate only", "Fraud only"])
    amount_min, amount_max = float(df["Amount"].min()), float(df["Amount"].max())
    amount_range = st.slider("Transaction amount range", min_value=amount_min, max_value=amount_max,
                             value=(amount_min, amount_max))
    filtered = df[df["Amount"].between(*amount_range)]
    if class_filter == "Legitimate only":
        filtered = filtered[filtered["Class"] == 0]
    elif class_filter == "Fraud only":
        filtered = filtered[filtered["Class"] == 1]
    st.caption(f"{len(filtered):,} transactions match your filters.")
    st.dataframe(filtered[["Time", "Amount", "Class"]].head(500), use_container_width=True, hide_index=True)

# ---------------- TRANSACTION DETECTION ----------------
elif page == "Transaction Detection":
    st.subheader("🔎 Transaction Detection")
    st.write("Historical dataset se example select karo. Prediction input mein actual Class label use nahi hota.")
    scenario = st.radio("Example type", ["Legitimate transaction", "Fraud transaction"], horizontal=True)
    class_value = 0 if scenario == "Legitimate transaction" else 1
    indices = df.index[df["Class"] == class_value].tolist()
    if not indices:
        st.warning("Is category mein transaction nahi mili.")
        st.stop()
    selected = st.selectbox("Select transaction example", indices[:min(100, len(indices))],
                            format_func=lambda idx: f"Transaction ID {idx}")
    row = df.loc[selected]
    c1, c2 = st.columns(2)
    c1.metric("Transaction Amount", f"{row['Amount']:.2f}")
    c2.metric("Transaction Time", f"{row['Time']:.2f} sec")
    with st.expander("View anonymized transaction features"):
        st.dataframe(pd.DataFrame([row[features].to_dict()]), use_container_width=True)

    if st.button("🔍 Predict Transaction", type="primary"):
        try:
            input_df = pd.DataFrame([row[features].to_dict()], columns=features)
            prediction = int(model.predict(input_df)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                if 1 in classes:
                    probability = float(probabilities[classes.index(1)])
            st.divider()
            st.markdown("### Prediction Result")
            if prediction == 1:
                st.error("⚠️ Model Prediction: Potential Fraud")
            else:
                st.success("✅ Model Prediction: Legitimate")
            if probability is not None:
                st.metric("Estimated Fraud Probability", f"{probability:.2%}")
                st.progress(max(0.0, min(1.0, probability)))
            actual = int(row["Class"])
            if prediction == actual:
                st.success("Prediction matches the dataset label.")
            else:
                st.warning("Prediction differs from the dataset label.")
            st.caption("Historical dataset demonstration only; not a live bank integration or a guarantee of safety.")
        except Exception as error:
            st.error(f"Prediction error: {error}")

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.subheader("📈 Model Performance")
    st.write("Metrics calculated on the held-out test set during training.")
    cols = st.columns(3)
    for col, label, key, fmt in [
        (cols[0], "Fraud Precision", "fraud_precision", ".1%"),
        (cols[1], "Fraud Recall", "fraud_recall", ".1%"),
        (cols[2], "Fraud F1-score", "fraud_f1_score", ".3f"),
    ]:
        col.metric(label, format(metrics[key], fmt))
    cols = st.columns(3)
    cols[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    cols[1].metric("Average Precision", f"{metrics['average_precision']:.3f}")
    cols[2].metric("Test Transactions", f"{metrics['test_rows']:,}")

    st.divider()
    st.markdown("#### Confusion Matrix")
    cm = metrics.get("confusion_matrix")
    if cm:
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted Legitimate", "Predicted Fraud"],
                                   y=["Actual Legitimate", "Actual Fraud"], text=cm, texttemplate="%{text}",
                                   colorscale="Blues", hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"))
        show_plot(fig, "model_confusion_matrix", 400)

    report = metrics.get("classification_report", {})
    if report:
        st.markdown("#### Classification Report")
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.warning("Fraud recall batata hai ki actual fraud cases mein se kitne detect hue. Imbalanced data par high accuracy alone sufficient nahi hoti.")
    roc_data, pr_data = metrics.get("roc_curve", {}), metrics.get("precision_recall_curve", {})
    left, right = st.columns(2)
    with left:
        st.markdown("#### ROC Curve")
        if all(k in roc_data for k in ("fpr", "tpr")):
            fig = px.line(x=roc_data["fpr"], y=roc_data["tpr"], labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                          title=f"ROC-AUC = {metrics.get('roc_auc', 0):.3f}")
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(dash="dash"))
            show_plot(fig, "roc_curve")
        else:
            st.info("ROC curve coordinates saved nahi hain. Training script update karni hogi.")
    with right:
        st.markdown("#### Precision-Recall Curve")
        if all(k in pr_data for k in ("precision", "recall")):
            fig = px.line(x=pr_data["recall"], y=pr_data["precision"],
                          labels={"x": "Recall", "y": "Precision"},
                          title=f"Average Precision = {metrics.get('average_precision', 0):.3f}")
            show_plot(fig, "precision_recall_curve")
        else:
            st.info("Precision-recall curve coordinates saved nahi hain. Training script update karni hogi.")

# ---------------- ABOUT ----------------
else:
    st.subheader("🎓 About This Project")
    st.markdown("""
**Project:** Online Credit Card Fraud Detection using Machine Learning

**Algorithm:** Random Forest Classifier

**Dataset:** Historical credit card transaction dataset

**Workflow:**
1. Load and inspect transaction data.
2. Separate features and target label.
3. Split training/test data using stratification.
4. Train a Random Forest model.
5. Evaluate precision, recall, F1-score, Average Precision, ROC-AUC and confusion matrix.
6. Use the saved model to predict selected transactions.

**Limitations:** Educational prototype only. It does not connect to a bank or process live transactions.
""")
    st.markdown("#### Model Details")
    st.write("Algorithm:", metrics.get("model_name", "Random Forest"))
    st.write("Training rows:", f"{metrics['training_rows']:,}")
    st.write("Test rows:", f"{metrics['test_rows']:,}")
    st.write("Features used:", len(features))

st.divider()
st.caption("FraudShield AI • Educational Machine Learning Project")
