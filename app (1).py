import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "creditcard.csv"
MODEL = BASE / "artifacts" / "fraud_model.joblib"
METRICS = BASE / "artifacts" / "metrics.json"

st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#070b14,#101b2c 60%,#080e19);color:#edf3ff}
.block-container{max-width:1450px;padding-top:1.5rem}
section[data-testid="stSidebar"]{background:#0b1422;border-right:1px solid #26364d}
[data-testid="stMetric"]{background:linear-gradient(145deg,#17263a,#101a2a);border:1px solid #2b405c;border-radius:14px;padding:15px}
[data-testid="stMetricLabel"]{color:#a8bbd7}
h1,h2,h3,[data-testid="stMetricValue"]{color:#f5f8ff}
div.stButton>button{width:100%;min-height:45px;border-radius:10px;background:#176b56;color:white;border:1px solid #2bba8a;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ FRAUDSHIELD")
st.caption("AI TRANSACTION INTELLIGENCE  /  CREDIT CARD FRAUD LAB")

for path in (DATA, MODEL, METRICS):
    if not path.exists():
        st.error(f"Required file missing: {path.relative_to(BASE)}")
        st.stop()

@st.cache_data
def read_csv(path):
    return pd.read_csv(path)

@st.cache_data
def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def read_model(path):
    return joblib.load(path)

try:
    df, metrics, model = read_csv(str(DATA)), read_json(str(METRICS)), read_model(str(MODEL))
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

features = metrics.get("feature_columns", ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"])
if not hasattr(model, "predict") or not all(c in df.columns for c in features + ["Class"]):
    st.error("Model or dataset does not contain the expected features.")
    st.stop()

def chart(fig, key, height=340):
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=15, r=15, t=45, b=15),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=key)

page = st.sidebar.radio("CONTROL ROOM", ["Live Scanner", "Overview", "Model Lab", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("LOCAL MODEL • HISTORICAL DATA DEMO")

total = len(df)
fraud = int(df["Class"].sum())
legit = total - fraud

if page == "Live Scanner":
    st.subheader("Transaction scanner")
    st.write("Pick a real anonymized transaction from the dataset, then run it through your saved model.")
    mode = st.radio("Choose a test scenario", ["Quick demo", "Choose from dataset"], horizontal=True)
    if mode == "Quick demo":
        a, b = st.columns(2)
        if a.button("🟢 Load legitimate example"):
            st.session_state.scan_id = int(df.index[df["Class"] == 0][0])
        if b.button("🔴 Load fraud example"):
            st.session_state.scan_id = int(df.index[df["Class"] == 1][0])
        idx = st.session_state.get("scan_id", int(df.index[0]))
        st.caption("Quick demo loads a labelled historical example; the model still makes its own prediction.")
    else:
        kind = st.selectbox("Example category", ["Any transaction", "Legitimate", "Fraud-labelled"])
        subset = df if kind == "Any transaction" else df[df["Class"] == (0 if kind == "Legitimate" else 1)]
        idx = st.selectbox("Transaction ID", subset.index.tolist(), format_func=lambda x: f"TX-{x:06d}")
    row = df.loc[idx]
    x1, x2, x3 = st.columns(3)
    x1.metric("Transaction ID", f"TX-{idx:06d}")
    x2.metric("Amount", f"{row['Amount']:,.2f}")
    x3.metric("Dataset time", f"{row['Time']:,.0f} sec")
    with st.expander("Inspect anonymized model features"):
        st.dataframe(pd.DataFrame([row[features].to_dict()]), use_container_width=True, hide_index=True)
    if st.button("⚡ RUN FRAUD ANALYSIS", type="primary"):
        try:
            sample = pd.DataFrame([row[features].to_dict()], columns=features)
            prediction = int(model.predict(sample)[0])
            proba = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(sample)[0]
                classes = list(model.classes_)
                if 1 in classes:
                    proba = float(probs[classes.index(1)])
            st.session_state.last_scan = {
                "idx": int(idx), "prediction": prediction, "probability": proba,
                "actual": int(row["Class"])
            }
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    result = st.session_state.get("last_scan")
    if result and result["idx"] == int(idx):
        st.markdown("---")
        left, right = st.columns([1, 1.5])
        with left:
            st.markdown("### ANALYSIS RESULT")
            if result["prediction"] == 1:
                st.error("🚨 HIGH RISK — Potential fraud")
            else:
                st.success("🟢 LOW RISK — Model classified as legitimate")
            if result["probability"] is not None:
                p = result["probability"]
                st.metric("Model-estimated fraud probability", f"{p:.2%}")
                st.progress(max(0.0, min(1.0, p)))
                st.caption("Probability is the model's estimate, not a guarantee.")
        with right:
            st.markdown("### Transaction intelligence")
            st.metric("Model verdict", "POTENTIAL FRAUD" if result["prediction"] else "LEGITIMATE")
            st.metric("Dataset label", "Fraud-labelled" if result["actual"] else "Legitimate-labelled")
            st.write("**Check:** " + ("Prediction matches the historical label." if result["prediction"] == result["actual"] else "Prediction differs from the historical label."))
        st.caption("Educational demo using historical data. Not connected to a bank or live payment system.")

elif page == "Overview":
    st.subheader("System overview")
    c = st.columns(4)
    for col, label, value in zip(c, ["Transactions", "Fraud-labelled", "Legitimate", "Fraud rate"],
                                 [f"{total:,}", f"{fraud:,}", f"{legit:,}", f"{fraud/total:.3%}" if total else "0%"]):
        col.metric(label, value)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Dataset balance")
        chart(px.bar(x=["Legitimate", "Fraud"], y=[legit, fraud], color=["Legitimate", "Fraud"],
                     color_discrete_map={"Legitimate": "#28b887", "Fraud": "#ff5364"},
                     labels={"x": "Label", "y": "Transactions"}), "class_balance")
    with right:
        st.markdown("#### Transaction amount profile")
        chart(px.histogram(df, x="Amount", color=df["Class"].map({0: "Legitimate", 1: "Fraud"}),
                           nbins=40, log_y=True, labels={"color": "Dataset label", "Amount": "Amount"}),
              "amount_profile")
    left, right = st.columns(2)
    with left:
        avg = df.groupby("Class")["Amount"].mean().reindex([0, 1], fill_value=0)
        chart(px.bar(x=["Legitimate", "Fraud"], y=avg.values, color=["Legitimate", "Fraud"],
                     color_discrete_map={"Legitimate": "#28b887", "Fraud": "#ff5364"},
                     labels={"x": "Label", "y": "Average amount"}), "average_amount")
    with right:
        if "Time" in df and fraud:
            hourly = (df.loc[df["Class"] == 1, "Time"] // 3600).astype(int).value_counts().sort_index()
            chart(px.line(x=hourly.index, y=hourly.values, markers=True,
                          labels={"x": "Dataset time bucket (hour)", "y": "Fraud-labelled rows"}),
                  "fraud_timeline")
    st.markdown("#### Explore transactions")
    kind = st.selectbox("Dataset label", ["All", "Legitimate", "Fraud"])
    amount = st.slider("Amount range", float(df.Amount.min()), float(df.Amount.max()),
                       (float(df.Amount.min()), float(df.Amount.max())))
    view = df[df.Amount.between(*amount)]
    if kind != "All":
        view = view[view.Class == (1 if kind == "Fraud" else 0)]
    st.caption(f"{len(view):,} rows match. Showing up to 300.")
    st.dataframe(view[["Time", "Amount", "Class"]].head(300), use_container_width=True, hide_index=True)

elif page == "Model Lab":
    st.subheader("Model performance lab")
    st.caption("Saved evaluation metrics from the held-out test set.")
    for labels, keys, fmts in [
        (["Fraud precision", "Fraud recall", "Fraud F1-score"],
         ["fraud_precision", "fraud_recall", "fraud_f1_score"], [".1%", ".1%", ".3f"]),
        (["ROC-AUC", "Average precision", "Test rows"],
         ["roc_auc", "average_precision", "test_rows"], [".3f", ".3f", ","])
    ]:
        cols = st.columns(3)
        for col, label, key, fmt in zip(cols, labels, keys, fmts):
            value = metrics.get(key)
            col.metric(label, format(value, fmt) if value is not None else "N/A")
    cm = metrics.get("confusion_matrix")
    if cm:
        st.markdown("#### Confusion matrix")
        chart(go.Figure(go.Heatmap(z=cm, x=["Predicted legit", "Predicted fraud"],
                                   y=["Actual legit", "Actual fraud"], text=cm, texttemplate="%{text}",
                                   colorscale=[[0, "#101b2c"], [1, "#27c99a"]],
                                   hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>")),
              "confusion_matrix", 390)
    report = metrics.get("classification_report", {})
    if report:
        st.markdown("#### Classification report")
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    roc, pr = metrics.get("roc_curve", {}), metrics.get("precision_recall_curve", {})
    left, right = st.columns(2)
    with left:
        st.markdown("#### ROC curve")
        if "fpr" in roc and "tpr" in roc:
            fig = px.line(x=roc["fpr"], y=roc["tpr"], labels={"x": "False positive rate", "y": "True positive rate"},
                          title=f"ROC-AUC = {metrics.get('roc_auc', 0):.3f}")
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(dash="dash"))
            chart(fig, "roc_curve")
        else:
            st.info("ROC coordinates not found in metrics.json.")
    with right:
        st.markdown("#### Precision-recall curve")
        if "precision" in pr and "recall" in pr:
            chart(px.line(x=pr["recall"], y=pr["precision"],
                          labels={"x": "Recall", "y": "Precision"},
                          title=f"Average Precision = {metrics.get('average_precision', 0):.3f}"), "pr_curve")
        else:
            st.info("Precision-recall coordinates not found in metrics.json.")
    st.info("Fraud recall shows how many actual fraud cases the model catches. Accuracy alone can mislead on imbalanced datasets.")

else:
    st.subheader("About FraudShield")
    st.markdown("""
**Goal:** Online credit-card fraud detection using machine learning  
**Algorithm:** Saved Random Forest classifier  
**Data:** Historical, anonymized credit-card transactions  
**Flow:** Load dataset → run saved model → compare predictions with labels → inspect evaluation metrics.

**Limitations:** This is a college demonstration, not a bank integration or a live fraud prevention service.
""")
    cols = st.columns(3)
    cols[0].metric("Model", metrics.get("model_name", "Random Forest"))
    cols[1].metric("Training rows", f"{metrics.get('training_rows', 0):,}")
    cols[2].metric("Features", len(features))

st.markdown("---")
st.caption("FRAUDSHIELD AI  •  MACHINE LEARNING SECURITY LAB")
