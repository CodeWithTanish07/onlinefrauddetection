
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "dataset" / "creditcard.csv"
MODEL_PATH = BASE / "artifacts" / "fraud_model.joblib"
METRICS_PATH = BASE / "artifacts" / "metrics.json"

# ---------------- STYLE ----------------
st.markdown("""
<style>
/* Main app */
.stApp {
    background: linear-gradient(135deg, #080d19 0%, #101a2d 55%, #0b1220 100%);
    color: #e8eefc;
}

/* Page spacing */
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1322;
    border-right: 1px solid #24324a;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #dce6fa;
    font-weight: 500;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243a, #111c2e);
    border: 1px solid #2a3b56;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
}

[data-testid="stMetricLabel"] {
    color: #a9bad5;
    font-size: 0.9rem;
}

[data-testid="stMetricValue"] {
    color: #f4f7ff;
    font-weight: 700;
}

/* Headings */
h1, h2, h3 {
    color: #f4f7ff !important;
    letter-spacing: -0.4px;
}

/* Buttons */
div.stButton > button {
    width: 100%;
    min-height: 46px;
    border-radius: 10px;
    border: 1px solid #3b82f6;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    font-weight: 650;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: #93c5fd;
    background: #1d4ed8;
    transform: translateY(-1px);
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #111c2e;
    border-color: #30415e;
    border-radius: 9px;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #2a3b56;
    border-radius: 12px;
    overflow: hidden;
}

/* Dividers and captions */
hr {
    border-color: #263650;
}

.stCaption {
    color: #94a8c7;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ FraudShield AI")
st.caption("INTELLIGENT TRANSACTION MONITORING • MACHINE LEARNING")

# ---------------- LOAD FILES ----------------
for path in [DATA_PATH, MODEL_PATH, METRICS_PATH]:
    if not path.exists():
        st.error(f"Required file nahi mili: {path.relative_to(BASE)}")
        st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def load_metrics(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    df = load_data(str(DATA_PATH))
    metrics = load_metrics(str(METRICS_PATH))
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"File loading error: {error}")
    st.stop()

features = metrics.get(
    "feature_columns",
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
)

if not hasattr(model, "predict"):
    st.error("Saved file valid trained model nahi hai.")
    st.stop()

if not all(col in df.columns for col in features + ["Class"]):
    st.error("Dataset mein required model features nahi mile.")
    st.stop()

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Transaction Detection", "Model Performance", "About Project"]
)

st.sidebar.divider()
st.sidebar.caption("FraudShield AI | College Project")

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.subheader("📊 Dataset Overview")

    total = len(df)
    fraud = int((df["Class"] == 1).sum())
    legit = int((df["Class"] == 0).sum())
    rate = fraud / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Cases", f"{fraud:,}")
    c3.metric("Legitimate Cases", f"{legit:,}")
    c4.metric("Fraud Rate", f"{rate:.3f}%")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("#### Transaction Class Distribution")
        fig, ax = plt.subplots()
        ax.bar(
            ["Legitimate", "Fraud"],
            [legit, fraud]
        )
        ax.set_ylabel("Number of transactions")
        ax.set_title("Dataset labels")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### Transaction Amount Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Amount"].clip(upper=df["Amount"].quantile(0.99)),
                bins=35)
        ax.set_xlabel("Amount (upper 1% capped for display)")
        ax.set_ylabel("Transactions")
        ax.set_title("Amount distribution")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "Dataset statistics actual Class labels se calculate kiye gaye hain. "
        "Ye model predictions ke counts nahi hain."
    )
    st.divider()
    st.subheader("📈 Advanced Fraud Analytics")

    # Fraud vs legitimate transaction amount
    left, right = st.columns(2)

    with left:
        st.markdown("#### Average Transaction Amount")

        avg_amount = df.groupby("Class")["Amount"].mean()

        fig, ax = plt.subplots()
        ax.bar(
            ["Legitimate", "Fraud"],
            [
                avg_amount.get(0, 0),
                avg_amount.get(1, 0)
            ]
        )
        ax.set_ylabel("Average amount")
        ax.set_title("Average Amount by Transaction Class")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### Fraud Transactions Over Time")

        fraud_df = df[df["Class"] == 1].copy()

        if not fraud_df.empty:
            fraud_df["Time (hour)"] = (
                fraud_df["Time"] // 3600
            ).astype(int)

            hourly_fraud = (
                fraud_df.groupby("Time (hour)")
                .size()
                .reindex(range(48), fill_value=0)
            )

            fig, ax = plt.subplots()
            ax.plot(
                hourly_fraud.index,
                hourly_fraud.values,
                marker="o"
            )
            ax.set_xlabel("Dataset time bucket (hour)")
            ax.set_ylabel("Fraud transactions")
            ax.set_title("Fraud Count Across Dataset Timeline")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No fraud-labelled transactions available.")

    st.divider()
    st.markdown("#### 🔎 Transaction Data Explorer")

    class_filter = st.selectbox(
        "Filter transactions",
        ["All", "Legitimate only", "Fraud only"],
        key="overview_class_filter"
    )

    amount_min = float(df["Amount"].min())
    amount_max = float(df["Amount"].max())

    amount_range = st.slider(
        "Transaction amount range",
        min_value=amount_min,
        max_value=amount_max,
        value=(amount_min, amount_max),
        key="overview_amount_range"
    )

    filtered_df = df[
        df["Amount"].between(amount_range[0], amount_range[1])
    ]

    if class_filter == "Legitimate only":
        filtered_df = filtered_df[filtered_df["Class"] == 0]
    elif class_filter == "Fraud only":
        filtered_df = filtered_df[filtered_df["Class"] == 1]

    st.caption(f"{len(filtered_df):,} transactions match your filters.")

    st.dataframe(
        filtered_df[["Time", "Amount", "Class"]].head(500),
        use_container_width=True,
        hide_index=True
    )


# ---------------- TRANSACTION DETECTION ----------------
elif page == "Transaction Detection":
    st.subheader("🔎 Transaction Detection")
    st.write(
        "Historical dataset se example select karo. Prediction ke input "
        "mein actual Class label use nahi hota."
    )

    scenario = st.radio(
        "Example type",
        ["Legitimate transaction", "Fraud transaction"],
        horizontal=True
    )

    class_value = 0 if scenario == "Legitimate transaction" else 1
    indices = df.index[df["Class"] == class_value].tolist()

    if not indices:
        st.warning("Is category mein transaction nahi mili.")
        st.stop()

    examples = indices[:min(100, len(indices))]

    selected = st.selectbox(
        "Select transaction example",
        examples,
        format_func=lambda idx: f"Transaction ID {idx}"
    )

    row = df.loc[selected]

    c1, c2 = st.columns(2)
    c1.metric("Transaction Amount", f"{row['Amount']:.2f}")
    c2.metric("Transaction Time", f"{row['Time']:.2f} sec")

    with st.expander("View anonymized transaction features"):
        st.dataframe(
            pd.DataFrame([row[features].to_dict()]),
            use_container_width=True
        )

    if st.button("🔍 Predict Transaction", type="primary"):
        try:
            input_df = pd.DataFrame(
                [row[features].to_dict()],
                columns=features
            )

            prediction = int(model.predict(input_df)[0])
            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                if 1 in classes:
                    probability = float(
                        probabilities[classes.index(1)]
                    )

            st.divider()
            st.markdown("### Prediction Result")

            if prediction == 1:
                st.error("⚠️ Model Prediction: Potential Fraud")
            else:
                st.success("✅ Model Prediction: Legitimate")

            if probability is not None:
                st.metric(
                    "Estimated Fraud Probability",
                    f"{probability * 100:.2f}%"
                )
                st.progress(max(0.0, min(1.0, probability)))

            # Reveal actual label only after the prediction
            actual = int(row["Class"])
            if prediction == actual:
                st.success("Prediction matches the dataset label.")
            else:
                st.warning("Prediction differs from the dataset label.")

            st.caption(
                "This is a historical dataset demonstration, not a live "
                "bank integration or a guarantee of transaction safety."
            )

        except Exception as error:
            st.error(f"Prediction error: {error}")

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.subheader("📈 Model Performance")
    st.write("Metrics calculated on the held-out test set during training.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Precision", f"{metrics['fraud_precision']:.1%}")
    c2.metric("Fraud Recall", f"{metrics['fraud_recall']:.1%}")
    c3.metric("Fraud F1-score", f"{metrics['fraud_f1_score']:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c5.metric("Average Precision", f"{metrics['average_precision']:.3f}")
    c6.metric("Test Transactions", f"{metrics['test_rows']:,}")

    st.divider()
    st.markdown("#### Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots()
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )
    display.plot(ax=ax, colorbar=False)
    ax.set_title("Held-out Test Set")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Classification Report")
    report = metrics.get("classification_report", {})
    if report:
        st.dataframe(
            pd.DataFrame(report).transpose().round(3),
            use_container_width=True
        )

    st.warning(
        "Fraud recall batata hai ki actual fraud cases mein se kitne detect "
        "hue. High accuracy alone imbalanced data par sufficient nahi hoti."
    )
    st.divider()
    st.markdown("#### 📉 ROC Curve & Precision-Recall Curve")

    # Curves require saved test-set predictions and true labels.
    # Do not calculate them from training data or invent values.
    roc_data = metrics.get("roc_curve", {})
    pr_data = metrics.get("precision_recall_curve", {})

    curve_left, curve_right = st.columns(2)

    with curve_left:
        st.markdown("##### ROC Curve")

        if all(key in roc_data for key in ["fpr", "tpr"]):
            fig, ax = plt.subplots()
            ax.plot(
                roc_data["fpr"],
                roc_data["tpr"],
                label=f"ROC-AUC = {metrics.get('roc_auc', 0):.3f}"
            )
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                "ROC curve data saved nahi hai. "
                "Training script mein test predictions save karni hongi."
            )

    with curve_right:
        st.markdown("##### Precision-Recall Curve")

        if all(key in pr_data for key in ["precision", "recall"]):
            fig, ax = plt.subplots()
            ax.plot(
                pr_data["recall"],
                pr_data["precision"],
                label=(
                    f"Average Precision = "
                    f"{metrics.get('average_precision', 0):.3f}"
                )
            )
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                "Precision-recall curve data saved nahi hai. "
                "Training script update karni hogi."
            )


# ---------------- ABOUT ----------------
else:
    st.subheader("🎓 About This Project")

    st.markdown("""
    **Project:** Online Credit Card Fraud Detection using Machine Learning

    **Algorithm:** Random Forest Classifier

    **Dataset:** Historical credit card transaction dataset

    **Workflow:**
    1. Load and inspect transaction data.
    2. Separate features and the target label.
    3. Split data into training and test sets using stratification.
    4. Train a Random Forest model.
    5. Evaluate predictions using precision, recall, F1-score,
       Average Precision, ROC-AUC and a confusion matrix.
    6. Use the saved model to predict selected transactions.

    **Limitations:** This is an educational prototype. It does not connect
    to a bank or process live transactions. Dataset examples are historical.
    """)

    st.markdown("#### Model details")
    st.write("Algorithm:", metrics.get("model_name", "Random Forest"))
    st.write("Training rows:", f"{metrics['training_rows']:,}")
    st.write("Test rows:", f"{metrics['test_rows']:,}")
    st.write("Features used:", len(features))

st.divider()
st.caption("FraudShield AI • Educational Machine Learning Project")
