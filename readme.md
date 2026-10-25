# FraudShield AI 🛡️

An AI/ML-based credit card fraud detection system built using Python, Scikit-learn, Pandas, and Streamlit.

## 📌 About the Project

FraudShield AI is a machine learning project that analyzes credit card transaction data and predicts whether a transaction is legitimate or potentially fraudulent.

The project uses a Random Forest Classifier trained on historical transaction data.

## ✨ Features

- Machine learning-based fraud classification
- Interactive Streamlit dashboard
- Model performance evaluation
- Confusion matrix visualization
- Precision, Recall, F1-score, and ROC-AUC metrics
- Saved model for reuse without retraining every time

## 🛠️ Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib

## 📂 Project Structure

```text
onlinefrauddetection/
├── app.py
├── train_model.py
├── requirements.txt
├── dataset/
│   └── creditcard.csv
└── artifacts/
    ├── fraud_model.joblib
    ├── metrics.json
    └── confusion_matrix.csv
```

## ⚙️ How to Run

### 1. Clone or download the project

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model if needed

```bash
python train_model.py
```

### 5. Start the dashboard

```bash
streamlit run app.py
```

## 📊 Model Evaluation

The model is evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Confusion Matrix

## ⚠️ Limitations

This project is an educational prototype. It is not connected to a real bank or live payment system. Predictions depend on the dataset, model, and input features. The model requires the same feature format used during training.

## 👨‍💻 Project Purpose

This project demonstrates practical applications of supervised machine learning, classification, model evaluation, and interactive dashboard development.