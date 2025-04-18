# Credit Card Fraud Detection 🛡️

This project aims to detect fraudulent credit card transactions using a machine learning model trained on anonymized transaction data. The model is built using a Random Forest Classifier and evaluated for accuracy, precision, recall, and F1-score.

---

## 🔍 Problem Statement

Credit card fraud causes massive financial loss every year. Detecting fraudulent transactions in real time is a critical challenge due to:
- The **imbalance** between fraudulent and non-fraudulent data.
- The need for **high precision and recall**.
- The complexity of **anonymized features**.

This project provides a machine learning-based solution to predict whether a transaction is fraudulent (`Class = 1`) or not (`Class = 0`).

---

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows**: 284,807
- **Features**:
  - `V1` to `V28`: Anonymized principal components
  - `Time`: Seconds since the first transaction
  - `Amount`: Transaction amount
  - `Class`: Target variable (`1 = Fraud`, `0 = Not Fraud`)

---

## 🧠 Solution Approach

### 1. **Data Preprocessing**
- Drop the `Class` column to separate features and labels.
- Scale the features using `StandardScaler`.

### 2. **Train/Test Split**
- 70% training, 30% testing using `train_test_split`.

### 3. **Model Used**
- `RandomForestClassifier` with 100 estimators and a fixed random seed for reproducibility.

### 4. **Evaluation Metrics**
- **Confusion Matrix**
- **Classification Report** (Accuracy, Precision, Recall, F1-Score)

### 5. **Model Persistence**
- Trained model and scaler are saved using `joblib`.

---

## 🧪 How to Run

```bash
# In Google Colab

# Upload the dataset
from google.colab import files
uploaded = files.upload()

# Paste the rest of the Python code provided in the notebook
