# 🛡️ SmartPay Guard – ML-Based Fraud Detection System

> **A beginner-friendly, end-to-end Machine Learning web application designed to detect suspicious and fraudulent financial transactions in real time.**

---

## 📌 Project Overview

In digital payment platforms (credit cards, mobile wallets, banking APIs, UPI), fraud detection is a critical challenge. Rule-based systems (like *"block any transaction over $1,000"*) fail because legitimate customers get blocked while clever fraudsters slip through.

**SmartPay Guard** uses **Machine Learning (Random Forest Classification)** to inspect multiple behavioral signals simultaneously—such as transaction amount, hour of day, location distance, account age, and failed logins—to calculate an accurate **Fraud Probability Score** and provide an **instant plain-English explanation** of the decision.

---

## 🌟 Key Features

1. **Synthetic Data Engine (`data/generate_data.py`)**: Automatically simulates 5,000 realistic financial transactions with real-world fraud patterns (night spikes, high transfers, suspicious bursts).
2. **Preprocessing Pipeline (`models/train_model.py`)**: Handles numerical scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) seamlessly inside a Scikit-Learn `Pipeline`.
3. **Random Forest Classifier**: Robust ensemble of 120 decision trees trained with `class_weight='balanced'` to effectively handle fraud class imbalance.
4. **Comprehensive Evaluation**: Measures **Accuracy (93.8%)**, **Precision (82.0%)**, **Recall (65.1%)**, **F1-Score (72.6%)**, and **ROC-AUC (91.5%)**.
5. **Interactive Web Application (`app.py`)**:
   - **Live Predictor**: Input transaction parameters or click **1-Click Demo Presets** to test immediately.
   - **Fraud Probability Gauge**: Shows exact risk percentage (e.g. 92.7% Fraud Risk).
   - **Explainability Engine**: Explains in plain English *why* a transaction was flagged (e.g., midnight hour + 3 failed logins + foreign location).
   - **EDA Analytics Dashboard**: Visual charts showing transaction distributions, hourly fraud curves, and payment type risk.
   - **Model Metrics Explorer**: Confusion matrix heatmap and plain-English metric explanations.
   - **Dataset Viewer**: Filter and inspect the underlying data.

---

## 📂 Project Structure

```text
SmartPay-Fraud-Detection/
│
├── data/
│   ├── generate_data.py       # Script that creates realistic transaction dataset
│   └── transactions.csv       # 5,000 synthetic transaction records
│
├── models/
│   ├── train_model.py         # Preprocessing, model training & evaluation script
│   ├── fraud_model.joblib     # Saved, production-ready Random Forest pipeline
│   └── model_metrics.json     # Saved accuracy, precision, recall, and feature importances
│
├── app.py                     # Streamlit web application with UI, predictor & dashboard
├── requirements.txt           # Python dependencies
├── run_project.bat            # 1-click Windows starter script
└── README.md                  # Complete documentation and beginner guide
```

---

## 🚀 How to Run the Project (Step-by-Step for Complete Beginners)

You do **NOT** need prior programming experience. Simply follow these steps:

### Option A: The Easiest 1-Click Method (Windows)
Double-click the **`run_project.bat`** file inside this folder.
- It will verify Python, train the model, and automatically open the web application in your browser!

---

### Option B: Using the Terminal / Command Prompt

#### Step 1: Open Terminal in this folder
- On Windows: Press `Win + R`, type `powershell` or `cmd`, and press Enter.
- Navigate to the project folder:
  ```powershell
  cd C:\Users\win10\Desktop\SmartPay-Fraud-Detection
  ```

#### Step 2: Install Required Libraries (If not already installed)
Run:
```bash
pip install -r requirements.txt
```

#### Step 3: (Optional) Re-generate Dataset
To generate fresh transaction data:
```bash
python data/generate_data.py
```
*(This creates `data/transactions.csv` with 5,000 transactions).*

#### Step 4: Train and Evaluate the Model
Run:
```bash
python models/train_model.py
```
You will see the model training output in your terminal, showing accuracy, precision, recall, and the confusion matrix. The trained model is automatically saved to `models/fraud_model.joblib`.

#### Step 5: Start the Streamlit Web Application
Run:
```bash
python -m streamlit run app.py
```
*(Or simply `streamlit run app.py`)*

Your default browser will immediately open with the application running at:
`http://localhost:8501`

---

## 🎯 How to Demo the Project at a Hackathon

When presenting to judges:

1. **Open the Live Predictor Tab**:
   - Click the green button: **"🟢 Load Typical Normal Transaction"**.
   - Click **"🔍 Analyze Transaction with AI"**.
   - Show the judges: **✅ Approved (Green)** with Low Risk (e.g. 7.3% risk).
   - Point out the explanation: normal amount, verified device, daytime.

2. **Trigger Fraud Detection**:
   - Click the red button: **"🚨 Load High-Risk Suspicious Transaction"**.
   - Click **"🔍 Analyze Transaction with AI"**.
   - Show the judges: **🚨 Flagged as Suspicious (Red)** with High Risk (e.g. 92.7% risk).
   - Show the **Factor Breakdown**:
     - *Large transfer ($4,850.00)*
     - *Midnight hour (03:00 AM)*
     - *3 failed password attempts prior to transfer*
     - *International / foreign location (1,850 km away)*
   - Show the **Feature Importance Chart** explaining the model's top predictive factors.

3. **Show the Fraud Analytics Dashboard**:
   - Switch to the **📊 Fraud Analytics Dashboard** tab.
   - Point out the midnight fraud spike chart and the difference between average normal vs fraud amounts.

4. **Show Model Metrics & Evaluation**:
   - Switch to the **🎯 Model Performance & Metrics** tab.
   - Explain the Confusion Matrix and why **Recall** (catching real fraud) is critical in FinTech.

---

## 🧠 Machine Learning Concepts Explained for Beginners

- **What is a Feature?**
  A feature is any data point or clue about the transaction (e.g., amount, hour of the day, distance).
- **What is Random Forest?**
  Think of Random Forest as a committee of 120 digital detectives (decision trees). Each detective looks at a different random subset of clues and votes whether the transaction looks normal or suspicious. The majority vote wins!
- **What is Precision?**
  Out of all transactions our system flagged as fraud, how many were *actual* fraud? High precision means fewer angry, innocent customers.
- **What is Recall?**
  Out of all the real fraud attacks that happened, how many did our system catch? High recall means fewer stolen dollars.

---

## 🛠️ Built With
- **Python 3.14**
- **Scikit-Learn** (Random Forest Classifier, Pipelines, One-Hot Encoding, StandardScaler)
- **Pandas & NumPy** (Data processing and simulation)
- **Streamlit** (Interactive web user interface)
- **Matplotlib** (Data visualization & charts)
- **Joblib** (Model serialization and loading)

---

## 📄 License
This project is open-source and free to use for educational, hackathon, and portfolio purposes.
