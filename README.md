# Churn Analytics Engine

An end-to-end Machine Learning solution designed to identify high-risk customers for a telecommunications company. This project demonstrates a full data science lifecycle: from advanced preprocessing and handling class imbalance to deploying a production-ready web application.

---

##  Features

*   **Predictive Modeling:** Uses **XGBoost** for high-precision classification.
*   **Balance Correction:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to handle imbalanced datasets.
*   **Feature Engineering:** Custom pipeline with Label Encoding and One-Hot Encoding resulting in 31 optimized features.
*   **Real-time Dashboard:** Interactive **Streamlit** interface for instant customer risk assessment.
*   **Scalable Pipeline:** Standardized scaling and preprocessing for consistent production results.

## Tech Stack

*   **Language:** Python 3.x
*   **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`
*   **Visualization:** `matplotlib`, `seaborn`, `streamlit`
*   **Serialization:** `pickle`

##  Project Structure

*   `customer-churn-prediction.py`: Training pipeline (data cleaning, SMOTE, and model saving).
*   `churn-dashboard.py`: Streamlit-based web application for user interaction.
*   `model.pkl`: Serialized XGBoost model.
*   `scaler.pkl`: Serialized `StandardScaler` object.
*   `columns.pkl`: Metadata for feature alignment.
*   `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Raw dataset.

## Installation & Usage

### 1. Prerequisites
Ensure you have Python installed. It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or manually install: `pip install streamlit pandas numpy xgboost scikit-learn imbalanced-learn`)*

### 3. Run the Dashboard
```bash
streamlit run churn-dashboard.py
```

## Performance
The model is evaluated using **F1-Score** and **Confusion Matrices** to ensure a balance between Precision (minimizing false alarms) and Recall (catching actual churners).
