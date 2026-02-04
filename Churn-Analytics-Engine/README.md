\# Customer Churn Prediction System



An end-to-end Machine Learning solution designed to identify high-risk customers for a telecommunications company. This project demonstrates a full data science lifecycle: from advanced preprocessing and handling class imbalance to deploying a production-ready web application.



\## Technical Highlights



\* \*\*Advanced Algorithm:\*\* Utilizes \*\*XGBoost (Extreme Gradient Boosting)\*\* for high-precision classification.

\* \*\*Class Imbalance Management:\*\* Implemented \*\*SMOTE\*\* (Synthetic Minority Over-sampling Technique) to ensure the model accurately identifies churners in an imbalanced dataset.

\* \*\*Hybrid Feature Engineering:\*\* Optimized the feature space to \*\*31 dimensions\*\* using a custom pipeline of Label Encoding (for binary features) and One-Hot Encoding (for multi-category features).

\* \*\*Model Persistence:\*\* Integrated `pickle` for model serialization, allowing for instant predictions without re-training.

\* \*\*Interactive Deployment:\*\* A fully functional \*\*Streamlit\*\* web dashboard for real-time risk assessment.



\##  Tech Stack



\* \*\*Language:\*\* Python 3.x

\* \*\*ML Frameworks:\*\* Scikit-Learn, XGBoost

\* \*\*Sampling:\*\* Imbalanced-learn (SMOTE)

\* \*\*Web Framework:\*\* Streamlit

\* \*\*Data Science:\*\* Pandas, NumPy, Matplotlib, Seaborn



\## Project Structure



\* `app.py`: The Streamlit web application script.

\* `train\_model.py`: The training pipeline including cleaning, SMOTE, and model saving.

\* `model.pkl`: The serialized XGBoost "brain."

\* `scaler.pkl`: The saved StandardScaler for data consistency.

\* `columns.pkl`: Metadata for ensuring feature alignment in production.



\## ⚙️ Installation \& Usage



1\. \*\*Clone the repository:\*\*

&nbsp;  ```bash

&nbsp;  git clone 

&nbsp;  cd YOUR\_REPO\_NAME

2\. Install the required libraries: Open your terminal (or Anaconda Prompt) and run

pip install streamlit pandas numpy xgboost scikit-learn imbalanced-learn

3\. Launch the Application:

streamlit run app.py



📊 Evaluation \& Results

The model was evaluated using a Confusion Matrix to ensure a balance between catching actual churners (Recall) and minimizing false alarms (Precision). With the 31-column optimized input, the model achieves a strong F1-Score, making it highly reliable for business retention strategies.




