# Churn-Analytics-Engine

An end-to-end Machine Learning solution designed to identify high-risk customers for a telecommunications company. This project demonstrates a full data science lifecycle: from advanced preprocessing and handling class imbalance to deploying a production-ready web application.

## Technical Highlights

* **Advanced Algorithm:** Utilizes **XGBoost** (Extreme Gradient Boosting) for high-precision classification.
* **Class Imbalance Management:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model accurately identifies churners in an imbalanced dataset.
* **Hybrid Feature Engineering:** Optimized the feature space to **31 dimensions** using a custom pipeline of Label Encoding (for binary features) and One-Hot Encoding (for multi-category features).
* **Model Persistence:** Integrated `pickle` for model serialization, allowing for instant predictions without re-training.
