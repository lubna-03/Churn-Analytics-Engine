#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap -q')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import classification_report,f1_score,confusion_matrix

import xgboost as xgb
from imblearn.over_sampling import SMOTE


# In[3]:


df=pd.read_csv(r"C:\Users\HP\Downloads\archive (5)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df=df.drop('customerID',axis=1)

df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(),inplace=True)

df['Churn']=df['Churn'].map({'Yes':1,'No':0})

print("Shape:",df.shape)
print("churn rate:",round(df['Churn'].mean()*100,2),"%")
df.head(3)


# In[4]:


fig, ax=plt.subplots(1,3,figsize=(15,5))
sns.countplot(x='Contract',hue='Churn',data=df,ax=ax[0])
ax[0].set_title('Churn by Contract')
sns.boxplot(x='Churn',y='tenure',data=df,ax=ax[1])
ax[1].set_title('Tenure by Churn')
sns.boxplot(x='Churn',y='MonthlyCharges',data=df,ax=ax[2])
ax[2].set_title('Monthly Charges by Churn')
plt.tight_layout()
plt.show()


# In[5]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. List the columns you want to turn into 0 and 1 (Label Encoding)
label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

# 2. Apply LabelEncoder to those specific columns
le = LabelEncoder()
for col in label_cols:
    # We use fit_transform to change 'Yes/No' to 1/0
    df[col] = le.fit_transform(df[col])

# 3. Apply Dummies to the REST of the categorical columns
# This automatically skips the columns we already turned into numbers
df = pd.get_dummies(df, drop_first=True)

# 4. Standardize the numeric columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("Encoding Complete!")
df.head(5)


# In[6]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Before SMOTE - Train churn rate:", round(y_train.mean()*100, 2), "%")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE - Train churn rate:", round(y_train_smote.mean()*100, 2), "%")
print("New training size:", X_train_smote.shape)


# In[7]:


#train xgboost model
model=xgb.XGBClassifier(
n_estimator=500,
learning_rate=0.05,
max_depth=6,
random_state=42,
eval_metric='logloss'

)
model.fit(X_train_smote,y_train_smote)


pred=model.predict(X_test)
pred_prob=model.predict_proba(X_test)[:,1]

print("Model Performance on Test Set:")
print(classification_report(y_test, pred))

f1 = f1_score(y_test, pred)
print(f"\nF1-Score (main metric for churn): {f1:.4f}")


cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Stay', 'Predicted Churn'],
            yticklabels=['Actual Stay', 'Actual Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[8]:


import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save column names so the app knows the exact order
with open('columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Files saved: model.pkl, scaler.pkl, columns.pkl")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




