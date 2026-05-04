import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
info = pd.read_csv('data/train.csv')
print(f"Sales project: Loaded {info.shape[0]} rows x {info.shape[1]} cols")

# Prepare ML
ml_df = info.copy()
y = (ml_df['Profit'] > 0).astype(int)
chosen_cols = ['Category', 'Sub-Category', 'Segment', 'Region', 'Ship Mode', 'State', 'Discount', 'Quantity', 'Sales']
X = ml_df[chosen_cols].copy()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

cleaner = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])
ml_model = Pipeline(steps=[('preprocessor', cleaner), ('classifier', LogisticRegression(max_iter=2000, solver='saga'))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
ml_model.fit(X_train, y_train)
y_pred = ml_model.predict(X_test)
y_prob = ml_model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
try:
    roc = roc_auc_score(y_test, y_prob)
except Exception:
    roc = float('nan')

print(f"Sales model accuracy: {acc:.4f}")
print(f"Sales model ROC-AUC: {roc:.4f}")
print('\nDone.')
