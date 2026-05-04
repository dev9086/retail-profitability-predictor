import argparse
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

def run(save_dir="artifacts"):
    os.makedirs(save_dir, exist_ok=True)
    info = pd.read_csv('data/train.csv')
    print(f"Sales project: Loaded {info.shape[0]} rows x {info.shape[1]} cols")

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

    # Save model
    with open(os.path.join(save_dir, 'sales_model.pkl'), 'wb') as f:
        pickle.dump(ml_model, f)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC={roc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Sales Model ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_sales.png'))
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_sales.png'))
    plt.close()

    print(f"Artifacts saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', default='artifacts', help='Directory to save models and plots')
    args = parser.parse_args()
    run(save_dir=args.save_dir)
