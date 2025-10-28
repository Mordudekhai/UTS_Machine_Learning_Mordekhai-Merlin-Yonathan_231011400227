# ============================================================
# Pipeline lengkap EDA + Preprocessing + Evaluasi
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import os

# ----------------------------------------
# BAGIAN 1: EDA (Exploratory Data Analysis)
# ----------------------------------------
df = pd.read_csv("data/income.csv")
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
df.replace('?', np.nan, inplace=True)

print("Jumlah data:", df.shape)
print("Kolom:", df.columns.tolist())
print("Jumlah Missing per kolom:\n", df.isnull().sum())

if 'age' in df.columns:
    plt.figure(figsize=(6,4))
    df['age'].hist(bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribusi Umur")
    plt.tight_layout()
    os.makedirs("eda_outputs", exist_ok=True)
    plt.savefig("eda_outputs/hist_age.png")

# ----------------------------------------
# BAGIAN 2: PREPROCESSING DATA
# ----------------------------------------
df.dropna(inplace=True)
target = 'income'
y = df[target].apply(lambda v: 1 if '>50K' in str(v) else 0)
X = df.drop(columns=[target])

num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ----------------------------------------
# BAGIAN 3: MODEL DAN EVALUASI
# ----------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

results = []
os.makedirs("model_outputs", exist_ok=True)

for name, model in models.items():
    pipeline = Pipeline([('preprocess', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n=== {name} ===")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    # Simpan ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"model_outputs/roc_{name.replace(' ','_')}.png")
    plt.close()

    results.append([name, acc, prec, rec, f1, auc])

pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1-score","ROC-AUC"])\
  .to_csv("model_outputs/model_metrics_summary.csv", index=False)

print("\nEvaluasi selesai. Hasil disimpan di folder 'model_outputs/'.")
