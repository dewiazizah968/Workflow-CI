import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("telco_churn_preprocessing.csv")

for col in df.columns:
    if df[col].dtype == "int64":
        df[col] = df[col].astype("float64")

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data training : {X_train.shape[0]} baris")
print(f"Data testing  : {X_test.shape[0]} baris")
print(f"Distribusi target train: {y_train.value_counts().to_dict()}")

# =========================================================
# 2. TRAINING DENGAN MLFLOW AUTOLOG
# =========================================================
mlflow.set_experiment("telco_churn_experiment")

# Cek apakah sudah ada active run dari mlflow run command
# Kalau ada, pakai run yang sudah ada. Kalau tidak, buat baru.
active_run = mlflow.active_run()
if active_run:
    print(f"Menggunakan active run: {active_run.info.run_id}")
    run_context = mlflow.start_run(run_id=active_run.info.run_id, nested=True)
else:
    print("Membuat run baru...")
    run_context = mlflow.start_run(run_name="RandomForest_Autolog")

with run_context:
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print("\n=== Hasil Evaluasi Model ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nModel berhasil dicatat di MLflow!")
