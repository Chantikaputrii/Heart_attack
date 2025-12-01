import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_feature_engineering():
    print("===== EDA FEATURE ENGINEERING =====")

    # Create output folder if not exists
    os.makedirs("output", exist_ok=True)

    # Load cleaned dataset
    try:
        df = pd.read_csv("data/cleaned_heart_attack.csv")
        print(">> Dataset dimuat, jumlah data:", len(df))
    except FileNotFoundError:
        print("ERROR: File cleaned_heart_attack.csv tidak ditemukan!")
        return

    # Cek apakah target ada
    if "heart_attack" not in df.columns:
        print("ERROR: Kolom 'heart_attack' tidak ditemukan dalam dataset.")
        return

    # Separate features and target
    X = df.drop("heart_attack", axis=1)
    y = df["heart_attack"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(">> Kolom Kategorikal:")
    for col in categorical_cols:
        print(f"  - {col}")
    print(">> Kolom Numerikal:")
    for col in numerical_cols:
        print(f"  - {col}")

    # Encode categorical variables
    X_encoded = X.copy()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

    print(">> Encoding kategorikal selesai.")

    # Scale numerical features
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    print(">> Scaling numerikal selesai.")

    # Combine back processed dataset
    df_processed = X_encoded.copy()
    df_processed["heart_attack"] = y

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    print(">> Split data: Train =", len(X_train), ", Test =", len(X_test))

    # Save processed files
    df_processed.to_csv("data/heart_attack_processed.csv", index=False)
    X_train.to_csv("data/features_train.csv", index=False)
    X_test.to_csv("data/features_test.csv", index=False)
    y_train.to_csv("data/target_train.csv", index=False)
    y_test.to_csv("data/target_test.csv", index=False)

    print("\n--- FILE YANG BERHASIL DISIMPAN ---")
    print("✓ Dataset yang telah diproses: data/heart_attack_processed.csv")
    print("✓ Fitur training: data/features_train.csv")
    print("✓ Fitur testing: data/features_test.csv")
    print("✓ Target training: data/target_train.csv")
    print("✓ Target testing: data/target_test.csv")
