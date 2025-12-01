import pandas as pd

def run_cleaning():
    # ============== DATA CLEANING ==============

    df = pd.read_csv("data/heart_attack_prediction_indonesia.csv")

    print("===== DATA CLEANING =====")

    print("\n>> Missing Values (Before Cleaning):")
    print(df.isna().sum())

    print("\n>> Duplikasi (Before Cleaning):", df.duplicated().sum())

    df_clean = df.copy()

    # Hapus duplikasi
    df_clean = df_clean.drop_duplicates()

    # Tangani missing values
    for col in df_clean.columns:
        if df_clean[col].dtype != "object":
            df_clean.fillna({col: df_clean[col].median()}, inplace=True)
        else:
            df_clean.fillna({col: df_clean[col].mode()[0]}, inplace=True)

    print("\n>> Missing Values (After Cleaning):")
    print(df_clean.isna().sum())

    print("\n>> Duplikasi (After Cleaning):", df_clean.duplicated().sum())

    print("\n>> Jumlah Data Setelah Cleaning:", len(df_clean))

    print("\n>> Tipe Data Setelah Cleaning:")
    print(df_clean.dtypes)

    # Simpan hasil cleaning
    df_clean.to_csv("data/cleaned_heart_attack.csv", index=False)
    print("\n>> File 'data/cleaned_heart_attack.csv' berhasil disimpan.\n")
