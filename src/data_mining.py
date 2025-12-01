import pandas as pd

def run_data_mining():
    print("===== DATA MINING =====\n")

    df = pd.read_csv("data/heart_attack_prediction_indonesia.csv")

    print(">> 5 Baris Pertama:")
    print(df.head(), "\n")

    print(">> Info Dataset:")
    df.info()
    print()

    print(">> Statistik Deskriptif:")
    print(df.describe(), "\n")

    print(">> Missing Values (Before Cleaning):")
    print(df.isna().sum(), "\n")

    print(">> Jumlah Data:", len(df), "\n")
