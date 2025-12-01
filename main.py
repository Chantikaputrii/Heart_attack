# main.py
# Menjalankan seluruh proses: Data Mining → Cleaning → EDA → Feature Engineering

import sys

# Tambahkan folder "src" agar modul bisa ditemukan
sys.path.append("src")

from data_mining import run_data_mining
from data_cleaning import run_cleaning
from data_exploration import run_eda
from feature_engineering import run_feature_engineering
from modeling import run_modeling

def main():
    print("\n==============================")
    print("   MEMULAI DATA PIPELINE ML  ")
    print("==============================\n")

    print(">> Menjalankan Data Mining...\n")
    run_data_mining()

    print("\n>> Menjalankan Data Cleaning...\n")
    run_cleaning()

    print("\n>> Menjalankan Feature Engineering...\n")
    run_feature_engineering()

    print("\n>> Menjalankan Predictive Modeling...\n")
    run_modeling()

    print("\n==============================")
    print("   SEMUA PROSES SUKSES JALAN ")
    print("==============================")

if __name__ == "__main__":
    main()
