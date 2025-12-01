import sys

# Tambahkan folder "src" agar modul bisa ditemukan
sys.path.append("src")

from data_mining import run_data_mining
from data_cleaning import run_cleaning
from data_exploration import run_eda
from feature_engineering import run_feature_engineering
from modeling import run_modeling
# [BARU] Import modul visualization
from data_visualization import run_visualization

def main():
    print("\n==============================")
    print("   MEMULAI DATA PIPELINE ML  ")
    print("==============================\n")

    print(">> Menjalankan Data Mining...\n")
    run_data_mining()

    print("\n>> Menjalankan Data Cleaning...\n")
    run_cleaning()
    
    # Anda bisa mengaktifkan kembali EDA jika diperlukan
    # print("\n>> Menjalankan Data Exploration (EDA)...\n")
    # run_eda()

    print("\n>> Menjalankan Feature Engineering...\n")
    run_feature_engineering()

    print("\n>> Menjalankan Predictive Modeling...\n")
    run_modeling()

    # [BARU] Menjalankan tahap 8: Visualisasi & Komunikasi
    print("\n>> Menjalankan Data Visualization & Communication...\n")
    run_visualization()

    print("\n==============================")
    print("   SEMUA PROSES SUKSES JALAN ")
    print("==============================")

if __name__ == "__main__":
    main()