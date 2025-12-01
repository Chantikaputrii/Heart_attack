import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def run_eda():
    # ============== EXPLORATORY DATA ANALYSIS ==============

    # Create output folder if not exists
    os.makedirs("output", exist_ok=True)

    print("===== EDA =====")

    try:
        df = pd.read_csv("data/cleaned_heart_attack.csv")
    except FileNotFoundError:
        print("ERROR: File cleaned_heart_attack.csv tidak ditemukan!")
        exit()

    print(">> Jumlah Data:", len(df))
    print(">> 5 Baris Pertama:")
    print(df.head(), "\n")
