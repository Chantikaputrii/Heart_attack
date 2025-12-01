import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_visualization():
    print("===== DATA VISUALIZATION & COMMUNICATION =====")
    
    os.makedirs("output", exist_ok=True)

    # 1. Load Model Terbaik dan Data Testing
    try:
        model_path = "models/best_heart_attack_model.pkl"
        model = joblib.load(model_path)
        
        X_test = pd.read_csv("data/features_test.csv")
        y_test = pd.read_csv("data/target_test.csv")
        feature_names = X_test.columns.tolist()
        
        print(f">> Model '{type(model).__name__}' dan data testing berhasil dimuat.")
    except FileNotFoundError:
        print("ERROR: File model atau data testing tidak ditemukan. Pastikan proses Modeling sudah berjalan sukses.")
        return

    # 2. Visualisasi Utama: Feature Importance
    importance = None
    importance_source = ""

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        importance_source = "Feature Importances"
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
        importance_source = "Coefficients"
    
    if importance is not None:
        feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(10) # Ambil top 10

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
        plt.title(f'Top 10 Faktor Pemicu Heart Attack ({type(model).__name__})')
        plt.xlabel('Tingkat Kepentingan (Importance Score)')
        plt.ylabel('Fitur')
        plt.tight_layout()
        
        output_viz_path = "output/viz_feature_importance.png"
        plt.savefig(output_viz_path)
        plt.close()
        print(f">> Visualisasi insight disimpan: {output_viz_path}")
        
        top_feature = feat_imp.iloc[0]['Feature']
    else:
        print(">> Model ini tidak mendukung visualisasi Feature Importance secara langsung.")
        top_feature = "Tidak dapat diidentifikasi secara langsung oleh model ini"

    # 3. Komunikasi & Interpretasi Hasil
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    report_content = []
    report_content.append("=== LAPORAN KOMUNIKASI HASIL ANALISIS ===")
    report_content.append(f"Model Terbaik: {type(model).__name__}")
    report_content.append("-" * 40)
    
    # Interpretasi Metrik
    report_content.append("1. PERFORMA MODEL")
    report_content.append(f"   - Akurasi ({acc:.1%}): Seberapa sering model memprediksi dengan benar secara keseluruhan.")
    report_content.append(f"   - Recall ({rec:.1%}): Kemampuan model mendeteksi pasien yang BENAR-BENAR terkena serangan jantung.")
    report_content.append(f"     (Penting dalam medis: Kita tidak ingin melewatkan pasien yang sakit).")
    report_content.append(f"   - Precision ({prec:.1%}): Dari pasien yang diprediksi sakit, berapa persen yang benar-benar sakit.")
    
    # Insight
    report_content.append("\n2. INSIGHT UTAMA")
    if importance is not None:
        report_content.append(f"   - Faktor risiko paling dominan adalah: '{top_feature}'.")
        report_content.append(f"   - 10 fitur teratas pada grafik 'viz_feature_importance.png' harus menjadi perhatian utama.")
    else:
        report_content.append("   - Insight fitur spesifik tidak tersedia untuk algoritma ini.")

    # Rekomendasi
    report_content.append("\n3. REKOMENDASI DAN RENCANA TINDAK LANJUT")
    report_content.append("   - Gunakan model ini sebagai alat 'Early Warning System' di fasilitas kesehatan.")
    report_content.append(f"   - Pasien dengan nilai indikator '{top_feature}' yang tinggi harus segera mendapatkan pemeriksaan lanjutan.")
    report_content.append("   - Lakukan validasi medis lebih lanjut terhadap hasil prediksi positif untuk mengurangi kecemasan pasien (jika False Positive).")

    # Cetak ke layar
    print("\n" + "\n".join(report_content))

    # Simpan ke file teks
    report_path = "output/communication_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_content))
    
    print(f"\n>> Laporan lengkap disimpan: {report_path}")

if __name__ == "__main__":
    run_visualization()