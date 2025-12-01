import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def run_modeling():
    print("===== PREDICTIVE MODELING =====")
    print("1. Algoritma yang digunakan: KNN, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting")
    print("2. Pembagian data: Train-Test Split (80% training, 20% testing)")
    print("3. Hasil evaluasi model: Accuracy, Precision, Recall, F1-Score")
    print("4. Visualisasi hasil model: Confusion Matrix, ROC Curve")
    print(">> Menggunakan library scikit-learn (sklearn) untuk machine learning.")

    # Load data
    try:
        X_train = pd.read_csv("data/features_train.csv")
        X_test = pd.read_csv("data/features_test.csv")
        y_train = pd.read_csv("data/target_train.csv").values.ravel()
        y_test = pd.read_csv("data/target_test.csv").values.ravel()
        print(">> Data training dan testing berhasil dimuat.")
    except FileNotFoundError:
        print("ERROR: File data training/testing tidak ditemukan!")
        return

    # Models to train (supervised classification algorithms)
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_f1 = 0
    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    print("\n--- RINGKASAN PERBANDINGAN MODEL ---")
    for name, metrics in results.items():
        print(f"{name}: F1-Score = {metrics['F1-Score']:.4f}")

    print(f"\n>> Model terbaik: {type(best_model).__name__} dengan F1-Score {best_f1:.4f}")

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_heart_attack_model.pkl")
    print(">> Model terbaik disimpan sebagai 'models/best_heart_attack_model.pkl'")

    # Classification report for best model
    y_pred_best = best_model.predict(X_test)
    print("\n--- LAPORAN KLASIFIKASI MODEL TERBAIK ---")
    print(classification_report(y_test, y_pred_best))

    print("\n--- MATRIKS KEKERUTAN MODEL TERBAIK ---")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Attack', 'Heart Attack'], yticklabels=['No Heart Attack', 'Heart Attack'])
    plt.title('Confusion Matrix - Model Terbaik')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("output/model_confusion_matrix.png")
    plt.close()
    print(">> Plot matriks kebingungan disimpan sebagai 'output/model_confusion_matrix.png'")

    # Plot ROC curve for best model (if applicable)
    if hasattr(best_model, "predict_proba"):
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Model Terbaik')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("output/model_roc_curve.png")
        plt.close()
        print(">> Plot kurva ROC disimpan sebagai 'output/model_roc_curve.png'")

if __name__ == "__main__":
    run_modeling()
