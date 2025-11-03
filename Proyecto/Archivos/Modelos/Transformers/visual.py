
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SALIDAS     = os.path.join("Proyecto", "Data", "Salidas")
REPORT_DIR  = os.path.join(SALIDAS, "reportes_transformers")
MODELOS_DIR = os.path.join(SALIDAS, "modelos", "Transformers")
GRAF_DIR    = os.path.join(MODELOS_DIR, "graficas")
os.makedirs(GRAF_DIR, exist_ok=True)

REPORT_PATH = os.path.join(REPORT_DIR, "report_transformers.csv")

def plot_bar_metric(df, metric_col, title, fname):
    modelos = df["modelo"].tolist()
    valores = df[metric_col].tolist()

    plt.figure()
    plt.bar(modelos, valores)
    plt.title(title)
    plt.ylabel(metric_col)
    plt.tight_layout()
    out_path = os.path.join(GRAF_DIR, fname)
    plt.savefig(out_path)
    plt.close()
    print("Guardado:", out_path)

def plot_confusion(cm, labels, title, fname):
    plt.figure()
    plt.imshow(cm, cmap="gray")
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    out_path = os.path.join(GRAF_DIR, fname)
    plt.savefig(out_path)
    plt.close()
    print("Guardado:", out_path)

def main():
    if not os.path.exists(REPORT_PATH):
        raise FileNotFoundError(f"No se encontró {REPORT_PATH}. Corre primero entrenar_transformers.py")

    df = pd.read_csv(REPORT_PATH)
    print("Reporte leído:")
    print(df)

    # barras de test (coherente con clásicos/secuenciales)
    plot_bar_metric(df, "f1_test",  "F1 (test) - Transformers",      "f1_test_transformers.png")
    plot_bar_metric(df, "prec_test","Precisión (test) - Transformers","precision_test_transformers.png")
    plot_bar_metric(df, "rec_test", "Recall (test) - Transformers",   "recall_test_transformers.png")
    plot_bar_metric(df, "acc_test", "Accuracy (test) - Transformers", "accuracy_test_transformers.png")

    # matrices de confusión
    for _, row in df.iterrows():
        modelo  = row["modelo"]
        cm_file = row["cm_path"]
        cm_path = os.path.join(REPORT_DIR, cm_file)
        if os.path.exists(cm_path):
            cm = np.load(cm_path)
            plot_confusion(cm, ["Real", "Fake"], f"Matriz de confusión - {modelo}", f"cm_{modelo}.png")
        else:
            print(f"No se encontró la matriz de confusión de {modelo}: {cm_path}")

if __name__ == "__main__":
    main()
