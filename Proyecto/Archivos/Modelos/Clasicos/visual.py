import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SALIDAS_ROOT = os.path.join("Proyecto", "Data", "Salidas")


CLASICOS_DIR = os.path.join(SALIDAS_ROOT, "modelos", "clasicos")


REPORT_PATH = os.path.join(SALIDAS_ROOT,"reportes_clasicos","report_clasicos.csv")


CM_DIR = os.path.join(SALIDAS_ROOT, "reportes_clasicos")


GRAF_DIR = os.path.join(CLASICOS_DIR, "graficas")
os.makedirs(GRAF_DIR, exist_ok=True)


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
    plt.imshow(cm, cmap="Blues")
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
    print(" Guardado:", out_path)


def main():
    if not os.path.exists(REPORT_PATH):
        raise FileNotFoundError(f"No se encontró {REPORT_PATH}. Corre primero entrenar_clasicos.py")

    df = pd.read_csv(REPORT_PATH)
    print("Reporte leído:")
    print(df)


    plot_bar_metric(df, "f1_test", "Comparativa F1 (test) modelos clásicos", "f1_test_clasicos.png")
    plot_bar_metric(df, "prec_test", "Comparativa Precisión (test)", "precision_test_clasicos.png")
    plot_bar_metric(df, "rec_test", "Comparativa Recall (test)", "recall_test_clasicos.png")
    plot_bar_metric(df, "acc_test", "Comparativa Accuracy (test)", "accuracy_test_clasicos.png")


    for _, row in df.iterrows():
        modelo = row["modelo"]
        cm_file = row["cm_path"] 


        cm_path = os.path.join(CM_DIR, cm_file)


        if not os.path.exists(cm_path):
            cm_path_alt = os.path.join(SALIDAS_ROOT, cm_file)
            if os.path.exists(cm_path_alt):
                cm_path = cm_path_alt

        if os.path.exists(cm_path):
            cm = np.load(cm_path)
            plot_confusion(
                cm,
                labels=["Real", "Fake"],
                title=f"Matriz de confusión - {modelo}",
                fname=f"cm_{modelo}.png"
            )
        else:
            print(f"No se encontró la matriz de confusión para {modelo}: {cm_path}")


if __name__ == "__main__":
    main()
