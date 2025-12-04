import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SALIDAS = os.path.join("Proyecto", "Data", "Salidas")
REPORT_DIR = os.path.join(SALIDAS, "reportes_secuenciales")
MODELOS_DIR = os.path.join(SALIDAS, "modelos", "secuenciales")
GRAF_DIR = os.path.join(MODELOS_DIR, "graficas")
os.makedirs(GRAF_DIR, exist_ok=True)

REPORT_PATH = os.path.join(REPORT_DIR, "report_secuenciales.csv")

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
    print(" Guardado:", out_path)

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
    print("Guardado:", out_path)

def plot_training_curves(history_file, model_name):
    history_path = os.path.join(REPORT_DIR, history_file)

    if not os.path.exists(history_path):
        print(f"No se encontró {history_path}")
        return

    history = np.load(history_path, allow_pickle=True).item()
    epochs = range(1, len(history["loss"]) + 1)

    # --- Loss ---
    plt.figure()
    plt.plot(epochs, history["loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title(f"Loss por época - {model_name}")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(GRAF_DIR, f"loss_{model_name}.png")
    plt.savefig(out_path)
    plt.close()
    print("Guardado:", out_path)

    # --- Accuracy ---
    plt.figure()
    plt.plot(epochs, history["accuracy"], label="Train Acc")
    plt.plot(epochs, history["val_accuracy"], label="Val Acc")
    plt.title(f"Accuracy por época - {model_name}")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(GRAF_DIR, f"accuracy_{model_name}.png")
    plt.savefig(out_path)
    plt.close()
    print("Guardado:", out_path)

def main():
    if not os.path.exists(REPORT_PATH):
        raise FileNotFoundError(f"No se encontró {REPORT_PATH}. Corre primero entrenar_secuenciales.py")

    df = pd.read_csv(REPORT_PATH)
    print("Reporte leído:")
    print(df)

    # barras
    plot_bar_metric(df, "f1_test", "F1 (test) modelos secuenciales", "f1_test_secuenciales.png")
    plot_bar_metric(df, "prec_test", "Precisión (test) modelos secuenciales", "precision_test_secuenciales.png")
    plot_bar_metric(df, "rec_test", "Recall (test) modelos secuenciales", "recall_test_secuenciales.png")
    plot_bar_metric(df, "acc_test", "Accuracy (test) modelos secuenciales", "accuracy_test_secuenciales.png")

    # matrices
    for _, row in df.iterrows():
        modelo = row["modelo"]
        cm_file = row["cm_path"]
        cm_path = os.path.join(REPORT_DIR, cm_file)
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

        # Curvas de entrenamiento por época
    plot_training_curves("history_lstm.npy", "lstm_w2v")
    plot_training_curves("history_bilstm.npy", "bilstm_w2v")
    
if __name__ == "__main__":
    main()
