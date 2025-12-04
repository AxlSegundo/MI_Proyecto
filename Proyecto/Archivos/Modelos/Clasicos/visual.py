import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.model_selection import learning_curve
from Archivos.loader import load_tfidf_and_labels



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

def plot_learning_curve_model(model_filename, model_name, scoring="f1"):
    """
    Genera la curva de aprendizaje (score train y validación) para un modelo clásico.
    """
    print(f"\n=== Curva de aprendizaje para {model_name} ({model_filename}) ===")

    # Cargar datos completos (X, y)
    X, y = load_tfidf_and_labels()

    # Cargar el modelo entrenado (mejor estimador guardado)
    model_path = os.path.join(CLASICOS_DIR, model_filename)
    if not os.path.exists(model_path):
        print(f"No se encontró el modelo {model_path}")
        return

    est = load(model_path)

    # learning_curve clonará el estimador internamente, 
    # así que da igual que ya esté entrenado
    train_sizes, train_scores, val_scores = learning_curve(
        est,
        X,
        y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    # Promedios sobre las folds
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    # Graficar
    plt.figure()
    plt.plot(train_sizes, train_mean, marker="o", label="Train")
    plt.fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.2)

    plt.plot(train_sizes, val_mean, marker="s", label="Validación")
    plt.fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.2)

    plt.title(f"Curva de aprendizaje ({scoring}) - {model_name}")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel(scoring)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(GRAF_DIR, f"learning_curve_{model_name}_{scoring}.png")
    plt.savefig(out_path)
    plt.close()
    print("Guardado:", out_path)


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

        # Curvas de aprendizaje para cada modelo clásico
    # Asegúrate de que estos nombres coinciden con los que guardas en train.py
    modelos_joblib = [
        ("svm_tfidf.joblib", "svm_tfidf"),
        ("logreg_tfidf.joblib", "logreg_tfidf"),
        ("rf_tfidf.joblib", "rf_tfidf"),
    ]

    for fname, nombre in modelos_joblib:
        plot_learning_curve_model(fname, nombre, scoring="f1")

    
if __name__ == "__main__":
    main()
