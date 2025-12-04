import pandas as pd
import matplotlib.pyplot as plt
import os

def graficar_curvas_modelo(path_csv, save_dir="Proyecto/Data/Salidas/graficas_comparativas"):

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(path_csv)


    modelo_col = [c for c in df.columns if "modelo" in c.lower()]
    if not modelo_col:
        print("No se encontró una columna de modelo.")
        return
    modelo_col = modelo_col[0]

    metricas = ["acc", "prec", "rec", "f1"]
    for _, fila in df.iterrows():
        modelo = fila[modelo_col]
        valores_train, valores_test = [], []
        etiquetas = []


        for m in metricas:
            col_train = [c for c in df.columns if m in c and "train" in c]
            col_test  = [c for c in df.columns if m in c and "test"  in c]
            if col_train and col_test:
                valores_train.append(fila[col_train[0]])
                valores_test.append(fila[col_test[0]])
                etiquetas.append(m.upper())

        if not etiquetas:
            continue

        # Gráfico tipo "curva"
        plt.figure(figsize=(6, 4))
        plt.plot(etiquetas, valores_train, marker='o', label="Entrenamiento", color='royalblue')
        plt.plot(etiquetas, valores_test, marker='o', label="Test", color='darkorange')

        plt.title(f"Desempeño del modelo: {modelo}")
        plt.xlabel("Métrica")
        plt.ylabel("Valor")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Guardar
        filename = f"{modelo}_curvas.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()

    print(f"Gráficas guardadas en: {save_dir}")

clasicos = "Proyecto/Data/Salidas/reportes_clasicos/report_clasicos.csv"
graficar_curvas_modelo(clasicos)
secuenciales = "Proyecto/Data/Salidas/reportes_secuenciales/report_secuenciales.csv"
graficar_curvas_modelo(secuenciales)   
transformers = "Proyecto/Data/Salidas/reportes_transformers/report_transformers.csv"
graficar_curvas_modelo(transformers)