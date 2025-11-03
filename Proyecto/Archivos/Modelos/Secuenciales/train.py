import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


BASE_DATA = os.path.join("Proyecto", "Data")
SALIDAS   = os.path.join(BASE_DATA, "Salidas")

W2V_PATH  = os.path.join(SALIDAS, "w2v_tfidf.npy")
CSV_PATH  = os.path.join(BASE_DATA, "fakes1000.csv")

MODELOS_DIR   = os.path.join(SALIDAS, "modelos", "secuenciales")
REPORTES_DIR  = os.path.join(SALIDAS, "reportes_secuenciales")

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(REPORTES_DIR, exist_ok=True)



def load_w2v_and_labels():
    X = np.load(W2V_PATH)
    df = pd.read_csv(CSV_PATH)
    y = df["class"].astype(int).values
    return X, y


def build_lstm(input_dim: int):
    model = Sequential([

        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Reshape((1, input_dim)),
        LSTM(128, return_sequences=False),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )
    return model


def build_bilstm(input_dim: int):
    model = Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((1, input_dim)),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )
    return model


def compute_metrics_keras(model, X_train, y_train, X_test, y_test, name):
    # predicciones
    y_pred_train = (model.predict(X_train) > 0.5).astype("int32").ravel()
    y_pred_test  = (model.predict(X_test)  > 0.5).astype("int32").ravel()

    # métricas train
    acc_tr  = accuracy_score(y_train, y_pred_train)
    prec_tr = precision_score(y_train, y_pred_train, zero_division=0)
    rec_tr  = recall_score(y_train, y_pred_train, zero_division=0)
    f1_tr   = f1_score(y_train, y_pred_train, zero_division=0)

    # métricas test
    acc_te  = accuracy_score(y_test, y_pred_test)
    prec_te = precision_score(y_test, y_pred_test, zero_division=0)
    rec_te  = recall_score(y_test, y_pred_test, zero_division=0)
    f1_te   = f1_score(y_test, y_pred_test, zero_division=0)

    cm = confusion_matrix(y_test, y_pred_test)
    cm_path = os.path.join(REPORTES_DIR, f"cm_{name}.npy")
    np.save(cm_path, cm)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = None

    return {
        "modelo": name,
        "acc_train": acc_tr,
        "prec_train": prec_tr,
        "rec_train": rec_tr,
        "f1_train": f1_tr,
        "acc_test": acc_te,
        "prec_test": prec_te,
        "rec_test": rec_te,
        "f1_test": f1_te,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cm_path": f"cm_{name}.npy"
    }


def main():

    X, y = load_w2v_and_labels()
    print("X shape:", X.shape, "| y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    input_dim = X_train.shape[1]


    es = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    results = []


    print("\n=== Entrenando LSTM sobre W2V ponderado ===")
    lstm_model = build_lstm(input_dim)
    lstm_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )
    lstm_path = os.path.join(MODELOS_DIR, "lstm_w2v.keras")
    lstm_model.save(lstm_path)
    print("LSTM guardado en:", lstm_path)

    lstm_metrics = compute_metrics_keras(
        lstm_model, X_train, y_train, X_test, y_test, name="lstm_w2v"
    )
    results.append(lstm_metrics)


    print("\n=== Entrenando BiLSTM sobre W2V ponderado ===")
    bilstm_model = build_bilstm(input_dim)
    bilstm_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )
    bilstm_path = os.path.join(MODELOS_DIR, "bilstm_w2v.keras")
    bilstm_model.save(bilstm_path)
    print("BiLSTM guardado en:", bilstm_path)

    bilstm_metrics = compute_metrics_keras(
        bilstm_model, X_train, y_train, X_test, y_test, name="bilstm_w2v"
    )
    results.append(bilstm_metrics)


    df_report = pd.DataFrame(results)
    report_path = os.path.join(REPORTES_DIR, "report_secuenciales.csv")
    df_report.to_csv(report_path, index=False, encoding="utf-8")
    print("\nReporte de modelos secuenciales guardado en:", report_path)
    print(df_report)


if __name__ == "__main__":
    main()
