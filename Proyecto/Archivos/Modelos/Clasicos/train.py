import os
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from Archivos.loader import load_tfidf_and_labels
from Archivos.Modelos.Clasicos.busquedas import (
    search_svm_linear,
    search_logreg,
    search_rf,
)

OUT_DIR = os.path.join("Proyecto", "Data", "Salidas", "modelos", "clasicos")
REPORTS_DIR = os.path.join("Proyecto", "Data", "Salidas", "reportes_clasicos")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_metrics(model, X_train, y_train, X_test, y_test, name: str):

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_tr = accuracy_score(y_train, y_pred_train)
    prec_tr = precision_score(y_train, y_pred_train, zero_division=0)
    rec_tr = recall_score(y_train, y_pred_train, zero_division=0)
    f1_tr = f1_score(y_train, y_pred_train, zero_division=0)


    acc_te = accuracy_score(y_test, y_pred_test)
    prec_te = precision_score(y_test, y_pred_test, zero_division=0)
    rec_te = recall_score(y_test, y_pred_test, zero_division=0)
    f1_te = f1_score(y_test, y_pred_test, zero_division=0)


    cm = confusion_matrix(y_test, y_pred_test)
    cm_path = os.path.join(REPORTS_DIR, f"cm_{name}.npy")
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
        "cm_path": os.path.basename(cm_path),
    }


def main():
    X, y = load_tfidf_and_labels()
    print("X shape:", X.shape, "| y shape:", y.shape)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = []


    print("\n=== Búsqueda SVM (LinearSVC) ===")
    svm_search = search_svm_linear(X_train, y_train)
    print("Mejores params SVM:", svm_search.best_params_)
    print("Mejor F1 CV SVM:", svm_search.best_score_)
    best_svm = svm_search.best_estimator_
    dump(best_svm, os.path.join(OUT_DIR, "svm_tfidf.joblib"))

    svm_metrics = compute_metrics(
        best_svm, X_train, y_train, X_test, y_test, name="svm_tfidf"
    )
    results.append(svm_metrics)


    print("\n=== Búsqueda Regresión Logística ===")
    lr_search = search_logreg(X_train, y_train)
    print("Mejores params LR:", lr_search.best_params_)
    print("Mejor F1 CV LR:", lr_search.best_score_)
    best_lr = lr_search.best_estimator_
    dump(best_lr, os.path.join(OUT_DIR, "logreg_tfidf.joblib"))

    lr_metrics = compute_metrics(
        best_lr, X_train, y_train, X_test, y_test, name="logreg_tfidf"
    )
    results.append(lr_metrics)


    print("\n=== Búsqueda Random Forest ===")
    rf_search = search_rf(X_train, y_train)
    print("Mejores params RF:", rf_search.best_params_)
    print("Mejor F1 CV RF:", rf_search.best_score_)
    best_rf = rf_search.best_estimator_
    dump(best_rf, os.path.join(OUT_DIR, "rf_tfidf.joblib"))

    rf_metrics = compute_metrics(
        best_rf, X_train, y_train, X_test, y_test, name="rf_tfidf"
    )
    results.append(rf_metrics)


    df_report = pd.DataFrame(results)
    report_path = os.path.join(REPORTS_DIR, "report_clasicos.csv")
    df_report.to_csv(report_path, index=False, encoding="utf-8")
    print("\nReporte guardado en:", report_path)
    print(df_report)

    print("\nModelos clásicos entrenados y guardados en:", OUT_DIR)
    print("Matrices de confusión guardadas en:", REPORTS_DIR)


if __name__ == "__main__":
    main()
