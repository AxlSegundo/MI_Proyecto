import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    set_seed,
)


BASE_DATA   = os.path.join("Proyecto", "Data")
SALIDAS     = os.path.join(BASE_DATA, "Salidas")
CSV_PATH    = os.path.join(BASE_DATA, "fakes1000.csv")

REPORT_DIR  = os.path.join(SALIDAS, "reportes_transformers")
MODELOS_DIR = os.path.join(SALIDAS, "modelos", "Transformers")
GRAF_DIR    = os.path.join(MODELOS_DIR, "graficas")  

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(GRAF_DIR, exist_ok=True)

MODEL_NAME   = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH   = 256
EPOCHS       = 3
SEED         = 42
LRS          = [5e-5, 3e-5, 2e-5]    
BATCH_SIZES  = [8, 16]               

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)


@dataclass
class TextDataset(Dataset):
    encodings: Dict[str, torch.Tensor]
    labels: List[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_texts_and_labels(csv_path: str, text_col: str = "Text", label_col: str = "class"):
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels

def tokenize_texts(tokenizer, texts: List[str], max_length: int):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

def compute_clf_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def predict_with_trainer(trainer: "Trainer", dataset: Dataset):
    preds_output = trainer.predict(dataset)
    logits = preds_output.predictions
    y_pred = logits.argmax(axis=-1)
    return y_pred


def make_training_args(output_dir, lr, bs, epochs, seed):

    from transformers import TrainingArguments
    try:

        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=lr,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            num_train_epochs=epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            logging_steps=50,
            report_to=[], 
            seed=seed,
        )
    except TypeError:
        try:

            return TrainingArguments(
                output_dir=output_dir,
                learning_rate=lr,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                num_train_epochs=epochs,
                logging_steps=50,
                seed=seed,
            )
        except TypeError:

            return TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                num_train_epochs=epochs,
                seed=seed,
            )


def train_and_select_best(tokenizer, X_train, y_train, X_val, y_val):
    enc_train = tokenize_texts(tokenizer, X_train, MAX_LENGTH)
    enc_val   = tokenize_texts(tokenizer, X_val, MAX_LENGTH)

    ds_train = TextDataset(enc_train, y_train)
    ds_val   = TextDataset(enc_val, y_val)

    best_f1 = -1.0
    best_cfg = None
    best_model_dir = None

    for lr in LRS:
        for bs in BATCH_SIZES:
            run_name = f"beto_lr{lr}_bs{bs}"
            output_dir = os.path.join(MODELOS_DIR, run_name)
            os.makedirs(output_dir, exist_ok=True)

            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
            model.to(device)

            args = make_training_args(output_dir, lr, bs, EPOCHS, SEED)


            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds_train,
                eval_dataset=ds_val,
                tokenizer=tokenizer,
            )

            trainer.train()


            try:
                eval_res = trainer.evaluate(eval_dataset=ds_val)
                f1_val = float(
                    eval_res.get("eval_f1") or
                    eval_res.get("f1") or
                    0.0
                )
            except Exception:

                y_val_pred = predict_with_trainer(trainer, ds_val)
                f1_val = f1_score(y_val, y_val_pred)

            print(f"[{run_name}] F1 val = {f1_val:.4f}")

            if f1_val > best_f1:
                best_f1 = f1_val
                best_cfg = {"lr": lr, "batch_size": bs}
                best_model_dir = output_dir

    print("Mejor config:", best_cfg, " | Mejor F1 val:", best_f1)
    return best_cfg, best_model_dir


def main():

    texts, labels = load_texts_and_labels(CSV_PATH, "Text", "class")

    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=SEED, stratify=y_temp
    )


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    best_cfg, best_model_dir = train_and_select_best(tokenizer, X_train, y_train, X_val, y_val)


    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_dir, num_labels=2).to(device)

    enc_train = tokenize_texts(tokenizer, X_train, MAX_LENGTH)
    enc_test  = tokenize_texts(tokenizer, X_test,  MAX_LENGTH)
    ds_train  = TextDataset(enc_train, y_train)
    ds_test   = TextDataset(enc_test,  y_test)

    pred_trainer = Trainer(model=best_model, tokenizer=tokenizer)
    y_pred_train = predict_with_trainer(pred_trainer, ds_train)
    y_pred_test  = predict_with_trainer(pred_trainer, ds_test)


    m_train = compute_clf_metrics(y_train, y_pred_train)
    m_test  = compute_clf_metrics(y_test,  y_pred_test)

    cm = confusion_matrix(y_test, y_pred_test)
    cm_name = "cm_beto.npy"
    np.save(os.path.join(REPORT_DIR, cm_name), cm)
    tn, fp, fn, tp = cm.ravel()


    row = {
        "modelo": "beto_finetuned",
        "acc_train": m_train["accuracy"],
        "prec_train": m_train["precision"],
        "rec_train": m_train["recall"],
        "f1_train": m_train["f1"],
        "acc_test": m_test["accuracy"],
        "prec_test": m_test["precision"],
        "rec_test": m_test["recall"],
        "f1_test": m_test["f1"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "cm_path": cm_name,
        "best_lr": best_cfg["lr"],
        "best_batch_size": best_cfg["batch_size"],
        "model_dir": os.path.relpath(best_model_dir, start=SALIDAS),
    }

    report_path = os.path.join(REPORT_DIR, "report_transformers.csv")
    pd.DataFrame([row]).to_csv(report_path, index=False, encoding="utf-8")

    print("\nReporte guardado en:", report_path)
    print(pd.DataFrame([row]))
    print("Mejor modelo en:", best_model_dir)

if __name__ == "__main__":
    main()
