import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from Archivos.preprocesamiento import preprocess_spanish


CSV_PATH   = "Proyecto/Data/fakes1000.csv"
TEXT_COL   = "Text"
OUTPUT_DIR = "Proyecto/Data/Salidas"
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(CSV_PATH)
texts = df[TEXT_COL].astype(str).tolist()


texts = [" ".join(preprocess_spanish(t)) for t in texts]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


@torch.no_grad()
def get_beto_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    ).to(device)

    outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


embeddings = []
for text in tqdm(texts, desc="Generando embeddings BETO"):
    emb = get_beto_embedding(text)
    embeddings.append(emb)

X_beto = np.vstack(embeddings)
print("BETO embeddings shape:", X_beto.shape)


np.save(os.path.join(OUTPUT_DIR, "beto_embeddings.npy"), X_beto)
print("Embeddings BETO guardados en:", os.path.join(OUTPUT_DIR, "beto_embeddings.npy"))
