# word2vec_weighted_only.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load, dump
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from Archivos.preprocesamiento import preprocess_spanish


CSV_PATH   = "Proyecto/Data/fakes1000.csv"
TEXT_COL   = "Text"
OUTPUT_DIR = "Proyecto/Data/Salidas/"


TFIDF_FOR_W2V_PATH = os.path.join(OUTPUT_DIR, "tfidf_for_w2v_vectorizer.joblib")


VEC_SIZE = 700
WINDOW   = 5
MIN_CNT  = 2
EPOCHS   = 10
SG       = 1       
NEG      = 10
SEED     = 42
N_WORKERS = 4


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def build_or_load_tfidf_vectorizer(csv_path: str, text_col: str, out_path: str):

    if os.path.exists(out_path):
        vec = load(out_path)
        print(f"TF-IDF cargado: {out_path}")
        return vec

    print("No existe TF-IDF para ponderado. Se creará en:", out_path)
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str)


    tokens = texts.apply(preprocess_spanish)
    texts_preproc = tokens.apply(" ".join)

    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
        lowercase=False  
    )
    vec.fit(texts_preproc)
    dump(vec, out_path)
    print(f"TF-IDF creado y guardado en: {out_path}")
    return vec

def doc_embedding_tfidf(tokens, w2v: Word2Vec, idf_map):

    vecs, weights = [], []
    for w in tokens:
        if w in w2v.wv:
            wt = float(idf_map.get(w, 1.0))
            vecs.append(w2v.wv[w] * wt)
            weights.append(wt)
    if not vecs:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return (np.sum(vecs, axis=0) / (np.sum(weights) + 1e-8)).astype(np.float32)

def main():
    ensure_output_dir(OUTPUT_DIR)

    df = pd.read_csv(CSV_PATH)
    texts = df[TEXT_COL].astype(str).tolist()
    tokens_all = [preprocess_spanish(t) for t in texts]

    vec = build_or_load_tfidf_vectorizer(CSV_PATH, TEXT_COL, TFIDF_FOR_W2V_PATH)
    idf_map = dict(zip(vec.get_feature_names_out(), vec.idf_))

    w2v = Word2Vec(
        sentences=tokens_all,
        vector_size=VEC_SIZE,
        window=WINDOW,
        min_count=MIN_CNT,
        workers=N_WORKERS,
        sg=SG,
        negative=NEG,
        epochs=EPOCHS,
        seed=SEED
    )
    w2v_path = os.path.join(OUTPUT_DIR, "word2vec_model.bin")
    w2v.save(w2v_path)
    print("Modelo W2V guardado en:", w2v_path)

    X_tfidf = np.vstack([doc_embedding_tfidf(t, w2v, idf_map) for t in tqdm(tokens_all, desc="W2V tf-idf")])
    out_path = os.path.join(OUTPUT_DIR, "w2v_tfidf.npy")
    np.save(out_path, X_tfidf)
    print("Embeddings W2V ponderados guardados en:", out_path, "| Shape:", X_tfidf.shape)

if __name__ == "__main__":
    main()
