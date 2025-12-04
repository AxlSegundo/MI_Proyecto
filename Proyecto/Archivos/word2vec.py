import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from Archivos.preprocesamiento import preprocess_spanish



CSV_PATH    = "Proyecto/Data/fakes1000.csv"
TEXT_COL    = "Text"
OUTPUT_DIR  = "Proyecto/Data/Salidas/"


VEC_SIZE   = 700
WINDOW     = 5
MIN_CNT    = 2
EPOCHS     = 10
SG         = 1       
NEG        = 10
SEED       = 42
N_WORKERS  = 4



def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def doc_embedding_mean(tokens, w2v: Word2Vec):

    vecs = []
    for w in tokens:
        if w in w2v.wv:
            vecs.append(w2v.wv[w])
    if not vecs:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)



def main():
    ensure_output_dir(OUTPUT_DIR)


    df = pd.read_csv(CSV_PATH)
    texts = df[TEXT_COL].astype(str).tolist()
    tokens_all = [preprocess_spanish(t) for t in texts]


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


    X_w2v_mean = np.vstack([
        doc_embedding_mean(t, w2v) for t in tqdm(tokens_all, desc="W2V mean")
    ])


    out_path = os.path.join(OUTPUT_DIR, "w2v_mean.npy")
    np.save(out_path, X_w2v_mean)
    print("Embeddings W2V (promedio) guardados en:", out_path, "| Shape:", X_w2v_mean.shape)


if __name__ == "__main__":
    main()
