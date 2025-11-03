import pandas as pd
from joblib import dump
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from Archivos.preprocesamiento import preprocess_spanish


def identity(x):
    return x

def passthrough(x):
    return x

CSV_PATH = "Proyecto/Data/fakes1000.csv"
TEXT_COL = "Text"

df = pd.read_csv(CSV_PATH)
texts = df[TEXT_COL].astype(str)


tokens = texts.apply(preprocess_spanish)


vectorizer = TfidfVectorizer(
    tokenizer=identity,        
    preprocessor=passthrough,
    token_pattern=None,        
    lowercase=False,
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
    min_df=2
)

X_tfidf = vectorizer.fit_transform(tokens)


dump(vectorizer, "Proyecto/Data/Salidas/tfidf_vectorizer.joblib")
sparse.save_npz("Proyecto/Data/Salidas/tfidf_X.npz", X_tfidf)

print("Proceso terminado: ")
print(" - Vectorizador: tfidf_vectorizer.joblib")
print(" - Matriz:       tfidf_X.npz")
print(" - Shape:", X_tfidf.shape)
