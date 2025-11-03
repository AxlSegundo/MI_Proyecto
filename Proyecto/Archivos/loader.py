import os
import numpy as np
import pandas as pd
from scipy import sparse

BASE_DATA = os.path.join("Proyecto", "Data")
SALIDAS   = os.path.join(BASE_DATA, "Salidas")
CSV_PATH  = os.path.join(BASE_DATA, "fakes1000.csv")
TFIDF_PATH = os.path.join(SALIDAS, "tfidf_X.npz")

def load_tfidf_and_labels():

    X = sparse.load_npz(TFIDF_PATH)
    df = pd.read_csv(CSV_PATH)
    y = df["class"].astype(int).values
    return X, y
