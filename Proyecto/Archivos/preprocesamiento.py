import re
import unicodedata
from typing import List, Iterable, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer

try:
    _ = stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords")
    _ = stopwords.words("spanish")

SPANISH_STOPWORDS = set(_)
STEMMER = SpanishStemmer()

_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_TAG_RE   = re.compile(r"[@#]\w+")
_NUM_RE   = re.compile(r"\d+")
_WS_RE    = re.compile(r"\s+")

def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )

def basic_normalize(text: str, *, keep_accents: bool = False) -> str:
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _NUM_RE.sub(" <num> ", text)
    text = re.sub(r"[^\w\sáéíóúñü]", " ", text, flags=re.IGNORECASE)
    text = _WS_RE.sub(" ", text).strip()
    if not keep_accents:
        text = strip_accents(text)
    return text

def tokenize(text: str) -> List[str]:
    return text.split()

def remove_stopwords(tokens: Iterable[str], extra_stops: Optional[Iterable[str]] = None) -> List[str]:
    sw = SPANISH_STOPWORDS.copy()
    if extra_stops:
        sw.update(map(str.lower, extra_stops))
    return [t for t in tokens if t not in sw and len(t) > 1]

def stem_tokens(tokens: Iterable[str]) -> List[str]:
    return [STEMMER.stem(t) for t in tokens]

def preprocess_spanish(text: str) -> List[str]:
    norm = basic_normalize(text)
    toks = tokenize(norm)
    toks = remove_stopwords(toks)
    toks = stem_tokens(toks)
    return toks
