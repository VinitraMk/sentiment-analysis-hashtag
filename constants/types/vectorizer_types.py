from enum import Enum

class VectorizerTypes(str, Enum):
    TFIDF = 'TFIDF'
    WC = 'WC'
    