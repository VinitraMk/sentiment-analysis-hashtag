from enum import Enum

class SamplingMethod(str, Enum):
    SMOTE_SAMPLING = 'SMOTE_SAMPLING'
    RANDOM_OVERSAMPLING = 'RANDOM_OVERSAMPLING'
    RANDOM_UNDERSAMPLING = 'RANDOM_UNDERSAMPLING'
    HYBRID_SAMPLING = 'HYBRID_SAMPLING'
    WORD_AUGMENTER = 'WORD_AUGMENTER'