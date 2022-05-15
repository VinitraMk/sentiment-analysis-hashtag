from enum import Enum

class ValidationMethod(str, Enum):

    NORMAL_SPLIT = 'NORMAL_SPLIT'
    STRATIFIED_K_FOLD = 'STRATIFIED_K_FOLD'
    K_FOLD = 'K_FOLD'