from enum import Enum

class EnsembleMethod(str, Enum):
    VOTING = 'voting'
    STACKING = 'stacking'