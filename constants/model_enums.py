import enum

class Model(str, enum.Enum):
    SVM = 'svm'
    XGB = 'xgb'
    CTB = 'ctb'
    DECISION_TREE = 'decision_tree'
    RFA = 'rfa'
    ENSEMBLER = 'ensembler'
    ANN = 'ann'
    RNN = 'rnn'
    CNN = 'cnn'