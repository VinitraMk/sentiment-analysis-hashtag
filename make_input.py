import pandas as pd

train = pd.read_csv('./input/train.csv', encoding='ISO-8859-1', error_bad_lines=False)
test = pd.read_csv('./input/test.csv', encoding = 'ISO-8859-1')
data = pd.concat([train, test])
data.to_csv('./input/data.csv', index = False)