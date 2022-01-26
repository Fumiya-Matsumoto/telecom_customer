import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/input/Telecom_customer churn.csv')

# トレーニングデータ,テストデータの分割
train, test = train_test_split(df, test_size=0.2, random_state=0)

train.to_csv('./data/input/train.csv')
test.to_csv('./data/input/test.csv')