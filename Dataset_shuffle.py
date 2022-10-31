import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('D:/Project_and_Case_Study_1/Final.csv')
df = shuffle(df)
df.to_csv('D:/Project_and_Case_Study_1/Final_shuffle.csv')