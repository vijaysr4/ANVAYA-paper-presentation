# Importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('D:/Project_and_Case_Study_1/Final_shuffle.csv')
df = df.drop(['Domain'], axis = 1)
df = df.drop(['num'], axis = 1)

# Correlation Heatmap
plt.figure(figsize=(34, 12))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
