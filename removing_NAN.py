import pandas as pd

da = pd.read_csv('D:/Project_and_Case_Study_1/url_dataset.csv',
                 encoding ='latin1',
                 low_memory = False)
print(da.shape)

df = da.dropna(axis = 0, how = 'any')
df.to_csv('D:/Project_and_Case_Study_1/NAN_removed_df.csv', index = False)