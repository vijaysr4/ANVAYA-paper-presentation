import pandas as pd
import wget

url = 'http://data.phishtank.com/data/online-valid.csv'

wget.download(url, 'D:\Project_and_Case_Study_1\Phishtank.csv')