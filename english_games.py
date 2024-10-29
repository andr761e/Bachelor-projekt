import pandas as pd
import numpy as np
import warnings  # Import Warnings to suppress unnecessary warnings
import shap
import matplotlib.pyplot as plt

# Suppress warning messages
warnings.filterwarnings("ignore")

# Læs kun første række af CSV-filen for at få kolonnenavne
headers = pd.read_csv("raw_data/03-04.csv", nrows=0).columns[:57]

# Read the dataset into a Pandas DataFrame
df1 = pd.read_csv("raw_data/03-04.csv",usecols=range(57)).to_numpy()
df2 = pd.read_csv("raw_data/04-05.csv",usecols=range(57),encoding="utf-8").to_numpy()
df3 = pd.read_csv("raw_data/05-06.csv",usecols=range(57)).to_numpy()
df4 = pd.read_csv("raw_data/06-07.csv",usecols=range(57)).to_numpy()
df5 = pd.read_csv("raw_data/07-08.csv",usecols=range(57)).to_numpy()
df6 = pd.read_csv("raw_data/08-09.csv",usecols=range(57)).to_numpy()
df7 = pd.read_csv("raw_data/09-10.csv",usecols=range(57)).to_numpy()
df8 = pd.read_csv("raw_data/10-11.csv",usecols=range(57)).to_numpy()
df9 = pd.read_csv("raw_data/11-12.csv",usecols=range(57)).to_numpy()
df10 = pd.read_csv("raw_data/12-13.csv",usecols=range(57)).to_numpy()
df11 = pd.read_csv("raw_data/13-14.csv",usecols=range(57)).to_numpy()
df12 = pd.read_csv("raw_data/14-15.csv",usecols=range(57)).to_numpy()
df13 = pd.read_csv("raw_data/15-16.csv",usecols=range(57)).to_numpy()
df14 = pd.read_csv("raw_data/16-17.csv",usecols=range(57)).to_numpy()
df15 = pd.read_csv("raw_data/17-18.csv",usecols=range(57)).to_numpy()

stacked_array = np.vstack((df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15))

print(stacked_array)
print(stacked_array.shape)


pd.DataFrame(stacked_array,columns=headers).to_excel("engelske_kampe_raw.xlsx", index=False)