import pandas as pd
import numpy as np
import warnings  # Import Warnings to suppress unnecessary warnings
import shap
import matplotlib.pyplot as plt

# Suppress warning messages
warnings.filterwarnings("ignore")

# Læs kun første række af CSV-filen for at få kolonnenavne
headers = pd.read_csv("Fase0_raw_data/Bundesliga/06-07.csv", nrows=0).columns[:49]
print(headers)

# Read the dataset into a Pandas DataFrame
df4 = pd.read_csv("Fase0_raw_data/Bundesliga/06-07.csv",usecols=range(49)).to_numpy()
df5 = pd.read_csv("Fase0_raw_data/Bundesliga/07-08.csv",usecols=range(49)).to_numpy()
df6 = pd.read_csv("Fase0_raw_data/Bundesliga/08-09.csv",usecols=range(49)).to_numpy()
df7 = pd.read_csv("Fase0_raw_data/Bundesliga/09-10.csv",usecols=range(49)).to_numpy()
df8 = pd.read_csv("Fase0_raw_data/Bundesliga/10-11.csv",usecols=range(49)).to_numpy()
df9 = pd.read_csv("Fase0_raw_data/Bundesliga/11-12.csv",usecols=range(49)).to_numpy()
df10 = pd.read_csv("Fase0_raw_data/Bundesliga/12-13.csv",usecols=range(49)).to_numpy()
df11 = pd.read_csv("Fase0_raw_data/Bundesliga/13-14.csv",usecols=range(49)).to_numpy()
df12 = pd.read_csv("Fase0_raw_data/Bundesliga/14-15.csv",usecols=range(49)).to_numpy()
df13 = pd.read_csv("Fase0_raw_data/Bundesliga/15-16.csv",usecols=range(49)).to_numpy()
df14 = pd.read_csv("Fase0_raw_data/Bundesliga/16-17.csv",usecols=range(49)).to_numpy()
df15 = pd.read_csv("Fase0_raw_data/Bundesliga/17-18.csv",usecols=range(49)).to_numpy()
df16 = pd.read_csv("Fase0_raw_data/Bundesliga/18-19.csv",usecols=range(49)).to_numpy()
# Fjern kolonnen "Time" fra df17 til df21
df17 = pd.read_csv("Fase0_raw_data/Bundesliga/19-20.csv", usecols=range(50)).drop(columns=["Time"]).to_numpy()
df18 = pd.read_csv("Fase0_raw_data/Bundesliga/20-21.csv", usecols=range(50)).drop(columns=["Time"]).to_numpy()
df19 = pd.read_csv("Fase0_raw_data/Bundesliga/21-22.csv", usecols=range(50)).drop(columns=["Time"]).to_numpy()
df20 = pd.read_csv("Fase0_raw_data/Bundesliga/22-23.csv", usecols=range(50)).drop(columns=["Time"]).to_numpy()
df21 = pd.read_csv("Fase0_raw_data/Bundesliga/23-24.csv", usecols=range(50)).drop(columns=["Time"]).to_numpy()
df22 = pd.read_csv("Fase0_raw_data/Bundesliga/24-25.csv", usecols=range(50)).drop(columns=["Time"]).to_numpy()

stacked_array = np.vstack((df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22))

print(stacked_array)
print(stacked_array.shape)

result = pd.DataFrame(stacked_array)
result.columns = headers
result.to_excel("Fase1_Datamanipulation/Bundesliga/tyske_kampe_raw.xlsx", index=False)