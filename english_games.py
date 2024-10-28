import pandas as pd
import numpy as np
import warnings  # Import Warnings to suppress unnecessary warnings
import shap
import matplotlib.pyplot as plt

# Suppress warning messages
warnings.filterwarnings("ignore")

# Read the dataset into a Pandas DataFrame
df1 = pd.read_csv("raw_data/03-04.csv")

df1.to_excel("engelske_kampe.xlsx", index=False)