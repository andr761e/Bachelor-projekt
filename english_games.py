import pandas as pd
import numpy as np
import warnings  # Import Warnings to suppress unnecessary warnings
import shap
import matplotlib.pyplot as plt

# Suppress warning messages
warnings.filterwarnings("ignore")

# Read the dataset into a Pandas DataFrame
df = pd.read_csv("E0.csv")

df.to_excel("engelske_kampe.xlsx", index=False)