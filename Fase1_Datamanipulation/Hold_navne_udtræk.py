import pandas as pd
import numpy as np
import warnings  # Import Warnings to suppress unnecessary warnings
import shap
import matplotlib.pyplot as plt

# Læs kolonnen med holdnavne fra Excel-filen
HomeTeams = pd.read_excel("Fase1_Datamanipulation/engelske_kampe_scrapped.xlsx", usecols="B", skiprows=0, nrows=7981).to_numpy()

# Find unikke værdier og sorter dem alfabetisk
unique_teams = np.unique(HomeTeams)

# Print de unikke holdnavne
print(unique_teams)

# Print antallet af hold
print(f"Antallet af unikke hold er: {len(unique_teams)}")