import pandas as pd
import numpy as np
import warnings  # Import Warnings to suppress unnecessary warnings
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Læs kun relevante kolonner fra Excel-arket
columns_to_use = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
                  "HTR", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
matches = pd.read_excel("engelske_kampe_scrapped.xlsx", usecols=columns_to_use)

# Encode holdnavne som heltal
encoder = LabelEncoder()
home_team_encoded = encoder.fit_transform(matches["HomeTeam"])
away_team_encoded = encoder.transform(matches["AwayTeam"])


print(home_team_encoded)
print(away_team_encoded)
print(len(home_team_encoded))