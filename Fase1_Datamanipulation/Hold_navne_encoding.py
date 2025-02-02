import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Læs kun relevante kolonner fra Excel-arket
columns_to_use = ["HomeTeam", "AwayTeam"]
matches = pd.read_excel("engelske_kampe_scrapped.xlsx", usecols=columns_to_use)

# Encode holdnavne som heltal
encoder = LabelEncoder()
home_team_encoded = encoder.fit_transform(matches["HomeTeam"])
away_team_encoded = encoder.transform(matches["AwayTeam"])

# Opret en DataFrame med de to lister
encoded_teams_df = pd.DataFrame({
    'Home Team Encoded': home_team_encoded,
    'Away Team Encoded': away_team_encoded
})

# Gem DataFrame som et Excel-ark
encoded_teams_df.to_excel("team_names_encoded.xlsx", index=False)