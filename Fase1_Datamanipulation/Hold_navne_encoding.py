import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Indlæs relevante kolonner fra Excel-arket
columns_to_use = ["HomeTeam", "AwayTeam"]
matches = pd.read_excel("Fase1_Datamanipulation/engelske_kampe_scrapped.xlsx", usecols=columns_to_use)

# Initialiser one-hot-encoderen
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # sparse_output=False sikrer en tæt matrix

# One-hot-encode hjemmeholdene
home_encoded = encoder.fit_transform(matches[["HomeTeam"]])
away_encoded = encoder.fit_transform(matches[["AwayTeam"]])

# Konverter de encodede arrays til DataFrames
home_encoded_df = pd.DataFrame(home_encoded)
away_encoded_df = pd.DataFrame(away_encoded)

# Kombiner de to DataFrames side om side
combined_df = pd.concat([home_encoded_df, away_encoded_df], axis=1)

# Gem resultatet i et Excel-ark
combined_df.to_excel("Fase1_Datamanipulation/one_hot_encoded_teams.xlsx", index=False)