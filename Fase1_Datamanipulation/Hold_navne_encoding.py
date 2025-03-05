import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Funktion til at konvertere holdnavne til binære værdier

def encode_team_name_binary(df, team_column, max_length=64):
    """
    Konverterer holdnavne til binære værdier direkte fra ASCII og deler dem i kolonner.

    :param df: DataFrame med en kolonne af holdnavne
    :param team_column: Navnet på kolonnen med holdnavne
    :param max_length: Maksimum binære længde (kan trimmes eller pad’es)
    :return: DataFrame med binære features
    """
    def to_binary_array(team):
        bin_str = ''.join(format(ord(char), '08b') for char in team)  # Konverter til binær
        bin_str = bin_str[:max_length].ljust(max_length, '0')  # Trim/pad til fast længde
        return [int(b) for b in bin_str]  # Liste af 0'er og 1'er
    
    # Anvend binær konvertering på alle hold
    binary_encoded = df[team_column].apply(to_binary_array).tolist()
    
    # Opret en DataFrame med de binære værdier som separate kolonner
    binary_columns = [f"{team_column}_bit_{i}" for i in range(max_length)]
    binary_df = pd.DataFrame(binary_encoded, columns=binary_columns)

    return binary_df

# Indlæs relevante kolonner fra Excel-arket
columns_to_use = ["HomeTeam", "AwayTeam"]
matches = pd.read_excel("Fase1_Datamanipulation/engelske_kampe_scrapped.xlsx", usecols=columns_to_use)

# Konverter hjemme- og udehold til binær repræsentation
binary_home = encode_team_name_binary(matches, "HomeTeam")
binary_away = encode_team_name_binary(matches, "AwayTeam")

# Kombiner de to DataFrames side om side
combined_df = pd.concat([binary_home, binary_away], axis=1)

# Gem resultatet i et Excel-ark
combined_df.to_excel("Fase1_Datamanipulation/binary_encoded_teams.xlsx", index=False)

"""
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
"""