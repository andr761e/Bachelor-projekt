import pandas as pd
import numpy as np
import warnings  # Import Warnings to suppress unnecessary warnings
import shap
import matplotlib.pyplot as plt

#Y DELEN
def normalize_odds_to_probabilities(odds_matrix):
    """
    Convert an odds matrix to probabilities by taking the inverse of each entry 
    and normalizing each row to sum to 1.

    Args:
        odds_matrix: numpy array of shape (n, k), where each row contains odds for k outcomes.

    Returns:
        probabilities: numpy array of shape (n, k), where each row contains normalized probabilities.
    """
    # Step 1: Take the inverse of each entry
    probabilities = 1 / odds_matrix
    
    # Step 2: Normalize each row to sum to 1
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    
    return probabilities


# Load only column "Q" and rows from Q2 to Q2641
HomeWinProb = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_kampe_scrapped.xlsx", usecols="AE", skiprows=0, nrows=5716)
DrawProb = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_kampe_scrapped.xlsx", usecols="AF", skiprows=0, nrows=5716)
AwayWinProb = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_kampe_scrapped.xlsx", usecols="AG", skiprows=0, nrows=5716)

#Concatenate data to matrix
y = np.hstack((HomeWinProb.to_numpy(), DrawProb.to_numpy(), AwayWinProb.to_numpy()))
Y = normalize_odds_to_probabilities(y)


#X DELEN
# Læs kun relevante kolonner fra Excel-arket
columns_to_use = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
                  "HTR", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
matches = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_kampe_scrapped.xlsx", usecols=columns_to_use)

#Indløs ELO data
HomeTeamElo = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_elo_data.xlsx", usecols="A", skiprows=0, nrows=5716)
AwayTeamElo = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_elo_data.xlsx", usecols="B", skiprows=0, nrows=5716)

#Indlæs One-Hot-Encoded data
HomeTeamEncoded = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_binary_encoded_teams.xlsx", usecols=range(0,64), skiprows=0, nrows=5716)
AwayTeamEncoded = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_binary_encoded_teams.xlsx", usecols=range(64,128), skiprows=0, nrows=5716)

# Resultattabel
form_stats = []

for i, match in matches.iterrows():
    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    # Find de seneste 5 kampe for hjemmeholdet
    home_matches = matches.loc[:i - 1]
    home_history = home_matches[(home_matches["HomeTeam"] == home_team) | (home_matches["AwayTeam"] == home_team)].tail(5)

    # Find de seneste 5 kampe for udeholdet
    away_matches = matches.loc[:i - 1]
    away_history = away_matches[(away_matches["HomeTeam"] == away_team) | (away_matches["AwayTeam"] == away_team)].tail(5)

    # Hvis der ikke er nok historik, markér som manglende
    if len(home_history) < 5 or len(away_history) < 5:
        form_stats.append([np.nan] * 16)
        continue

    # Beregn gennemsnit af modstandernes ELO for hjemme- og udehold
    home_opponents_elo = home_history.apply(lambda row: AwayTeamElo.iloc[row.name].item() if row["HomeTeam"] == home_team else HomeTeamElo.iloc[row.name].item(), axis=1).mean()
    away_opponents_elo = away_history.apply(lambda row: AwayTeamElo.iloc[row.name].item() if row["HomeTeam"] == away_team else HomeTeamElo.iloc[row.name].item(), axis=1).mean()

    # Funktion til at vægte statistik med modstandernes ELO
    def weighted_mean(history, team, opponents_elo):
        return history.apply(lambda row: (row["FTHG"] if row["HomeTeam"] == team else row["FTAG"]) * opponents_elo, axis=1).mean()

    # Beregn vægtede gennemsnit for hjemmeholdets form
    home_goals = weighted_mean(home_history, home_team, home_opponents_elo)
    home_points = home_history.apply(lambda row: (3 if (row["HomeTeam"] == home_team and row["FTR"] == "H") or (row["AwayTeam"] == home_team and row["FTR"] == "A") else 1 if row["FTR"] == "D" else 0) * home_opponents_elo, axis=1).mean()
    home_shots = home_history.apply(lambda row: (row["HS"] if row["HomeTeam"] == home_team else row["AS"]) * home_opponents_elo, axis=1).mean()
    home_shots_on_target = home_history.apply(lambda row: (row["HST"] if row["HomeTeam"] == home_team else row["AST"]) * home_opponents_elo, axis=1).mean()
    home_fouls = home_history.apply(lambda row: (row["HF"] if row["HomeTeam"] == home_team else row["AF"]) * home_opponents_elo, axis=1).mean()
    home_corners = home_history.apply(lambda row: (row["HC"] if row["HomeTeam"] == home_team else row["AC"]) * home_opponents_elo, axis=1).mean()
    home_yellow_cards = home_history.apply(lambda row: (row["HY"] if row["HomeTeam"] == home_team else row["AY"]) * home_opponents_elo, axis=1).mean()
    home_red_cards = home_history.apply(lambda row: (row["HR"] if row["HomeTeam"] == home_team else row["AR"]) * home_opponents_elo, axis=1).mean()

    # Beregn vægtede gennemsnit for udeholdets form
    away_goals = weighted_mean(away_history, away_team, away_opponents_elo)
    away_points = away_history.apply(lambda row: (3 if (row["HomeTeam"] == away_team and row["FTR"] == "H") or (row["AwayTeam"] == away_team and row["FTR"] == "A") else 1 if row["FTR"] == "D" else 0) * away_opponents_elo, axis=1).mean()
    away_shots = away_history.apply(lambda row: (row["HS"] if row["HomeTeam"] == away_team else row["AS"]) * away_opponents_elo, axis=1).mean()
    away_shots_on_target = away_history.apply(lambda row: (row["HST"] if row["HomeTeam"] == away_team else row["AST"]) * away_opponents_elo, axis=1).mean()
    away_fouls = away_history.apply(lambda row: (row["HF"] if row["HomeTeam"] == away_team else row["AF"]) * away_opponents_elo, axis=1).mean()
    away_corners = away_history.apply(lambda row: (row["HC"] if row["HomeTeam"] == away_team else row["AC"]) * away_opponents_elo, axis=1).mean()
    away_yellow_cards = away_history.apply(lambda row: (row["HY"] if row["HomeTeam"] == away_team else row["AY"]) * away_opponents_elo, axis=1).mean()
    away_red_cards = away_history.apply(lambda row: (row["HR"] if row["HomeTeam"] == away_team else row["AR"]) * away_opponents_elo, axis=1).mean()

    # Tilføj statistikker til resultatet
    form_stats.append([
        HomeTeamElo.iloc[i].item(), home_goals, home_points, home_shots, home_shots_on_target, home_fouls, home_corners, home_yellow_cards, home_red_cards,
        AwayTeamElo.iloc[i].item(), away_goals, away_points, away_shots, away_shots_on_target, away_fouls, away_corners, away_yellow_cards, away_red_cards
    ])

# Konverter resultatet til en DataFrame med kun de specifikke kolonner
columns = [
    "HomeTeamELO","HomeGoals5", "HomePoints5", "HomeShots5", "HomeShotsOnTarget5", "HomeFouls5",
    "HomeCorners5", "HomeYellowCards5", "HomeRedCards5",
    "AwayTeamELO", "AwayGoals5", "AwayPoints5", "AwayShots5", "AwayShotsOnTarget5", "AwayFouls5",
    "AwayCorners5", "AwayYellowCards5", "AwayRedCards5"
]
form_stats_df = pd.DataFrame(form_stats, columns=columns)

# Opret en kopi af din eksisterende DataFrame
final_df = form_stats_df.copy()

df1 = form_stats_df.iloc[:, :9]   # Første del (kolonner 1 til 37)
df2 = form_stats_df.iloc[:, 9:18] # Anden del (kolonner 38 til 83)

# Sammenkæd dem igen med nul-kolonnerne indsat de rigtige steder
form_stats_df = pd.concat([HomeTeamEncoded, df1, AwayTeamEncoded, df2], axis=1)

# Debugging: Tjek det endelige datasæt
print(form_stats_df)

# Find rækker med manglende data (NaN)
rows_with_nan = form_stats_df[form_stats_df.isna().any(axis=1)].index

# Fjern rækker med manglende data
form_stats_df_cleaned = form_stats_df.dropna()
print(form_stats_df_cleaned)

Y_df = pd.DataFrame(Y)
Y_cleaned = Y_df.drop(index=rows_with_nan)

# Gem det nye datasæt
form_stats_df_cleaned.to_excel("Fase1_Datamanipulation/Bundesliga/tyske_processed_input_data.xlsx", index=False)
Y_cleaned.to_excel("Fase1_Datamanipulation/Bundesliga/tyske_processed_output_labels.xlsx", index=False)

#Lav ark med kampresultater
# Læs kun relevante kolonner fra Excel-arket
columns_to_use = ["Date", "HomeTeam", "AwayTeam","FTR"]
matches = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_kampe_scrapped.xlsx", usecols=columns_to_use)
match_results= pd.concat([matches.drop(index=rows_with_nan), final_df.drop(index=rows_with_nan)], axis=1)

match_results.to_excel("Fase1_Datamanipulation/Bundesliga/tyske_match_results.xlsx", index=False)