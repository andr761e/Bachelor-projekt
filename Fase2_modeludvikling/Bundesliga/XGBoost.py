import numpy as np
import pandas as pd
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Methods import methods
met = methods()

# Indlæs data
X = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_processed_input_data.xlsx").to_numpy()
Y = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_processed_output_labels.xlsx").to_numpy() 
match_results = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_match_results.xlsx").to_numpy()

#Korrekt tidsafhængig split
# Definér splitpunktet baseret på antal rækker
split_point = int(0.8 * len(X))
# Tidsbaseret split
X_train, X_test = X[:split_point], X[split_point:]
Y_train, Y_test = Y[:split_point], Y[split_point:]
#Split kampresultaterne på samme måde (skal bruges senere)
match_result_split = match_results[split_point:]

# Opdel Y_train i separate arrays for hver klasse
Y_train_home = Y_train[:, 0]   # Sandsynlighed for hjemmebanesejr
Y_train_draw = Y_train[:, 1]   # Sandsynlighed for uafgjort
Y_train_away = Y_train[:, 2]   # Sandsynlighed for udebanesejr

# Initialiser modeller
model_home = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3, 
                          random_state=42, tree_method='gpu_hist', predictor='gpu_predictor')
model_draw = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3, 
                          random_state=42, tree_method='gpu_hist', predictor='gpu_predictor')
model_away = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3, 
                          random_state=42, tree_method='gpu_hist', predictor='gpu_predictor')

# Træn modellerne
model_home.fit(X_train, Y_train_home)
model_draw.fit(X_train, Y_train_draw)
model_away.fit(X_train, Y_train_away)

# Lav probabilistiske forudsigelser
probs_home = model_home.predict(X_test)
probs_draw = model_draw.predict(X_test)
probs_away = model_away.predict(X_test)

# Kombiner sandsynlighederne
probs_combined = np.column_stack([probs_home, probs_draw, probs_away])

#Prediction
Y_pred_proba = met.direct_normalization(probs_combined)

#Kalibrering
Y_pred_proba = met.polynomial_calibration(Y_pred_proba, Y_test, degree=3)

# Manuel beregning af log-loss
epsilon = 1e-15  # For at undgå log(0)
Y_pred_proba = np.clip(Y_pred_proba, epsilon, 1 - epsilon)  # Klip sandsynlighederne
log_loss_manual = -np.mean(np.sum(Y_test * np.log(Y_pred_proba), axis=1))
print(f"Manuel Log Loss: {log_loss_manual}")

# Eksempel på sandsynlighedsfordelinger
print("Første 3 rækker af Y_test (originale sandsynlighedsfordelinger):")
print(pd.DataFrame(Y_test[:3]))

print("\nFørste 3 rækker af Y_pred_proba (forudsigede sandsynligheder):")
print(pd.DataFrame(Y_pred_proba[:3]))


#PLOTS HERFRA
np.random.seed(42)

# Plot 1: Histogram for predicted probabilities
met.plot_histogram(Y_pred_proba, classes=["Hjemmesejr", "Uafgjort", "Udesejr"], colors=["blue", "orange", "green"])

# Plot 2: Comparison of actual vs predicted probabilities (first 10 games)
met.plot_comparison(Y_pred_proba, Y_test, classes=["Hjemmesejr", "Uafgjort", "Udesejr"], colors=["blue", "orange", "green"])

# Plot 3: Correlation between actual and predicted probabilities
met.plot_correlation(Y_pred_proba, Y_test, classes=["Hjemmesejr", "Uafgjort", "Udesejr"], colors=["blue", "orange", "green"])

# Plot 4: Line chart of log-loss contributions for each sample
met.plot_log_loss(Y_pred_proba, Y_test)

#EXPORT RESULTS FOR STATISTICAL ANALYSIS
#COLUMNS TO USE
columns = [
    "Date", "HomeTeam","AwayTeam","FTR",
    "HomeTeamELO","HomeGoals5", "HomePoints5", "HomeShots5", "HomeShotsOnTarget5", "HomeFouls5",
    "HomeCorners5", "HomeYellowCards5", "HomeRedCards5",
    "AwayTeamELO", "AwayGoals5", "AwayPoints5", "AwayShots5", "AwayShotsOnTarget5", "AwayFouls5",
    "AwayCorners5", "AwayYellowCards5", "AwayRedCards5", "MaxHodds","MaxDodds","MaxAodds",
    "YpredH", "YpredD","YpredA", "YtrueH", "YtrueD","YtrueA","DiffH", "DiffD","DiffA",
    "%DiffH","%DiffD","%DiffA"
]

result = pd.concat([pd.DataFrame(match_result_split), pd.concat([pd.DataFrame(Y_pred_proba), pd.concat([pd.DataFrame(Y_test),pd.concat([pd.DataFrame(Y_pred_proba - Y_test),pd.DataFrame((Y_pred_proba - Y_test)/Y_test*100)],axis=1)], axis=1)], axis=1)],axis=1)
result.columns = columns
result.to_excel("Fase3_ValueBettingAnalysis/Bundesliga/XGBoostResultsUnfiltered.xlsx", index=False)