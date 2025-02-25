import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from Methods import methods
met = methods()

# Indlæs data
X = pd.read_excel("Fase1_Datamanipulation/processed_input_data.xlsx").to_numpy()
Y = pd.read_excel("Fase1_Datamanipulation/processed_output_labels.xlsx").to_numpy()
match_results = pd.read_excel("Fase1_Datamanipulation/match_results.xlsx").to_numpy()

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#Split kampresultaterne på samme måde (skal bruges senere)
_, match_result_split, _, _ = train_test_split(match_results, Y, test_size=0.2, random_state=42)

# Opdel Y_train i separate arrays for hver klasse
Y_train_home = Y_train[:, 0]
Y_train_draw = Y_train[:, 1]
Y_train_away = Y_train[:, 2]

# Initialiser modeller
model_xgb = XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.01, max_depth=4, random_state=42)
model_lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, random_state=42)
model_cat = CatBoostRegressor(iterations=500, learning_rate=0.01, depth=4, verbose=0, random_seed=42)

# Ensemble modeller
ensemble_home = VotingRegressor(estimators=[('xgb', model_xgb), ('lgbm', model_lgbm), ('cat', model_cat)])
ensemble_draw = VotingRegressor(estimators=[('xgb', model_xgb), ('lgbm', model_lgbm), ('cat', model_cat)])
ensemble_away = VotingRegressor(estimators=[('xgb', model_xgb), ('lgbm', model_lgbm), ('cat', model_cat)])

# Træn hver ensemble-model
ensemble_home.fit(X_train, Y_train_home)
ensemble_draw.fit(X_train, Y_train_draw)
ensemble_away.fit(X_train, Y_train_away)

# Lav probabilistiske forudsigelser
probs_home = ensemble_home.predict(X_test)
probs_draw = ensemble_draw.predict(X_test)
probs_away = ensemble_away.predict(X_test)

# Kombiner sandsynlighederne med direkte normalisering
probs_combined = np.column_stack([probs_home, probs_draw, probs_away])

#Prediction
Y_pred_proba = met.direct_normalization(probs_combined)
#Alternativ
#Y_pred_proba = softmax(probs_combined, axis=1)  # Softmax sikrer, at hver række summer til 1

#Kalibrering
Y_pred_proba = met.polynomial_calibration(Y_pred_proba, Y_test, degree=3)

# Manuel beregning af log-loss
epsilon = 1e-15
Y_pred_proba = np.clip(Y_pred_proba, epsilon, 1 - epsilon)
log_loss_manual = -np.mean(np.sum(Y_test * np.log(Y_pred_proba), axis=1))
print(f"Manuel Log Loss: {log_loss_manual}")

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
    "AwayCorners5", "AwayYellowCards5", "AwayRedCards5",
    "YpredH", "YpredD","YpredA", "YtrueH", "YtrueD","YtrueA","DiffH", "DiffD","DiffA",
    "%DiffH","%DiffD","%DiffA"
]

result = pd.concat([pd.DataFrame(match_result_split), pd.concat([pd.DataFrame(Y_pred_proba), pd.concat([pd.DataFrame(Y_test),pd.concat([pd.DataFrame(Y_pred_proba - Y_test),pd.DataFrame((Y_pred_proba - Y_test)/Y_test*100)],axis=1)], axis=1)], axis=1)],axis=1)
result.columns = columns
result.to_excel("Fase3_ValueBettingAnalysis/GradientEnsembleResultsUnfiltered.xlsx", index=False)