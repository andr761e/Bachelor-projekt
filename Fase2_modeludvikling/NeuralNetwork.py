import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from Methods import methods
met = methods()

# Indlæs data
X = pd.read_excel("Fase1_Datamanipulation/processed_input_data.xlsx").to_numpy()
Y = pd.read_excel("Fase1_Datamanipulation/processed_output_labels.xlsx").to_numpy()  # One-hot encoded labels
match_results = pd.read_excel("Fase1_Datamanipulation/match_results.xlsx").to_numpy()

#Korrekt tidsafhængig split
# Definér splitpunktet baseret på antal rækker
split_point = int(0.8 * len(X))
# Tidsbaseret split
X_train, X_test = X[:split_point], X[split_point:]
Y_train, Y_test = Y[:split_point], Y[split_point:]
#Split kampresultaterne på samme måde (skal bruges senere)
match_result_split = match_results[split_point:]

# Definerer et avanceret neuralt netværk
model = keras.Sequential([
    keras.layers.Dense(256, activation="relu", input_shape=(X.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(3, activation="softmax")  # Outputlag med 3 sandsynligheder
])

# Kompiler modellen
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Callback til early stopping for at undgå overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Træn modellen
history = model.fit(X_train, Y_train,
                    epochs=100,
                    batch_size=64,
                    validation_data=(X_test, Y_test),
                    callbacks=[early_stopping],
                    verbose=1)

# Lav forudsigelser
Y_pred_proba = model.predict(X_test)

# Beregn log loss
manual_logloss = -np.mean(np.sum(Y_test * np.log(Y_pred_proba + 1e-15), axis=1))
print(f"Manuel Log Loss: {manual_logloss}")

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
    "AwayCorners5", "AwayYellowCards5", "AwayRedCards5",
    "YpredH", "YpredD","YpredA", "YtrueH", "YtrueD","YtrueA","DiffH", "DiffD","DiffA",
    "%DiffH","%DiffD","%DiffA"
]

result = pd.concat([pd.DataFrame(match_result_split), pd.concat([pd.DataFrame(Y_pred_proba), pd.concat([pd.DataFrame(Y_test),pd.concat([pd.DataFrame(Y_pred_proba - Y_test),pd.DataFrame((Y_pred_proba - Y_test)/Y_test*100)],axis=1)], axis=1)], axis=1)],axis=1)
result.columns = columns
result.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliersResultsUnfiltered.xlsx", index=False)

X_tysk = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_processed_input_data.xlsx").to_numpy()
Y_tysk = pd.read_excel("Fase1_Datamanipulation/Bundesliga/tyske_processed_output_labels.xlsx").to_numpy()

Y_pred_proba_tysk = model.predict(X_tysk)

# Beregn log loss
manual_logloss = -np.mean(np.sum(Y_tysk * np.log(Y_pred_proba_tysk + 1e-15), axis=1))
print(f"Manuel Log Loss: {manual_logloss}")

# Eksempel på sandsynlighedsfordelinger
print("Første 3 rækker af Y_test (originale sandsynlighedsfordelinger):")
print(pd.DataFrame(Y_tysk[:3]))

print("\nFørste 3 rækker af Y_pred_proba (forudsigede sandsynligheder):")
print(pd.DataFrame(Y_pred_proba_tysk[:3]))

#PLOTS HERFRA
np.random.seed(42)

# Plot 1: Histogram for predicted probabilities
met.plot_histogram(Y_pred_proba_tysk, classes=["Hjemmesejr", "Uafgjort", "Udesejr"], colors=["blue", "orange", "green"])

# Plot 2: Comparison of actual vs predicted probabilities (first 10 games)
met.plot_comparison(Y_pred_proba_tysk, Y_tysk, classes=["Hjemmesejr", "Uafgjort", "Udesejr"], colors=["blue", "orange", "green"])

# Plot 3: Correlation between actual and predicted probabilities
met.plot_correlation(Y_pred_proba_tysk, Y_tysk, classes=["Hjemmesejr", "Uafgjort", "Udesejr"], colors=["blue", "orange", "green"])

# Plot 4: Line chart of log-loss contributions for each sample
met.plot_log_loss(Y_pred_proba_tysk, Y_tysk)