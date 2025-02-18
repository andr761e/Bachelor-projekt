import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from Functions.Functions import DataUtils

# Indlæs data
X = pd.read_excel("Fase1_Datamanipulation/processed_input_data.xlsx").to_numpy()
Y = pd.read_excel("Fase1_Datamanipulation/processed_output_labels.xlsx").to_numpy()  # One-hot encoded labels
match_results = pd.read_excel("Fase1_Datamanipulation/match_results.xlsx").to_numpy()

# Split data i træning og test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#Split kampresultaterne på samme måde (skal bruges senere)
_, match_result_split, _, _ = train_test_split(match_results, Y, test_size=0.2, random_state=42)

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

# Classes
classes = ["Hjemmesejr", "Uafgjort", "Udesejr"]
colors = ["blue", "orange", "green"]

# Plot 1: Histogram for predicted probabilities
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.hist(Y_pred_proba[:, i], bins=20, alpha=0.5, label=f"Forudsagt: {classes[i]}", color=colors[i])
plt.title("Fordeling af forudsagte sandsynligheder pr. klasse")
plt.xlabel("Sandsynlighed")
plt.ylabel("Antal kampe")
plt.legend()
plt.show()

# Plot: Histogram for faktiske sandsynligheder
plt.figure(figsize=(10, 6))
for i in range(3):  # Antal klasser (hjemmesejr, uafgjort, udesejr)
    plt.hist(Y_test[:, i], bins=20, alpha=0.5, label=f"Faktisk: {classes[i]}", color=colors[i])
plt.title("Fordeling af faktiske sandsynligheder pr. klasse")
plt.xlabel("Sandsynlighed")
plt.ylabel("Antal kampe")
plt.legend()
plt.show()

# Plot 2: Comparison of actual vs predicted probabilities (first 10 games)
plt.figure(figsize=(12, 6))
x = np.arange(10)
for i in range(3):
    plt.plot(x, Y_test[:10, i], marker='o', linestyle='--', label=f"Faktisk: {classes[i]}", color=colors[i])
    plt.plot(x, Y_pred_proba[:10, i], marker='x', linestyle='-', label=f"Forudsagt: {classes[i]}", color=colors[i])
plt.title("Faktiske vs. forudsagte sandsynligheder (første 10 kampe)")
plt.xlabel("Kamp")
plt.ylabel("Sandsynlighed")
plt.xticks(x, [f"Kamp {i+1}" for i in x])
plt.legend()
plt.show()

# Plot 3: Correlation between actual and predicted probabilities
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.scatter(Y_test[:, i], Y_pred_proba[:, i], alpha=0.5, label=classes[i], color=colors[i])
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # Diagonal line for perfect predictions
plt.title("Sammenhæng mellem faktiske og forudsagte sandsynligheder")
plt.xlabel("Faktiske sandsynligheder")
plt.ylabel("Forudsagte sandsynligheder")
plt.legend()
plt.show()

# Plot 4: Line chart of log-loss contributions for each sample
log_loss_contributions = -np.sum(Y_test * np.log(Y_pred_proba + 1e-15), axis=1)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(Y_test)), log_loss_contributions, marker='o', linestyle='-', color='purple')
plt.title("Log-loss bidrag pr. kamp")
plt.xlabel("Kamp ID")
plt.ylabel("Log-loss bidrag")
plt.grid()
plt.show()

#EXPORT OUTLIERS FROM HERE
#COLUMNS TO USE
columns = [
    "Date", "HomeTeam","AwayTeam","FTR",
    "HomeTeamELO","HomeGoals5", "HomePoints5", "HomeShots5", "HomeShotsOnTarget5", "HomeFouls5",
    "HomeCorners5", "HomeYellowCards5", "HomeRedCards5",
    "AwayTeamELO", "AwayGoals5", "AwayPoints5", "AwayShots5", "AwayShotsOnTarget5", "AwayFouls5",
    "AwayCorners5", "AwayYellowCards5", "AwayRedCards5",
    "YpredH", "YpredD","YpredA", "YtrueH", "YtrueD","YtrueA","DiffH", "DiffD","DiffA"
]

#PERCENTAGE DEVIATION
# Apply filtering (10, 20, 30, 40 and 50% deviations)
rows_to_keep1 = DataUtils.get_filtered_rows_percentwise(Y_pred_proba, Y_test, x=0.1)
rows_to_keep2 = DataUtils.get_filtered_rows_percentwise(Y_pred_proba, Y_test, x=0.2)
rows_to_keep3 = DataUtils.get_filtered_rows_percentwise(Y_pred_proba, Y_test, x=0.3)
rows_to_keep4 = DataUtils.get_filtered_rows_percentwise(Y_pred_proba, Y_test, x=0.4)
rows_to_keep5 = DataUtils.get_filtered_rows_percentwise(Y_pred_proba, Y_test, x=0.5)

# Use the mask to filter rows
Y_pred_proba_filtered1 = Y_pred_proba[rows_to_keep1]
Y_test_filtered1 = Y_test[rows_to_keep1]
Y_pred_proba_filtered2 = Y_pred_proba[rows_to_keep2]
Y_test_filtered2 = Y_test[rows_to_keep2]
Y_pred_proba_filtered3 = Y_pred_proba[rows_to_keep3]
Y_test_filtered3 = Y_test[rows_to_keep3]
Y_pred_proba_filtered4 = Y_pred_proba[rows_to_keep4]
Y_test_filtered4 = Y_test[rows_to_keep4]
Y_pred_proba_filtered5 = Y_pred_proba[rows_to_keep5]
Y_test_filtered5 = Y_test[rows_to_keep5]

#Make final outlier dataFrames
result_10_deviate = pd.concat([pd.DataFrame(match_result_split[rows_to_keep1]), pd.concat([pd.DataFrame(Y_pred_proba_filtered1), pd.concat([pd.DataFrame(Y_test_filtered1),pd.DataFrame(Y_pred_proba_filtered1 - Y_test_filtered1)], axis=1)], axis=1)],axis=1)
result_10_deviate.columns = columns
result_20_deviate = pd.concat([pd.DataFrame(match_result_split[rows_to_keep2]), pd.concat([pd.DataFrame(Y_pred_proba_filtered2), pd.concat([pd.DataFrame(Y_test_filtered2),pd.DataFrame(Y_pred_proba_filtered2 - Y_test_filtered2)], axis=1)], axis=1)],axis=1)
result_20_deviate.columns = columns
result_30_deviate = pd.concat([pd.DataFrame(match_result_split[rows_to_keep3]), pd.concat([pd.DataFrame(Y_pred_proba_filtered3), pd.concat([pd.DataFrame(Y_test_filtered3),pd.DataFrame(Y_pred_proba_filtered3 - Y_test_filtered3)], axis=1)], axis=1)],axis=1)
result_30_deviate.columns = columns
result_40_deviate = pd.concat([pd.DataFrame(match_result_split[rows_to_keep4]), pd.concat([pd.DataFrame(Y_pred_proba_filtered4), pd.concat([pd.DataFrame(Y_test_filtered4),pd.DataFrame(Y_pred_proba_filtered4 - Y_test_filtered4)], axis=1)], axis=1)],axis=1)
result_40_deviate.columns = columns
result_50_deviate = pd.concat([pd.DataFrame(match_result_split[rows_to_keep5]), pd.concat([pd.DataFrame(Y_pred_proba_filtered5), pd.concat([pd.DataFrame(Y_test_filtered5),pd.DataFrame(Y_pred_proba_filtered5 - Y_test_filtered5)], axis=1)], axis=1)],axis=1)
result_50_deviate.columns = columns

#Export to excel
result_10_deviate.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliers/PercentageDeviation/10pointDeviationOutliers.xlsx", index=False)
result_20_deviate.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliers/PercentageDeviation/20pointDeviationOutliers.xlsx", index=False)
result_30_deviate.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliers/PercentageDeviation/30pointDeviationOutliers.xlsx", index=False)
result_40_deviate.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliers/PercentageDeviation/40pointDeviationOutliers.xlsx", index=False)
result_50_deviate.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliers/PercentageDeviation/50pointDeviationOutliers.xlsx", index=False)


#PERCENTAGE POINT DIFFERENCE
# Apply filtering (0.1, 0.2, 0.3, 0.4 and 0.5 point differences)
rows_to_keep1 = DataUtils.get_filtered_rows_percent_point(Y_pred_proba, Y_test, x=0.1)
rows_to_keep2 = DataUtils.get_filtered_rows_percent_point(Y_pred_proba, Y_test, x=0.2)
rows_to_keep3 = DataUtils.get_filtered_rows_percent_point(Y_pred_proba, Y_test, x=0.3)
rows_to_keep4 = DataUtils.get_filtered_rows_percent_point(Y_pred_proba, Y_test, x=0.4)
rows_to_keep5 = DataUtils.get_filtered_rows_percent_point(Y_pred_proba, Y_test, x=0.5)

# Use the mask to filter rows
Y_pred_proba_filtered1 = Y_pred_proba[rows_to_keep1]
Y_test_filtered1 = Y_test[rows_to_keep1]
Y_pred_proba_filtered2 = Y_pred_proba[rows_to_keep2]
Y_test_filtered2 = Y_test[rows_to_keep2]
Y_pred_proba_filtered3 = Y_pred_proba[rows_to_keep3]
Y_test_filtered3 = Y_test[rows_to_keep3]
Y_pred_proba_filtered4 = Y_pred_proba[rows_to_keep4]
Y_test_filtered4 = Y_test[rows_to_keep4]
Y_pred_proba_filtered5 = Y_pred_proba[rows_to_keep5]
Y_test_filtered5 = Y_test[rows_to_keep5]

#Make final outlier dataFrames
result_10_deviate = pd.concat([pd.DataFrame(match_result_split[rows_to_keep1]), pd.concat([pd.DataFrame(Y_pred_proba_filtered1), pd.concat([pd.DataFrame(Y_test_filtered1),pd.DataFrame(Y_pred_proba_filtered1 - Y_test_filtered1)], axis=1)], axis=1)],axis=1)
result_10_deviate.columns = columns
result_20_deviate = pd.concat([pd.DataFrame(match_result_split[rows_to_keep2]), pd.concat([pd.DataFrame(Y_pred_proba_filtered2), pd.concat([pd.DataFrame(Y_test_filtered2),pd.DataFrame(Y_pred_proba_filtered2 - Y_test_filtered2)], axis=1)], axis=1)],axis=1)
result_20_deviate.columns = columns
result_30_deviate = pd.concat([pd.DataFrame(match_result_split[rows_to_keep3]), pd.concat([pd.DataFrame(Y_pred_proba_filtered3), pd.concat([pd.DataFrame(Y_test_filtered3),pd.DataFrame(Y_pred_proba_filtered3 - Y_test_filtered3)], axis=1)], axis=1)],axis=1)
result_30_deviate.columns = columns
result_40_deviate = pd.concat([pd.DataFrame(match_result_split[rows_to_keep4]), pd.concat([pd.DataFrame(Y_pred_proba_filtered4), pd.concat([pd.DataFrame(Y_test_filtered4),pd.DataFrame(Y_pred_proba_filtered4 - Y_test_filtered4)], axis=1)], axis=1)],axis=1)
result_40_deviate.columns = columns
result_50_deviate = pd.concat([pd.DataFrame(match_result_split[rows_to_keep5]), pd.concat([pd.DataFrame(Y_pred_proba_filtered5), pd.concat([pd.DataFrame(Y_test_filtered5),pd.DataFrame(Y_pred_proba_filtered5 - Y_test_filtered5)], axis=1)], axis=1)],axis=1)
result_50_deviate.columns = columns

#Export to excel
result_10_deviate.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliers/PercentagePointDifference/10pointDifferenceOutliers.xlsx", index=False)
result_20_deviate.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliers/PercentagePointDifference/20pointDifferenceOutliers.xlsx", index=False)
result_30_deviate.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliers/PercentagePointDifference/30pointDifferenceOutliers.xlsx", index=False)
result_40_deviate.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliers/PercentagePointDifference/40pointDifferenceOutliers.xlsx", index=False)
result_50_deviate.to_excel("Fase3_ValueBettingAnalysis/NeuralNetworkOutliers/PercentagePointDifference/50pointDifferenceOutliers.xlsx", index=False)