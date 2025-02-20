import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Indlæs data (ændr stierne hvis nødvendigt)
X = pd.read_excel("Fase1_Datamanipulation/processed_input_data.xlsx").to_numpy()
Y = pd.read_excel("Fase1_Datamanipulation/processed_output_labels.xlsx").to_numpy()  # One-hot encoded labels
match_results = pd.read_excel("Fase1_Datamanipulation/match_results.xlsx").to_numpy()

# Split data i træning og test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#Split kampresultaterne på samme måde (skal bruges senere)
_, match_result_split, _, _ = train_test_split(match_results, Y, test_size=0.2, random_state=42)

# Standardisering af data (anbefales til KNN)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def custom_weights(distances):
    return 1 / (distances + 1e-5)  # Omvendt proportionalitet med afstand

# Konverter Y_train til integer-klasser (kræves af KNN)
Y_train_classes = np.argmax(Y_train, axis=1)

# Initialiser og træn KNN-modellen med vægtning baseret på afstand
model = KNeighborsClassifier(n_neighbors=600, weights=custom_weights, algorithm='auto')
model.fit(X_train, Y_train_classes)

# Lav forudsigelser (sandsynligheder)
Y_pred_proba = model.predict_proba(X_test)

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
result.to_excel("Fase3_ValueBettingAnalysis/KNNResultsUnfiltered.xlsx", index=False)