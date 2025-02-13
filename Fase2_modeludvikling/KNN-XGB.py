import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Indlæs data
X = pd.read_excel("Fase1_Datamanipulation/processed_input_data.xlsx").to_numpy()
Y = pd.read_excel("Fase1_Datamanipulation/processed_output_labels.xlsx").to_numpy()

# Split data i træning og test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Konverter Y til integer-klasser (kræves af både KNN og XGBClassifier)
Y_train_classes = np.argmax(Y_train, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Skalér data til KNN, men ikke XGB
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definér modeller
model_knn = KNeighborsClassifier(n_neighbors=500, weights='distance', algorithm='auto')
model_xgb = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=4, use_label_encoder=False, eval_metric='mlogloss')
model_rf = RandomForestClassifier(n_estimators=500, max_depth=10)

# Voting ensemble med skaleret data for KNN
ensemble_model = VotingClassifier(
    estimators=[
        ('knn', model_knn),
        ('xgb', model_xgb),
        ('rf', model_rf)
    ],
    voting='soft'
)

# Træn ensemblet
ensemble_model.fit(X_train_scaled, Y_train_classes) # Brug skaleret data!

# Lav forudsigelser med ensemblet
Y_pred_proba = ensemble_model.predict_proba(X_test_scaled)

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
