import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Indlæs data
X = pd.read_excel("Fase1_Datamanipulation/processed_input_data.xlsx").to_numpy()
Y = pd.read_excel("Fase1_Datamanipulation/processed_output_labels.xlsx").to_numpy()

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Opdel Y_train i separate arrays for hver klasse
Y_train_home = Y_train[:, 0]   # Sandsynlighed for hjemmebanesejr
Y_train_draw = Y_train[:, 1]   # Sandsynlighed for uafgjort
Y_train_away = Y_train[:, 2]   # Sandsynlighed for udebanesejr

# Initialiser modeller
model_home = LGBMRegressor(boosting_type='gbdt', device='gpu', n_estimators=1000, learning_rate=0.01, max_depth=3, random_state=42)
model_draw = LGBMRegressor(boosting_type='gbdt', device='gpu', n_estimators=1000, learning_rate=0.01, max_depth=3, random_state=42)
model_away = LGBMRegressor(boosting_type='gbdt', device='gpu', n_estimators=1000, learning_rate=0.01, max_depth=3, random_state=42)

# Træn modellerne
model_home.fit(X_train, Y_train_home)
model_draw.fit(X_train, Y_train_draw)
model_away.fit(X_train, Y_train_away)

# Lav probabilistiske forudsigelser
probs_home = model_home.predict(X_test)
probs_draw = model_draw.predict(X_test)
probs_away = model_away.predict(X_test)

# Kombiner sandsynlighederne med softmax
probs_combined = np.column_stack([probs_home, probs_draw, probs_away])
Y_pred_proba = softmax(probs_combined, axis=1)

def polynomial_calibration(Y_pred_proba, Y_test, degree=3):
    calibrated_probs = []
    for i in range(3):
        # Fit polynomiel regression
        poly = Polynomial.fit(Y_pred_proba[:, i], Y_test[:, i], deg=degree)
        calibrated_probs.append(poly(Y_pred_proba[:, i]))
    return np.column_stack(calibrated_probs)

Y_pred_proba = polynomial_calibration(Y_pred_proba, Y_test, degree=3)

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