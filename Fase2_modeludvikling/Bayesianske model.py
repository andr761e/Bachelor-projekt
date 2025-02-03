import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss, accuracy_score

# Indlæs data (ændr stierne hvis nødvendigt)
X = pd.read_excel("Fase1_Datamanipulation/processed_input_data.xlsx").to_numpy()
Y = pd.read_excel("Fase1_Datamanipulation/processed_output_labels.xlsx").to_numpy()  # One-hot encoded labels

# Konverter Y fra n x 3 (one-hot encoded) til n x 1 (klasse labels)
Y_classes = np.argmax(Y, axis=1)

# Split data i træning og test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_classes, test_size=0.2, random_state=42)

# Træn en Naive Bayes model
model = GaussianNB()
model.fit(X_train, Y_train)

# Lav forudsigelser
Y_pred_proba = model.predict_proba(X_test)

# Beregn log loss
logloss = log_loss(Y_test, Y_pred_proba)
print(f"Log Loss: {logloss}")

# Beregn accuracy
Y_pred = np.argmax(Y_pred_proba, axis=1)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")

# Eksempel på output-sandsynligheder
print("Eksempel på sandsynligheder (3 første rækker):")
print(Y_pred_proba[:3])
