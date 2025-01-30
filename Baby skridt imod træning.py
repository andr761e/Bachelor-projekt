import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Indlæs input (X) og probabilistiske labels (Y)
X = pd.read_excel("processed_input_data.xlsx").to_numpy()
Y = pd.read_excel("processed_output_labels.xlsx").to_numpy()

# Opdel data i trænings- og testsæt
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Definer en simpel neural netværksmodel
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),  # Input layer
    Dense(64, activation="relu"),  # Skjult lag
    Dense(3, activation="softmax")  # Output layer (3 sandsynligheder)
])

# Kompiler modellen med kategorisk krydstabel-tap
model.compile(optimizer="adam", loss=CategoricalCrossentropy(), metrics=["accuracy"])

# Træn modellen
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2)

# Forudsig sandsynligheder
Y_pred_proba = model.predict(X_test)

# Udskriv sandsynligheder for den første række i testdata
print("Sandsynligheder for første kamp:")
print(f"Hjemmehold vinder: {Y_pred_proba[0][0]:.2f}, Uafgjort: {Y_pred_proba[0][1]:.2f}, Udehold vinder: {Y_pred_proba[0][2]:.2f}")

# Evaluér modellen med log-loss
loss = tf.keras.losses.CategoricalCrossentropy()(Y_test, Y_pred_proba).numpy() 
print(f"Log-loss: {loss:.4f}")
print(Y_pred_proba)
print(Y_test)
plus = Y_pred_proba[0][0] + Y_pred_proba[0][1] + Y_pred_proba[0][2]
print(plus)