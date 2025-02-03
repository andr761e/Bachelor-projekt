import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Indlæs data (ændr filstierne hvis nødvendigt)
X = pd.read_excel("Fase1_Datamanipulation/processed_input_data.xlsx").to_numpy()
Y = pd.read_excel("Fase1_Datamanipulation/processed_output_labels.xlsx").to_numpy()  # One-hot encoded labels

# Split data i træning og test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
print(Y_pred_proba)
print(Y_test)

# Beregn log loss
manual_logloss = -np.mean(np.sum(Y_test * np.log(Y_pred_proba + 1e-15), axis=1))
print(f"Manuel Log Loss: {manual_logloss}")

# Eksempel på output-sandsynligheder
print("Eksempel på sandsynligheder (3 første rækker):")
print(Y_pred_proba[:3])
