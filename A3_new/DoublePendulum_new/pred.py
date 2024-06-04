import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Carica il modello salvato
model_path = 'nn_DP_TensorFlow.h5'
model = load_model(model_path)

mean = np.array([3.13441092e+00, -2.05274944e-03,  3.14014654e+00,  3.21597412e-02])  # Mean dei dati di addestramento
std = np.array([0.46458406, 1.88237944, 0.45157242, 4.35383257])      # Std dei dati di addestramento

# Funzione per preprocessare l'input
def preprocess_input(configuration, mean, std):
    # Normalizza l'input basato su mean e std
    normalized_config = (configuration - mean) / std
    return normalized_config

# Funzione per predire il costo
def predict_cost(model, configuration, mean, std):
    # Preprocessa l'input
    preprocessed_config = preprocess_input(configuration, mean, std)
    # Aggiungi dimensione batch
    preprocessed_config = np.expand_dims(preprocessed_config, axis=0)
    # Effettua la predizione
    predicted_cost = model.predict(preprocessed_config)
    return predicted_cost

# Esempio di configurazione (modificare con i propri valori)
configuration = np.array([3.0630528372500483,1.0,2.9845130209103035,3.0])

# Predice il costo
predicted_cost = predict_cost(model, configuration, mean, std)
print(f"Predicted Cost: {predicted_cost}")

# Nota: sostituire q1_value, v1_value, q2_value, v2_value con i valori effettivi.



