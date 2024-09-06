import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.keras import layers
from keras import layers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    out1 = layers.Dense(64, activation='relu')(inputs)
    out2 = layers.Dense(32, activation='relu')(out1)
    out3 = layers.Dense(16, activation='relu')(out2)
    outputs = layers.Dense(1)(out3)

    model = tf.keras.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    # Generate a grid of states
    q1_vals = np.linspace(3/4*np.pi, 5/4*np.pi, 121)
    v1_vals = np.linspace(-10, 10, 121)
    q1_mesh, v1_mesh = np.meshgrid(q1_vals, v1_vals)
    state_array = np.column_stack((q1_mesh.ravel(), v1_mesh.ravel()))

    # Standardize the state array
    scaler_X = StandardScaler()
    state_array_scaled = scaler_X.fit_transform(state_array)

    # Create the model using the create_model function
    model = create_model(input_shape=state_array_scaled.shape[1])

    # Load the trained model
    model.load_weights("nn_SP_no_target.h5")
    #model.load_weights("nn_SP_target.h5")

    # Predict costs using the neural network
    cost_pred_scaled = model.predict(state_array_scaled)

    # Reshape the predictions to match the shape of the grid
    cost_grid = cost_pred_scaled.reshape(q1_mesh.shape)

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.contourf(q1_mesh, v1_mesh, cost_grid, cmap='viridis')
    plt.colorbar(label='Predicted Cost')
    plt.xlabel('q1 [rad]')
    plt.ylabel('v1 [rad/s]')
    plt.title('State Space with Predicted Costs')
    plt.show()