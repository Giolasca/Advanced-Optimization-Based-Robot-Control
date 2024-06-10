import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers
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
    # Load the CSV dataset
    data = pd.read_csv('ocp_data_NoBound.csv')

    # Extract features (initial state) and target (cost)
    X = data[['position', 'velocity']].values
    y = data['cost'].values.reshape(-1, 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=29)

    # Standardize the features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Standardize the target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Create the model using the create_model function
    model = create_model(input_shape=X_train_scaled.shape[1])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train_scaled, epochs=10, validation_data=(X_test_scaled, y_test_scaled))

    # Evaluate the model on the test set
    y_pred_scaled = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    print(f'Mean Squared Error on Test Set: {mse}')

    # Plotting the test batch
    plt.scatter(y_test_scaled, y_pred_scaled)
    plt.xlabel('True Values (scaled)')
    plt.ylabel('Predictions (scaled)')
    plt.title('True Values vs. Predictions on Test Set')
    plt.show()

    # Save the trained model
    model.save('nn_SP_1000.h5')

    # Create grid
    q1_vals = np.linspace(3/4*np.pi, 5/4*np.pi, 121)
    v1_vals = np.linspace(-10, 10, 121)
    q1_mesh, v1_mesh = np.meshgrid(q1_vals, v1_vals)
    all_states = np.column_stack((q1_mesh.ravel(), v1_mesh.ravel()))

    # Standardizing the states using the same scaler used during training
    new_states_scaled = scaler_X.transform(all_states)

    # Make the predictions using the trained model
    predicted_costs_scaled = model.predict(new_states_scaled)

    # De-standardize the predictions to obtain the results in the original scale
    predicted_costs = scaler_y.inverse_transform(predicted_costs_scaled)

    # Print the original state, scaled state, predicted scaled cost, and predicted cost
    for original_state, scaled_state, scaled_cost, cost in zip(all_states, new_states_scaled, predicted_costs_scaled, predicted_costs):
        print(f"Original State: {original_state}, Scaled State: {scaled_state}, Predicted Scaled Cost: {scaled_cost}, Predicted Cost: {cost}")

    # Creazione della colormap
    plt.figure(figsize=(10, 6))
    plt.contourf(q1_mesh, v1_mesh, predicted_costs_scaled.reshape(q1_mesh.shape), cmap='viridis')
    plt.colorbar(label='Predicted Cost')
    plt.xlabel('q1 [rad]')
    plt.ylabel('v1 [rad/s]')
    plt.title('Predicted Costs for State Space')
    plt.show()

    # Creazione della colormap
    plt.figure(figsize=(10, 6))
    plt.contourf(q1_mesh, v1_mesh, predicted_costs.reshape(q1_mesh.shape), cmap='viridis')
    plt.colorbar(label='Predicted Cost')
    plt.xlabel('q1 [rad]')
    plt.ylabel('v1 [rad/s]')
    plt.title('Predicted Costs for State Space')
    plt.show()