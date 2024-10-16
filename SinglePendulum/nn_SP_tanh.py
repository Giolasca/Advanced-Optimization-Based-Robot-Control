import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_model_tanh(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    out1 = layers.Dense(64, activation = 'tanh')(inputs)
    out2 = layers.Dense(32, activation = 'tanh')(out1)
    out3 = layers.Dense(16, activation = 'tanh')(out2)
    outputs = layers.Dense(1)(out3)

    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    # Load the CSV dataset
    data = pd.read_csv('Dataset/ocp_data_SP_target_180_unconstr.csv')

    # Extract features (initial state) and target (cost)
    X = data[['position', 'velocity']].values
    y = data['cost'].values.reshape(-1, 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

    # Standardize the features (scaling to mean 0 and standard deviation 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create the model using the create_model function
    model = create_model_tanh(input_shape=X_train_scaled.shape[1])

    # Set a specific learning rate
    learning_rate = 0.0005
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model with the specified learning rate
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=600, validation_data=(X_test_scaled, y_test))

    # Evaluate the model on the test set
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on Test Set: {mse}')

    # Plotting the test batch
    plt.scatter(y_test, y_pred, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='y=x', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs. Predictions on Test Set')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the trained model
    model.save('NN/nn_SP_180_unconstr_tanh.h5')

    # Create grid
    q1_vals = np.linspace(3/4*np.pi, 5/4*np.pi, 121)
    v1_vals = np.linspace(-10, 10, 121)
    q1_mesh, v1_mesh = np.meshgrid(q1_vals, v1_vals)
    all_states = np.column_stack((q1_mesh.ravel(), v1_mesh.ravel()))

    # Standardize the grid of states using the same scaler
    all_states_scaled = scaler.transform(all_states)

    # Make the predictions using the trained model
    predicted_costs = model.predict(all_states_scaled)

    # Print the original state and predicted cost
    for original_state, cost in zip(all_states, predicted_costs):
        print(f"Original State: {original_state}, Predicted Cost: {cost}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.contourf(q1_mesh, v1_mesh, predicted_costs.reshape(q1_mesh.shape), cmap='viridis')
    plt.colorbar(label='Predicted Cost')
    plt.xlabel('q1 [rad]')
    plt.ylabel('v1 [rad/s]')
    plt.title('Predicted Costs for State Space')
    plt.show()