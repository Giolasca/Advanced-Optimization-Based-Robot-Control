import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    data = pd.read_csv('ocp_data_SP_135.csv')

    # Extract features (initial state) and target (cost)
    X = data[['position', 'velocity']].values
    y = data['cost'].values.reshape(-1, 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=29)

    # Create the model using the create_model function
    model = create_model(input_shape=X_train.shape[1])

    # Compile the model
    #model.compile(optimizer='RMSprop', loss='mean_squared_error')
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=80, validation_data=(X_test, y_test))

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on Test Set: {mse}')

    # Plotting the test batch
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs. Predictions on Test Set')
    plt.show()

    # Save the trained model
    model.save('nn_SP_135_v2.h5')

    # Create grid
    q1_vals = np.linspace(3/4*np.pi, 5/4*np.pi, 121)
    v1_vals = np.linspace(-10, 10, 121)
    q1_mesh, v1_mesh = np.meshgrid(q1_vals, v1_vals)
    all_states = np.column_stack((q1_mesh.ravel(), v1_mesh.ravel()))

    # Make the predictions using the trained model
    predicted_costs = model.predict(all_states)

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
