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
    # Load the data
    data = pd.read_csv('ocp_data.csv')

    # Extract features (initial state) and target (cost)
    X = data[['Position (q)', 'Velocity (v)']].values
    y = data['Cost'].values.reshape(-1, 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    model.fit(X_train_scaled, y_train_scaled, epochs=200, validation_data=(X_test, y_test))

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
    model.save('ocp_nn_model.h5')