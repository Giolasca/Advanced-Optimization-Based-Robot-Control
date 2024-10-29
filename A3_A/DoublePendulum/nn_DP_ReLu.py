import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    out1 = layers.Dense(128, activation='relu')(inputs)
    out2 = layers.Dense(64, activation='relu')(out1)
    out3 = layers.Dense(32, activation='relu')(out2)
    outputs = layers.Dense(1)(out3)

    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    # Load the CSV dataset
    data = pd.read_csv('combined_data_180_180.csv')

    # Extract features (initial state) and target (cost)
    X = data[['q1', 'v1', 'q2', 'v2']].values
    y = data['cost'].values.reshape(-1, 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

    # Standardize the features (scaling to mean 0 and standard deviation 1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create the model using the create_model function
    model = create_model(input_shape=X_train_scaled.shape[1])

    # Set a learning rate
    learning_rate = 0.0005
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(optimizer = optimizer, loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=300, validation_data=(X_test_scaled, y_test))

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
    plt.grid(True)  # Add grid
    plt.show()

    # Save the trained model
    model.save('nn_DP_180_180_unconstr.h5')
