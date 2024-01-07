import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import mpc_single_pendulum_conf as conf

def create_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    out1 = layers.Dense(64, activation='relu')(inputs)
    out2 = layers.Dense(32, activation='relu')(out1)
    outputs = layers.Dense(1, activation='relu')(out2)

    model = tf.keras.Model(inputs, outputs)
    return model

## Dataset creation

if __name__ == "__main__":
    # Import dataset and labels from configuration file
    train_data = conf.train_data
    train_label = conf.train_label
    test_data = conf.test_data
    test_label = conf.test_label

    model = create_model(input_shape=train_data.shape[1])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_label, epochs=100)

    # Test trained neural network
    results = model.evaluate(test_data, test_label)
    print("Test accuracy:", results[1])

    # Save the model weights
    model.save_weights("single_pendulum_test.h5")

    viable_states = []
    no_viable_states = []

    # Creation of initial states grid
    _ , state_array = conf.grid_states(121, 121)
    # _ , state_array = conf.random_states(5000)

    # Make predictions on the test set
    state_array = conf.scaler.fit_transform(state_array)
    label_pred = model.predict(state_array)

    # Convert probabilities to binary predictions
    binary_label = (label_pred > 0.5).astype(int)

    for i, label in enumerate(binary_label):
        if (label):
            viable_states.append(state_array[i,:])
        else:
            no_viable_states.append(state_array[i,:])
        
    viable_states = np.array(viable_states)
    no_viable_states = np.array(no_viable_states)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    if len(viable_states) != 0:
        ax.scatter(viable_states[:,0], viable_states[:,1], c='r', label='viable')
        ax.legend()
    if len(no_viable_states) != 0:
        ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='b', label='non-viable')
        ax.legend()
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()

