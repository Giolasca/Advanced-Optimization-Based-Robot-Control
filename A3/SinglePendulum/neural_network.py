import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import mpc_single_pendulum_conf as conf
from sklearn.metrics import confusion_matrix
import seaborn as sns

def create_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    out1 = layers.Dense(64, activation='relu')(inputs)
    out2 = layers.Dense(32, activation='relu')(out1)
    out3 = layers.Dense(16, activation='relu')(out2)
    outputs = layers.Dense(1, activation='relu')(out3)

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
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_summary.png', show_shapes=True)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with loss history
    history = model.fit(train_data, train_label, epochs=300, validation_data=(test_data, test_label))

    # Test trained neural network
    results = model.evaluate(test_data, test_label)
    print("Test accuracy:", results[1])

    # Save the model weights
    model.save_weights("single_pendulum_funziona.h5")

    viable_states = []
    no_viable_states = []

    # Make predictions on the test set
    test_data = conf.scaler.fit_transform(test_data)
    label_pred = model.predict(test_data)

    # Convert probabilities to binary predictions
    binary_label = (label_pred > 0.5).astype(int)

    # Plot confusion matrix
    cm = confusion_matrix(test_label, binary_label)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Viable', 'Viable'], yticklabels=['Non-Viable', 'Viable'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot loss function
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    for i, label in enumerate(binary_label):
        if label:
            viable_states.append(test_data[i, :])
        else:
            no_viable_states.append(test_data[i, :])
        
    viable_states = np.array(viable_states)
    no_viable_states = np.array(no_viable_states)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()
    if len(viable_states) != 0:
        ax.scatter(viable_states[:, 0], viable_states[:, 1], c='r', label='viable')
        ax.legend()
    if len(no_viable_states) != 0:
        ax.scatter(no_viable_states[:, 0], no_viable_states[:, 1], c='b', label='non-viable')
        ax.legend()
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()
