import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ocp_double_pendulum_conf as conf
from tensorflow.python.ops.numpy_ops import np_config
import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import visualkeras

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
    # Import csv as DataFrame
    df = pd.read_csv('datasets/double_data.csv')
    labels = df['viable']
    df = df.drop('viable', axis=1)

    # Set the ratio between training set and test set
    train_size = 0.8

    # Extract training and test sets from overall DataFrame randomizing the order
    train_df, test_df, train_labels, test_labels = train_test_split(df, labels, train_size=train_size, random_state=42)

    model = create_model(input_shape=train_df.shape[1])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_df, train_labels, epochs=20)

    # Print model summary
    model.summary()

    model.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    visualkeras.layered_view(model, to_file='model_layers.png').show()

    # Save the model weights
    model.save_weights("double_pendulum.h5")

    viable_states = []
    no_viable_states = []

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(test_df, test_labels)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Creation of initial states grid
    #_ , state_array = conf.grid_states(50, 50, 50, 50)
    _ , state_array = conf.random_states(4000)

    # Make predictions on the test set
    label_pred = model.predict(state_array)

    # Convert probabilities to binary predictions
    # Dipende da come si comporta la rete che troviamo
    binary_label = (label_pred > 0.5).astype(int)

    # Calculate and plot the confusion matrix
    cm = confusion_matrix(test_labels, binary_label)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

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
        ax.scatter(viable_states[:,2], viable_states[:,3], c='r', label='viable')
        ax.legend()
    if len(no_viable_states) != 0:
        ax.scatter(no_viable_states[:,2], no_viable_states[:,3], c='b', label='non-viable')
        ax.legend()
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()