import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plot stuff and Tests
plot = 1
test = 1
random = 0
grid = 1 
plot_test = 1
plot_test_3D = 1

# Load data from a .mat file
data = loadmat('data_double.mat')  # Replace 'data.mat' with your file path or name
X = np.concatenate([data['viable_states'], data['non_viable_states']], axis=0)
y = np.concatenate([np.ones(data['viable_states'].shape[0]), np.zeros(data['non_viable_states'].shape[0])])

# Randomize the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the neural network model
def create_model(input_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='relu')  # Sigmoid for binary classification
    ])
    return model

model = create_model(input_shape=X_train.shape[1])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=100, validation_split=0.2)

# Print model summary
model.summary()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Save the trained model in the native Keras format
model.save('viable_nonviable_model.h5')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred_binary = (y_pred > 0.5).astype(int)


if(plot):
    # Calculate and plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Calculate and plot the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


## Neural network test ##
if(test):

    if(random):
        # Generates a random grid of velocity and position coordinates
        num_points = 100
        random_q1 = np.random.uniform(low=3/4*np.pi, high=5/4*np.pi, size=(num_points, 1))
        random_v1 = np.random.uniform(low=-10, high=10, size=(num_points, 1))
        random_q2 = np.random.uniform(low=3/4*np.pi, high=5/4*np.pi, size=(num_points, 1))
        random_v2 = np.random.uniform(low=-10, high=10, size=(num_points, 1))
        random_points = np.concatenate([random_q1, random_v1, random_q2, random_v2], axis=1)

        # Normalize the points
        random_points_normalized = scaler.transform(random_points)

        # Make predictions with the neural network
        predictions = model.predict(random_points_normalized)
        predictions_binary = (predictions > 0.5).astype(int)
    
        # Extract coordinates of points in each class
        viable_points = random_points[predictions_binary.flatten() == 1]
        non_viable_points = random_points[predictions_binary.flatten() == 0]

        if(plot_test):
            # Scatter plot for q1 and v1
            plt.scatter(viable_points[:, 0], viable_points[:, 1], label='Viable', color='red')
            plt.scatter(non_viable_points[:, 0], non_viable_points[:, 1], label='Non Viable', color='blue')
            plt.xlabel('Position q1')
            plt.ylabel('Velocity v1')
            plt.title('Classification of Random Points (q1 vs v1)')
            plt.legend()
            plt.show()

            # Scatter plot for q2 and v2 associated with selected q1 and v1
            plt.scatter(viable_points[:, 2], viable_points[:, 3], label='Viable', color='red')
            plt.scatter(non_viable_points[:, 2], non_viable_points[:, 3], label='Non Viable', color='blue')
            plt.xlabel('Position q2')
            plt.ylabel('Velocity v2')
            plt.title('Classification of Random Points (q2 vs v2 associated with selected q1 and v1)')
            plt.legend()
            plt.show()


    if(grid):
        # Creation of initial states grid
        n_pos_q1 = 12
        n_vel_v1 = 12
        n_pos_q2 = 20
        n_vel_v2 = 20
        n_ics = n_pos_q1 * n_pos_q2 * n_vel_v1 * n_vel_v2
        possible_q1 = np.linspace(3/4*np.pi, 5/4*np.pi, num=n_pos_q1)
        possible_v1 = np.linspace(-10, 10, num=n_vel_v1)
        possible_q2 = np.linspace(3/4*np.pi, 5/4*np.pi, num=n_pos_q2)
        possible_v2 = np.linspace(-10, 10, num=n_vel_v2)
        grid_points = np.zeros((n_ics, 4))

        i = 0
        for q1 in possible_q1:
            for v1 in possible_v1:
                for q2 in possible_q2:
                    for v2 in possible_v2:
                        grid_points[i, :] = np.array([q1, v1, q2, v2])
                        i += 1
        
        # Normalize the points
        grid_points_normalized = scaler.transform(grid_points)

        # Make predictions with the neural network
        predictions = model.predict(grid_points_normalized)
        predictions_binary = (predictions > 0.5).astype(int)
    
        # Extract coordinates of points in each class
        viable_points = grid_points[predictions_binary.flatten() == 1]
        non_viable_points = grid_points[predictions_binary.flatten() == 0]

        # Get unique combinations of [q1, v1]
        unique_combinations_viable, counts_viable = np.unique(viable_points[:, :2], axis=0, return_counts=True)
        unique_combinations_non_viable, counts_non_viable = np.unique(non_viable_points[:, :2], axis=0, return_counts=True)

        if(plot_test_3D):
            # Crea il grafico 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot Viable points in red
            ax.scatter(unique_combinations_viable[:, 0], unique_combinations_viable[:, 1], counts_viable, c='red', marker='o', label='Viable', s=20)

            # Plot Non_Viable points in blue
            ax.scatter(unique_combinations_non_viable[:, 0], unique_combinations_non_viable[:, 1], counts_non_viable, c='blue', marker='o', label='No Viable', s=20)

            ax.set_xlabel('q1 [rad]')
            ax.set_ylabel('v1 [rad/s]')
            ax.set_zlabel('Number of points')
            ax.set_title('Number of Viable/NoViable for given q1-v1')
            ax.legend()

        plt.show()