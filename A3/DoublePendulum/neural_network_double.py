import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import visualkeras

# Load data from a .mat file
data = loadmat('data.mat')  # Replace 'data.mat' with your file path or name
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
        layers.Dense(1, activation='relu') 
    ])
    return model

model = create_model(input_shape=X_train.shape[1])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=100, validation_split=0.2)

# Print model summary
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
visualkeras.layered_view(model, to_file='model_layers.png').show()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Save the trained model in the native Keras format
model.save('viable_nonviable_model.h5')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert probabilities to binary predictions
y_pred_binary = (y_pred > 0.5).astype(int)

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


# Genera una griglia casuale di coordinate di velocità e posizione
num_points = 10000
random_positions1 = np.random.uniform(low=3/4*np.pi, high=5/4*np.pi, size=(num_points, 1))
random_velocities1 = np.random.uniform(low=-10, high=10, size=(num_points, 1))
random_positions2 = np.random.uniform(low=3/4*np.pi, high=5/4*np.pi, size=(num_points, 1))
random_velocities2 = np.random.uniform(low=-10, high=10, size=(num_points, 1))

random_points = np.concatenate([random_positions1, random_positions2,random_velocities1, random_velocities2], axis=1)

# Normalizza i punti
random_points_normalized = scaler.transform(random_points)

# Fai previsioni con la rete neurale
predictions = model.predict(random_points_normalized)
predictions_binary = (predictions > 0.5).astype(int)

# Estrai le coordinate dei punti in ciascuna classe
viable_points = random_points[predictions_binary.flatten() == 1]
non_viable_points = random_points[predictions_binary.flatten() == 0]

# Plotta lo scatter plot
plt.scatter(viable_points[:, 2], viable_points[:, 3], label='Viable', color='red')
plt.scatter(non_viable_points[:, 2], non_viable_points[:, 3], label='Non Viable', color='blue')

# Imposta etichette e titolo del grafico
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Classification of Random Points')

# Aggiungi la legenda
plt.legend()

# Mostra il grafico
plt.show()
