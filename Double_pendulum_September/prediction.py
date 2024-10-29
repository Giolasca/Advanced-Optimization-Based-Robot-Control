import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Load the trained neural network model
model = tf.keras.models.load_model('nn_DP_180_180_unconstr.h5')

# Print the model summary
model.summary()

# 2. Load the original dataset
data = pd.read_csv('combined_data_180_180.csv')

# 3. Extract the features (q1, v1, q2, v2) and the target (cost)
X = data[['q1', 'v1', 'q2', 'v2']].values
y = data['cost'].values  # Assuming the true costs are in a column named 'cost'

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

# 5. Refit the scaler on the training data (using all 4 features: q1, v1, q2, v2)
scaler = StandardScaler()
scaler.fit(X_train)

# Function to scale the input state (q1, v1, q2, v2)
def scale_state(q1, v1, q2, v2):
    state = np.array([[q1, v1, q2, v2]])  # Create the input as a 2D array
    state_scaled = scaler.transform(state)  # Scale the input using the fitted scaler
    return state_scaled

# 6. Define a function that takes q1, v1, q2, v2 as input and returns the predicted cost
def predict_cost(q1, v1, q2, v2):
    # Scale (standardize) the input state
    state_scaled = scale_state(q1, v1, q2, v2)
    
    # 7. Predict the cost using the neural network
    cost_pred = model.predict(state_scaled)
    
    # 8. Return the predicted cost
    return cost_pred[0][0]  # The result will be a 2D array, so we extract the scalar value

# Example of predicting for a specific state
q1_example = 2.601552959399072  # Example q1 value
v1_example = 19.374289295467587  # Example v1 value
q2_example = 2.530950152613336 # Example q2 value
v2_example = -18.996300649132916  # Example v2 value

# Call the function to predict the cost for the example state
cost_example = predict_cost(q1_example, v1_example, q2_example, v2_example)
print(f"The cost associated with the state (q1={q1_example}, v1={v1_example}, q2={q2_example}, v2={v2_example}) is: {cost_example}")

