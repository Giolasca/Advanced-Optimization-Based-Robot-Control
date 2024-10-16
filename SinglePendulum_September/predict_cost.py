import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load the trained neural network model
model = tf.keras.models.load_model('nn_SP_225_unconstr.h5')

# 2. Load the original dataset
data = pd.read_csv('ocp_data_SP_target_225_unconstr.csv')

# 3. Extract the features (position and velocity) and the target (cost)
X = data[['position', 'velocity']].values
y = data['cost'].values  # Assuming the true costs are in a column named 'cost'

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

# 5. Refit the scaler on the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Function to scale the input state (position, velocity)
def scale_state(q, v):
    state = np.array([[q, v]])  # Create the input as a 2D array
    state_scaled = scaler.transform(state)  # Scale the input using the fitted scaler
    return state_scaled

# 6. Define a function that takes position and velocity as input and returns the predicted cost
def predict_cost(q, v):
    # Scale (standardize) the input state
    state_scaled = scale_state(q, v)
    
    # 7. Predict the cost using the neural network
    cost_pred = model.predict(state_scaled)
    
    # 8. Return the predicted cost
    return cost_pred[0][0]  # The result will be a 2D array, so we extract the scalar value

# 9. Calculate predictions for the test set
predicted_costs = []
for index, row in pd.DataFrame(X_test, columns=['position', 'velocity']).iterrows():
    q = row['position']
    v = row['velocity']
    cost = predict_cost(q, v)
    predicted_costs.append(cost)

# Convert predictions to numpy array
predicted_costs = np.array(predicted_costs)

# 10. Calculate evaluation metrics on the test set
mse = mean_squared_error(y_test, predicted_costs)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predicted_costs)
r2 = r2_score(y_test, predicted_costs)

# 11. Print the metrics for the test set
print(f'Mean Squared Error (MSE) on Test Set: {mse}')
print(f'Root Mean Squared Error (RMSE) on Test Set: {rmse}')
print(f'Mean Absolute Error (MAE) on Test Set: {mae}')
print(f'R² Score on Test Set: {r2}')

# Example of predicting for a specific state from the test set
q_example = X_test[0][0]  # Take the first position from the test set
v_example = X_test[0][1]  # Take the first velocity from the test set

# Call the function to predict the cost
cost_example = predict_cost(q_example, v_example)
print(f"The cost associated with the state (position={q_example}, velocity={v_example}) is: {cost_example}")
