import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load the trained neural network model
model = tf.keras.models.load_model('nn_DP_180_180_unconstr.h5')

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

# 9. Calculate predictions for the test set
predicted_costs = []
for index, row in pd.DataFrame(X_test, columns=['q1', 'v1', 'q2', 'v2']).iterrows():
    q1 = row['q1']
    v1 = row['v1']
    q2 = row['q2']
    v2 = row['v2']
    cost = predict_cost(q1, v1, q2, v2)
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
q1_example = X_test[0][0]  # Take the first q1 from the test set
v1_example = X_test[0][1]  # Take the first v1 from the test set
q2_example = X_test[0][2]  # Take the first q2 from the test set
v2_example = X_test[0][3]  # Take the first v2 from the test set

# Call the function to predict the cost
cost_example = predict_cost(q1_example, v1_example, q2_example, v2_example)
print(f"The cost associated with the state (q1={q1_example}, v1={v1_example}, q2={q2_example}, v2={v2_example}) is: {cost_example}")
