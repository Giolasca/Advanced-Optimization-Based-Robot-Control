import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nn_SP_v1 import create_model_v1
from nn_SP_v2 import create_model_v2


if __name__ == "__main__":
    # Generate a grid of states
    q1_vals = np.linspace(3/4*np.pi, 5/4*np.pi, 121)
    v1_vals = np.linspace(-10, 10, 121)
    q1_mesh, v1_mesh = np.meshgrid(q1_vals, v1_vals)
    state_array = np.column_stack((q1_mesh.ravel(), v1_mesh.ravel()))

    # Create the model using the create_model function
    #model = create_model_v1(input_shape=state_array.shape[1])
    model = create_model_v2(input_shape=state_array.shape[1])

    # Load the trained model
    #model.load_weights("nn_SP_135_v1.h5")
    model.load_weights("nn_SP_135_v2.h5")
    #model.load_weights("nn_SP_180_v1.h5")
    #model.load_weights("nn_SP_180_v2.h5")
    #model.load_weights("nn_SP_225_v1.h5")
    #model.load_weights("nn_SP_225_v1.h5")

    # Predict costs using the neural network
    cost_pred_scaled = model.predict(state_array)

    # Reshape the predictions to match the shape of the grid
    cost_grid = cost_pred_scaled.reshape(q1_mesh.shape)

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.contourf(q1_mesh, v1_mesh, cost_grid, cmap='viridis')
    plt.colorbar(label='Predicted Cost')
    plt.xlabel('q1 [rad]')
    plt.ylabel('v1 [rad/s]')
    plt.title('State Space with Predicted Costs')
    plt.show()