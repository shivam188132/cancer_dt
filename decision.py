import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pickle

# Load the model and scaler from disk
with open('best_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to plot decision boundary
def plot_decision_boundary(X, y, model):
    # Define the min and max values for the plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Flatten the meshgrid into vectors and stack them together
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Make predictions over the grid points
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FF0000', '#0000FF']))
    
    # Plot the original points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.title("Decision Boundary with Scatter Plot")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# For simplicity, let's use the first two features for plotting
# Ensure you have at least two features to visualize the decision boundary
X_test_sample = X_test_scaled[:, :2]
y_test_sample = y_test

# Plotting the decision boundary
plot_decision_boundary(X_test_sample, y_test_sample, best_model)
