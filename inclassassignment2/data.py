import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Load data
f = h5py.File('./data.hdf', 'r')
xpos = np.array(f['data/xpos'][:])
ypos = np.array(f['data/ypos'][:])

# Normalize the data for better gradient descent performance
x_mean, x_std = xpos.mean(), xpos.std()
y_mean, y_std = ypos.mean(), ypos.std()
x_norm = (xpos - x_mean) / x_std
y_norm = (ypos - y_mean) / y_std

def cubic_model(x, A, B, C, D):
    """Cubic polynomial model: y = Ax^3 + Bx^2 + Cx + D"""
    return A * x**3 + B * x**2 + C * x + D

def cost_function(x, y, A, B, C, D):
    """Mean Squared Error cost function"""
    predictions = cubic_model(x, A, B, C, D)
    mse = np.mean((predictions - y)**2)
    return mse

def compute_gradients(x, y, A, B, C, D):
    """Compute gradients of cost function with respect to A, B, C, D"""
    n = len(x)
    predictions = cubic_model(x, A, B, C, D)
    error = predictions - y
    
    # Partial derivatives
    dA = (2/n) * np.sum(error * x**3)
    dB = (2/n) * np.sum(error * x**2)
    dC = (2/n) * np.sum(error * x)
    dD = (2/n) * np.sum(error)
    
    return dA, dB, dC, dD

def gradient_descent(x, y, learning_rate=0.01, iterations=1000, tolerance=1e-6):
    """Gradient descent optimization"""
    # Initialize parameters
    A, B, C, D = 0.0, 0.0, 0.0, 0.0
    
    cost_history = []
    
    for i in range(iterations):
        # Compute current cost
        cost = cost_function(x, y, A, B, C, D)
        cost_history.append(cost)
        
        # Compute gradients
        dA, dB, dC, dD = compute_gradients(x, y, A, B, C, D)
        
        # Update parameters
        A -= learning_rate * dA
        B -= learning_rate * dB
        C -= learning_rate * dC
        D -= learning_rate * dD
        
        # Check for convergence
        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tolerance:
            print(f"Converged at iteration {i}")
            break
        
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost:.6f}")
    
    return A, B, C, D, cost_history

def likelihood_log(A, B, C, D, xpos, ypos): 
    A,B,C,D = A,B,C,D

    model = A * xpos ** 3 + B * xpos ** 2 + C * xpos + D 

    difference = ypos - model 

    log_L = -0.5 * sum(difference**2)

    return log_L



# Run gradient descent on normalized data
print("Training with Gradient Descent...")
A_norm, B_norm, C_norm, D_norm, cost_history = gradient_descent(
    x_norm, y_norm, 
    learning_rate=0.01, 
    iterations=2000
)
print(f"\nOptimized Parameters (normalized):")
print(f"A = {A_norm:.6f}")
print(f"B = {B_norm:.6f}")
print(f"C = {C_norm:.6f}")
print(f"D = {D_norm:.6f}")
print(f"Final Cost: {cost_history[-1]:.6f}")

# Transform parameters back to original scale
A = A_norm * y_std / x_std**3
B = B_norm * y_std / x_std**2
C = C_norm * y_std / x_std
D = y_mean + D_norm * y_std - A * x_mean**3 - B * x_mean**2 - C * x_mean

print(f"\nOptimized Parameters (original scale):")
print(f"A = {A:.6f}")
print(f"B = {B:.6f}")
print(f"C = {C:.6f}")
print(f"D = {D:.6f}")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Data and fitted curve
axes[0].scatter(xpos, ypos, alpha=0.5, label='Data')
x_line = np.linspace(xpos.min(), xpos.max(), 300)
y_pred = cubic_model(x_line, A, B, C, D)
axes[0].plot(x_line, y_pred, 'r-', linewidth=2, label='Fitted Cubic')
axes[0].set_xlabel('X Position')
axes[0].set_ylabel('Y Position')
axes[0].set_title('Cubic Polynomial Fit')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Cost function convergence
axes[1].plot(cost_history)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Cost (MSE)')
axes[1].set_title('Cost Function Convergence')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()

# Calculate Adjusted R-squared
y_pred_data = cubic_model(xpos, A, B, C, D)
ss_res = np.sum((ypos - y_pred_data)**2)
ss_tot = np.sum((ypos - ypos.mean())**2)
r_squared = 1 - (ss_res / ss_tot)

# Adjusted R-squared
n = len(ypos)  # number of data points
p = 4  # number of parameters (A, B, C, D)
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)


print(f"\nAdjusted R-squared: {adjusted_r_squared:.6f}")

#gives likelihood function 

print(f"The likelihood function: {likelihood_log(A = A, B = B, C = C, D = D, xpos= xpos, ypos = ypos)}")