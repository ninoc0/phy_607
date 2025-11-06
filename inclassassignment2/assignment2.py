import numpy as np 

# Define constants for your cubic function
A, B, C, D = 1, -2, 0.5, 0  # example coefficients

def f(x):
    return A*x**3 + B*x**2 + C*x + D

def df(x):
    return 3*A*x**2 + 2*B*x + C

def gradient_descent(starting_point, learning_rate, iterations):
    x = starting_point
    for i in range(iterations):
        grad = df(x)
        x = x - learning_rate * grad
        print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}")
    return x

starting_point = 0.0
learning_rate = 0.05
iterations = 20

minimum = gradient_descent(starting_point, learning_rate, iterations)
print(f"\nLocal minimum occurs at x = {minimum:.4f}, f(x) = {f(minimum):.4f}")
