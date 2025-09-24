"""
Explicit Euler method is a first order numerical integration method that 
approximates solutions to ODEs using a Taylor expansion.

"""
import numpy as np
def euler(f, y0, t):
    """
    Parameters
    -----------
    f: function f(t, y) -> is the derivative
    y0: initial state at t=0
    t: array of times
    
    Return
    -------
    y: array of cavity power states at each time
    """
    # import initial conditions
    y = np.zeros((len(t),), dtype=complex)
    y[0] = y0

    for i in range(len(t)-1):
        # find the derivative
        dt = t[i+1] - t[i]
        # find x_(n+1) = x(n) + ts * dx(n)/dt
        y[i+1] = y[i] + dt * f(t[i], y[i])
    return y
