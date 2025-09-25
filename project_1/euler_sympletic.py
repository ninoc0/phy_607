"""
Sympletic Euler method is a first order numerical integration method that 
approximates solutions to ODEs using two step Euler updating.

"""
import numpy as np

def euler_sympletic(f, y0, t):
    """
    Parameters
    ----------
    f: function f(t, y) -> is the derivative
    y0: initial state at t=0
    t: array of times

    Returns
    -------
    y: array of cavity power states at each time
    """
    t = np.asarray(t)
    y = np.zeros((len(t),) + np.shape(y0), dtype=np.result_type(y0))
    y[0] = y0

    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        y_next = y[i] + dt * f(t[i], y[i])
        y_next = y[i] + dt * f(t[i+1], y_next)

        y[i+1] = y_next

    return y
