"""
Runge Kutta method is a fourth order numerical integration method that 
approximates solutions to ODEs using four intermidate steps and creating a weighted average.

"""
import numpy as np

def rk4(f, y0, t):
    """
    Parameters
    ----------
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

    for i in range(len(t) - 1):
        # find the deriviative 
        dt = t[i+1] - t[i]
        ti, yi = t[i], y[i]
        # 4 intermediate steps
        k1 = f(ti, yi)
        k2 = f(ti + 0.5*dt, yi + 0.5*dt*k1)
        k3 = f(ti + 0.5*dt, yi + 0.5*dt*k2)
        k4 = f(ti + dt,     yi + dt*k3)
        # weighted averaging
        y[i+1] = yi + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y
