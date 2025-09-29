"""
Trapezodial rule is a integrtion method that divides an integral into trapezoids and summing their area. 
"""
import numpy as np

def trapz(f, a, b, n):
    """
    Parameters
    ----------
    f: function f(t, y)
    a, b: the area to be integrated
    n: number of midpoints(trapezoids to be integrated under)

    Return
    -------
    float giving the integral
    """
    x = np.linspace(a, b, n + 1)
    y = f(x)
    dx = (b - a) / n
    return float(dx * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))
