"""
A riemann sum is a method of approximating an integral by finding the area of rectangles for each sub-midpoint.

"""
import numpy as np

def riemann_midpoint(f, a, b, n):
    """
    Parameters
    ----------
    f: function f(t, y)
    a, b: the area to be integrated
    n: number of midpoints(rectangles to be integrated under)

    Return
    -------
    float giving the integral
    """
    x_edges = np.linspace(a, b, n + 1)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    dx = (b - a) / n
    return float(np.sum(f(x_mid)) * dx)
