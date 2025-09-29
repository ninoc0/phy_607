"""
Simspon's rule approximates an integral by dividing the area under the function into an even number 
of intervals and approximating each with a parabola.

"""
import numpy as np

def simpson(f, a, b, n):
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
    if n % 2 != 0:
        n += 1
        print("We increased your simpsons n by 1, since it needs to be even.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    dx = (b - a) / n
    return float((dx / 3.0) * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])))