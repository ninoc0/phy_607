## Importing Libraries
import numpy as np

## Constants
k_e = 8.9875517923e9        # Coulomb constant (N·m²/C²)
e_charge = 1.602176634e-19  # Elementary charge (C)

def coulomb_force(pos, q1, q2):
    """
    Compute the Coulomb force.

    Arguements:
        pos : position of the particle (m)
        q1, q2 : charges of the particles (C)

    Returns:
        force : force vector (N)
    """
    r = np.linalg.norm(pos) # magnitude of position vector
    if r == 0:
        return np.zeros(2) # fail safe for when the position hasnt moved
    F = k_e * q1 * q2 / r**2
    return F * (pos / r)  # repulsive force

def state_derivative(state, q1, q2, m):
    """
    Compute derivatives of position and velocity.

    Arguements:
        state : current state vector [x, y, vx, vy]
        q1, q2 : charges of the particles (C)
        m : mass of particle (kg)
    Returns:
        dstate_dt : time derivatives of state vector[vx, vy, ax, ay]
    """
    x, y, vx, vy = state
    pos = np.array([x, y])
    force = coulomb_force(pos, q1, q2)
    acc = force / m
    return np.array([vx, vy, acc[0], acc[1]])
