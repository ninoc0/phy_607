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

def rk4_step(state, dt, q1, q2, m):
    """Single Runge–Kutta 4 step."""
    k1 = state_derivative(state, q1, q2, m)
    k2 = state_derivative(state + 0.5*dt*k1, q1, q2, m)
    k3 = state_derivative(state + 0.5*dt*k2, q1, q2, m)
    k4 = state_derivative(state + dt*k3, q1, q2, m)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def evolve_trajectory(state0, q1, q2, m, dt=1e-21, steps=5000):
    """
    Evolve the trajectory using coulomb interaction.

    Returns:
        trajectory : array of shape (N, 4): [x, y, vx, vy] at each step
    """
    trajectory = np.zeros((steps, 4)) 
    trajectory[0] = state0
    for i in range(1, steps): # main loop
        trajectory[i] = rk4_step(trajectory[i-1], dt, q1, q2, m) # calls rk4 state change
    return trajectory







