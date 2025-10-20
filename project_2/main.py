## importing libraries
from interaction import evolve_trajectory
import sampling as s
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

## Constants
k_e = 8.9875517923e9        # Coulomb constant (N·m²/C²)
e_charge = 1.602176634e-19  # Elementary charge (C)

# Parameters
q_alpha = 2 * e_charge
q_gold = 79 * e_charge # nucleus
impact_params = [1e-13, 2e-13, 5e-13]
m_alpha = 6.64e-27
v0 = 1e7

plt.figure(figsize=(6,6))
plt.scatter(0,0,color='red',label='Gold nucleus')

for b in impact_params:
    state0 = np.array([-1e-11, b, v0, 0])
    traj = evolve_trajectory(state0, q_alpha, q_gold, m_alpha)
    plt.plot(traj[:,0], traj[:,1], label=f'b={b:.1e}')
    
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.legend()
plt.show()

# simulating impacts, add color map?? 
impact_params = s.sampling.sample_power_law(20, a=0.5) * 1e-14  # scale to meters
for b in impact_params:
    state0 = np.array([-2e-13, b, v0, 0])
    traj = evolve_trajectory(state0, q_alpha, q_gold, m_alpha)
    plt.plot(traj[:, 0], traj[:, 1])
plt.title("Simulated Scattering Trajectories with Power-Law")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.axis("equal")
plt.show()

# scattering angles
angles = []
for b in impact_params:
    state0 = np.array([-2e-13, b, v0, 0])
    traj = evolve_trajectory(state0, q_alpha, q_gold, m_alpha)
    final_v = traj[-1, 2:]
    theta = np.degrees(np.arctan2(final_v[1], final_v[0]))
    angles.append(theta)

plt.hist(angles, bins=15)
plt.xlabel("Scattering angle (degrees)")
plt.ylabel("Count")
plt.title("Distribution of Scattering Angles")
plt.show()