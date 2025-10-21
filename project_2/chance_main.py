## importing libraries
from interaction import evolve_trajectory
import sampling as s
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import linregress

## Constants
k_e = 8.9875517923e9        # Coulomb constant (N·m²/C²)
e_charge = 1.602176634e-19  # Elementary charge (C)

# Parameters
q_alpha = 2 * e_charge
q_gold = 79 * e_charge # nucleus
impact_params = [1e-13, 2e-13, 5e-13]
m_alpha = 6.64e-27
v0 = 1e7

# Sampling from Energy distribution
E_mean_ev, E_std_ev = 5.1e6, 0.1e6
E_mean_joules, E_std_joules = E_mean_ev * e_charge, E_std_ev * e_charge
energies = s.sampling.sample_gaussian(50, 1, 0.1) * E_mean_joules

def linear(x,m,b):
    return m * x + b


# scattering angles, allowing negative impact parameter, increasing number of sampled impact parameters
angles = []

N = 100

impact_params = s.sampling.sample_power_law(N, a = 0.5) * 1e-12

for b in impact_params:
    
    if np.random.uniform(0,1)<0.5:  #allows for random sign flip, demonstration of angular symmetry of problem
        b = -b

    state0 = np.array([-2e-13, b, v0, 0])
    traj = evolve_trajectory(state0, q_alpha, q_gold, m_alpha)
    final_v = traj[-1, 2:]
    theta = np.arctan2(final_v[1], final_v[0])
    angles.append(theta)

hist, bin_edges = np.histogram(angles, bins = 10)


plt.hist(angles, bins = 20)
plt.xlabel("Scattering Angle (Radians)")
plt.ylabel("Counts")
plt.show()


# scattering angles for different incident particle energies
angles = []

impact_param = 1e-12 #fixed impact parameter

for E in energies:
    
    v0 = np.sqrt((2 * E) / m_alpha)
        
    state0 = np.array([-2e-13, impact_param, v0, 0])
    traj = evolve_trajectory(state0, q_alpha, q_gold, m_alpha)
    final_v = traj[-1, 2:]
    theta = np.arctan2(final_v[1], final_v[0])
    angles.append(theta)

hist, bin_edges = np.histogram(angles, bins = 10)

regression_sol = linregress(energies/(e_charge * 1e6), 1/np.tan(np.array(angles)/2))

x_vals = np.linspace(min(energies), max(energies),1000)

plt.figure(figsize=(8,6))
plt.title("Relationship between Scattering Angle and Incident Energy", fontsize=14)
plt.scatter(energies/(e_charge * 1e6),1/np.tan(np.array(angles)/2),label="Data")
plt.plot(x_vals/(e_charge * 1e6), linear(x_vals/(e_charge * 1e6), regression_sol.slope, regression_sol.intercept),color='red', alpha=0.5,label="Line of Best Fit")
plt.xlabel("Incident Energy [MeV]",fontsize=14)
plt.legend(loc='lower right')
plt.text(4.2,72.5,rf"$R^2$ = {np.round(regression_sol.rvalue**2,8)}",fontsize=14)
plt.ylabel(r"$\cot(\theta/2)$",fontsize=14)
#plt.show()
#plt.savefig("EvsAngle.png",dpi=300)
