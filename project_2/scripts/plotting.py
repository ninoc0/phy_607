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

def plot_basic_trajectories():
    """Plot trajectories for fixed impact parameters."""
    impact_params = [1e-13, 2e-13, 5e-13]
    plt.figure(figsize=(6, 6))
    plt.scatter(0, 0, color="red", label="Gold nucleus")

    for b in impact_params:
        state0 = np.array([-1e-11, b, v0, 0])
        traj = evolve_trajectory(state0, q_alpha, q_gold, m_alpha)
        plt.plot(traj[:, 0], traj[:, 1], label=f"b={b:.1e}")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.legend()
    plt.title("Basic Scattering Trajectories")
    plt.show()

def plot_powerlaw_trajectories(N):
    """Plot multiple trajectories sampled from a power-law distribution."""
    impact_params = s.sampling.sample_power_law(N, a=0.5) * 1e-12

    plt.figure(figsize=(6, 6))
    plt.scatter(0, 0, color="red", label="Gold nucleus", zorder=3)

    norm = mcolors.Normalize(vmin=np.min(impact_params), vmax=np.max(impact_params))
    cmap = cm.turbo

    for b in impact_params:
        color = cmap(norm(b))
        state0 = np.array([-2e-13, b, v0, 0])
        traj = evolve_trajectory(state0, q_alpha, q_gold, m_alpha)
        plt.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5)

    plt.title(f"Simulated Scattering Trajectories (Power-Law, N={N})")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(alpha=0.5)
    plt.axis("equal")
    plt.show()

def plot_scattering_angle_vs_b(N=20):
    """Plot scattering angle as a function of impact parameter."""
    impact_params = s.sampling.sample_power_law(N, a=0.5) * 1e-12
    angles, imp = [], []

    for b in impact_params:
        state0 = np.array([-2e-13, b, v0, 0])
        traj = evolve_trajectory(state0, q_alpha, q_gold, m_alpha)
        final_v = traj[-1, 2:]
        theta = np.arctan2(final_v[1], final_v[0])
        angles.append(theta)
        imp.append(b)

    regression_sol = linregress(imp, 1/np.tan(np.array(angles)/2))

    xvals = np.linspace(min(imp), max(imp), 1000)
    plt.scatter(imp, 1/np.tan(np.array(angles)/2), marker='^')
    plt.plot(xvals, linear(xvals, regression_sol.slope, regression_sol.intercept), color='red')
    plt.text(0.2e-12, 30, rf"$R^2$ = {np.round(regression_sol.rvalue**2,5)}",fontsize=14)
    plt.xlabel(r"$b$ [m]",fontsize=14)
    plt.ylabel(r"$\cot(\theta/2)$",fontsize=14)
    plt.title("Relationship between Impact Parameter and Scattering Angle",fontsize=14)
    plt.grid(alpha = 0.5)
    plt.savefig("bvsangle.png",dpi=300)


def plot_scattering_angles(N = 100):
    # scattering angles, allowing negative impact parameter, increasing number of sampled impact parameters
    angles = []

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
<<<<<<< HEAD:project_2/chance_main.py
    plt.title(f"Frequency of Scattering Angles for {N} Simulations")
    plt.xlabel("Scattering Angle (Radians)")
=======
    plt.xlabel("Scattering Angle [Radians]",fontsize=14)
>>>>>>> de01dd4 (moved all plotting functions into one file, moved scripts into subdirectory.):project_2/scripts/plotting.py
    plt.ylabel("Counts")
    plt.show()

def plot_scattering_energy():
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
    plt.scatter(energies/(e_charge * 1e6),1/np.tan(np.array(angles)/2),label="Data",marker = "^")
    plt.plot(x_vals/(e_charge * 1e6), linear(x_vals/(e_charge * 1e6), regression_sol.slope, regression_sol.intercept),color='red', alpha=0.5,label="Line of Best Fit")
    plt.xlabel("Incident Energy [MeV]",fontsize=14)
    plt.legend(loc='lower right')
    plt.text(4.2,72.5,rf"$R^2$ = {np.round(regression_sol.rvalue**2,5)}",fontsize=14)
    plt.ylabel(r"$\cot(\theta/2)$",fontsize=14)
<<<<<<< HEAD:project_2/chance_main.py
    plt.show()
    # plt.savefig("EvsAngle.png",dpi=300)
    plt.clf()
=======
    plt.savefig("EvsAngle.png",dpi=300)
    plt.grid(alpha = 0.5)
>>>>>>> de01dd4 (moved all plotting functions into one file, moved scripts into subdirectory.):project_2/scripts/plotting.py

def plot_atomic_numbers():
    # different atomic numbers

    Z = np.array([55,60,70,78,79,82,83,87,92])
    nuclear_charges = Z * e_charge
    nuclei = ["Cs","Nd","Yb","Pt","Au","Pb","Bi","Fr","U"]

    angles_Z = []

    impact_param = 1e-12 #fixed impact parameter
    v0 = 1e7 #fixed initial velocity (i.e, fixed energy)

    for q in nuclear_charges:
            
        state0 = np.array([-2e-13, impact_param, v0, 0])
        traj = evolve_trajectory(state0, q_alpha, q, m_alpha)
        final_v = traj[-1, 2:]
        theta = np.arctan2(final_v[1], final_v[0])
        angles_Z.append(theta)

    #hist, bin_edges = np.histogram(angles, bins = 10)

    regression_sol_Z = linregress(1/Z, 1/np.tan(np.array(angles_Z)/2))

    x_vals_Z = np.linspace(min(1/Z), max(1/Z),1000)

    plt.figure(figsize=(8,6))
    plt.title("Relationship between Scattering Angle and Atomic Number", fontsize=14)
    plt.scatter(1/Z,1/np.tan(np.array(angles_Z)/2),label="Data", marker = "^")
    k = 0
    for i,c in zip(nuclei, Z):
        if k%2 == 0:
            plt.text(1/(Z[k]), 1/np.tan(angles_Z[k]/2)-0.5, rf"${i}^{{{c}}}$",fontsize=12)
        else:
            plt.text(1/(Z[k]), 1/np.tan(angles_Z[k]/2), rf"${i}^{{{c}}}$",fontsize=12)
        k+=1

    plt.text(1/Z[2], 42.5, rf"$R^2$ = {np.round(regression_sol_Z.rvalue**2,5)}",fontsize=14)
    plt.plot(x_vals_Z, linear(x_vals_Z, regression_sol_Z.slope, regression_sol_Z.intercept),color='red', alpha=0.5,label="Line of Best Fit")
    plt.xlabel(r"1/Z",fontsize=14)
    plt.legend(loc='lower right')
    plt.ylabel(r"$\cot(\theta/2)$",fontsize=14)
<<<<<<< HEAD:project_2/chance_main.py
    plt.show()
    # plt.savefig("ZvsAngle.png",dpi=300)
=======
    plt.grid(alpha = 0.5)
    plt.savefig("ZvsAngle.png",dpi=300)
>>>>>>> de01dd4 (moved all plotting functions into one file, moved scripts into subdirectory.):project_2/scripts/plotting.py

