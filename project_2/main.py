import argparse
import numpy as np
import matplotlib.pyplot as plt
from interaction import evolve_trajectory
import sampling as s
from chance_main import plot_atomic_numbers, plot_scattering_angles, plot_scattering_energy
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Constants
k_e = 8.9875517923e9        # Coulomb constant (N·m²/C²)
e_charge = 1.602176634e-19  # Elementary charge (C)

# Parameters
q_alpha = 2 * e_charge
q_gold = 79 * e_charge  # nucleus
m_alpha = 6.64e-27
v0 = 1e7

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
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    for b in impact_params:
        color = cmap(norm(b))
        state0 = np.array([-2e-13, b, v0, 0])
        traj = evolve_trajectory(state0, q_alpha, q_gold, m_alpha)
        plt.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5)

    plt.title(f"Simulated Scattering Trajectories (Power-Law, N={N})")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)
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

    plt.scatter(imp, angles)
    plt.xlabel("Impact Parameter (m)")
    plt.ylabel("Scattering Angle (radians)")
    plt.title("Scattering Angle vs Impact Parameter")
    plt.show()


# def plot_angle_histogram():
#     """Plot histogram of scattering angles."""
#     impact_params = s.sampling.sample_power_law(20, a=0.5) * 1e-12
#     angles = []

#     for b in impact_params:
#         state0 = np.array([-2e-13, b, v0, 0])
#         traj = evolve_trajectory(state0, q_alpha, q_gold, m_alpha)
#         final_v = traj[-1, 2:]
#         theta = np.arctan2(final_v[1], final_v[0])
#         angles.append(theta)

#     plt.hist(angles, bins=15)
#     plt.xlabel("Scattering Angle (radians)")
#     plt.ylabel("Count")
#     plt.title("Distribution of Scattering Angles")
#     plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot different scattering simulation results."
    )
    parser.add_argument(
        "--plot",
        choices=["basic", "powerlaw", "angle_vs_b", "hist", "energy_vs_angle", "atomic_number"],
        required=True,
        help=(
            "Choose which plot to generate: "
            "'basic' for fixed impact params, "
            "'powerlaw' for power-law trajectories, "
            "'angle_vs_b' for scattering vs b, "
            "'hist' for histogram of angles, "
            "'energy_vs_angle' for energy vs scattering, "
            "'atomic_number' for scattering at various atomic numbers, "
        ),
    )

    parser.add_argument(
        "--N",
        type=int,
        default=100,
        help="Number of samples to use (default: 100)."
    )

    args = parser.parse_args()

    if args.plot == "basic":
        plot_basic_trajectories()
    elif args.plot == "powerlaw":
        plot_powerlaw_trajectories(N=args.N)
    elif args.plot == "angle_vs_b":
        plot_scattering_angle_vs_b(N=args.N)
    elif args.plot == "hist":
        plot_scattering_angles(N=args.N)
    elif args.plot == "energy_vs_angle":
        plot_scattering_energy()
    elif args.plot == "atomic_number":
        plot_atomic_numbers()


if __name__ == "__main__":
    main()
