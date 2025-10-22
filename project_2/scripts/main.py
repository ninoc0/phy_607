import argparse
import numpy as np
import matplotlib.pyplot as plt
from interaction import evolve_trajectory
import sampling as s
from plotting import *
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import linregress


def main():
    parser = argparse.ArgumentParser(
        description="Plot different scattering simulation results."
    )
    parser.add_argument(
        "--plot",
        choices=[
            "basic",
            "powerlaw",
            "angle_vs_b",
            "hist",
            "energy_vs_angle",
            "atomic_number",
            "benchmark",
            "time_bar"
        ],
        required=True,
        help=(
            "Choose which plot to generate: "
            "'basic' for fixed impact params, "
            "'powerlaw' for power-law trajectories, "
            "'angle_vs_b' for scattering vs b, "
            "'hist' for histogram of angles, "
            "'energy_vs_angle' for energy vs scattering, "
            "'atomic_number' for scattering at various atomic numbers, "
            "'benchmark' for execution time vs. number of particles"
            "'time_bar' for execution time of individual routines, displayed as a bar chart"
        ),
    )

    parser.add_argument(
        "--N", type=int, default=100, help="Number of samples to use (default: 100)."
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
    elif args.plot == "benchmark":
        plot_benchmark()
    elif args.plot =="time_bar":
        plot_time_bar()


if __name__ == "__main__":
    main()
