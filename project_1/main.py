"""
Minimal Fabry–Perot cavity ODE demo.

Examples:
  python main.py ode --mode ringdown --tmax 0.5 --T1 3e-3 --T2 3e-3 --Lrt 1e-3
  python main.py ode --mode stepon  --delta 2e5 --tmax 5e-4 --npts 2000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from euler_explicit import euler 

# Analytic Model

def cavity_rhs(t, a, kappa, kext, delta, s_in):
    """RHS for cavity field a(t)."""
    s = s_in(t)
    return (1j * delta - 0.5 * kappa) * a + np.sqrt(kext) * s

def analytic_ringdown(t, a0, kappa, delta):
    return a0 * np.exp((1j * delta - 0.5 * kappa) * t)

def analytic_stepon(t, s0, kappa, kext, delta):
    a_ss = np.sqrt(kext) * s0 / (0.5 * kappa - 1j * delta)
    return a_ss * (1.0 - np.exp((1j * delta - 0.5 * kappa) * t))

# ---------- driver -----------------------------------------------------------

def run_ode(args):
    # defining params
    c = 299792458.0
    L = args.length
    kappa = (c / (2 * L)) * (args.T1 + args.T2 + args.Lrt)
    kext  = (c / (2 * L)) * args.T1
    delta = args.delta

    # time array
    t = np.linspace(0.0, args.tmax, args.npts)

    # options for interaction
    if args.mode == "ringdown":
        a0 = args.a0 + 0j
        s_in = (lambda _t: 0.0)             # input off
        a_ref = analytic_ringdown(t, a0, kappa, delta)
    else:  # stepon
        a0 = 0.0 + 0j
        s_in = (lambda _t: args.s0)  # constant input
        a_ref = analytic_stepon(t, args.s0, kappa, kext, delta)

    # Euler integration
    rhs = lambda ti, ai: cavity_rhs(ti, ai, kappa, kext, delta, s_in)
    a_num = euler(rhs, a0, t)

    # plot |a|^2
    plt.figure()
    plt.plot(t * 1e6, np.abs(a_ref) ** 2, label="analytic |a|^2")
    plt.plot(t * 1e6, np.abs(a_num) ** 2, "--", label="Euler |a|^2")
    plt.xlabel("time [µs]")
    plt.ylabel("Circulating Power [W]")
    plt.title(f"Cavity {args.mode} (Δ={delta:.2g} rad/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def build_parser():
    p = argparse.ArgumentParser(description="Cavity ODE (Euler vs analytic)")
    sub = p.add_subparsers(dest="which", required=True)

    q = sub.add_parser("ode", help="Fabry–Perot cavity")
    q.add_argument("--mode", choices=["ringdown", "stepon"], default="ringdown")
    q.add_argument("--tmax", type=float, default=300e-6)
    q.add_argument("--npts", type=int, default=2000)
    q.add_argument("--delta", type=float, default=0.0, help="detuning [rad/s]")
    q.add_argument("--a0", type=float, default=1.0, help="initial field for ringdown")
    q.add_argument("--s0", type=float, default=1.0, help="input field amplitude for step-on")
    q.add_argument("--length", type=float, default=4000.0, help="cavity length [m]")
    q.add_argument("--T1", type=float, default=3e-6, help="ITM power transmissivity")
    q.add_argument("--T2", type=float, default=3e-6, help="ETM power transmissivity")
    q.add_argument("--Lrt", type=float, default=1e-6, help="round-trip loss")
    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.which == "ode":
        run_ode(args)
