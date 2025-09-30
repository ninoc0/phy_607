"""
Project 1 main script

ODE Examples(shown in report):
  # Ringdown, no detuning
  python main.py ode --mode ringdown --T1 1e-3 --T2 1e-3 --Lrt 1e-4 --tmax 0.1 --npts 2000

  # Ringdown, detuned
  python main.py ode --mode ringdown --method rk4 --delta 2e5 --tmax 0.1 --npts 200000

  # Step-on, on resonance
  python main.py ode --mode stepon  --method all --tmax 5e-3 --npts 50000 --s0 1.0 --delta 0

  # Step-on, detuned
  python main.py ode --mode stepon  --method all --tmax 1e-3 --npts 20000 --s0 1.0 --delta 1e5

Integral Examples:
  python main.py integral --w 3e-3 --n 2000 --nsamp 25 --rule all
  python main.py integral --w 3e-3 --n 500  --nsamp 40 --rule simpson
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.integrate import solve_ivp

# My methods
from euler_explicit import euler
from runge_kutta import rk4
from euler_semi import euler_semi
from riemann_sum import riemann_midpoint
from simpson import simpson
from trapezoidal import trapz

# ODE Analytic
def cavity_rhs(t, a, kappa, kext, delta, s_in):
    s = s_in(t)
    return (1j * delta - 0.5 * kappa) * a + np.sqrt(kext) * s

def analytic_ringdown(t, a0, kappa, delta):
    return a0 * np.exp((1j * delta - 0.5 * kappa) * t)

def analytic_stepon(t, s0, kappa, kext, delta):
    a_ss = np.sqrt(kext) * s0 / (0.5 * kappa - 1j * delta)
    return a_ss * (1.0 - np.exp((1j * delta - 0.5 * kappa) * t))

# ODE From Functions
def run_ode(args):
    # setting params
    c = 299_792_458.0
    L = args.length
    # defining extra variables, to simply functions
    kappa = (c / (2 * L)) * (args.T1 + args.T2 + args.Lrt)
    kext  = (c / (2 * L)) * args.T1
    delta = args.delta
    # set time as linspace
    t = np.linspace(0.0, args.tmax, args.npts)

    # call which type to run
    if args.mode == "ringdown":
        a0 = args.a0 + 0j
        s_in = (lambda _t: 0.0)
        a_ref = analytic_ringdown(t, a0, kappa, delta)
    else: 
        a0 = 0.0 + 0j
        s_in = (lambda _t: args.s0)
        a_ref = analytic_stepon(t, args.s0, kappa, kext, delta)

    rhs = lambda ti, ai: cavity_rhs(ti, ai, kappa, kext, delta, s_in)

    # numerical solutions
    results = []
    if args.method in ("euler", "all"):
        a_euler = euler(rhs, a0, t)
        results.append(("Euler", a_euler))
    if args.method in ("rk4", "all"):
        a_rk4 = rk4(rhs, a0, t)
        results.append(("RK4", a_rk4))
    if args.method in ("semi", "all"):
        a_semi = euler_semi(rhs, a0, t)
        results.append(("Semi-Implicit Euler", a_semi))
    if args.method in ("scipy", "all"):
        def rhs_scipy(ti, y):
            return rhs(ti, y[0])
        y0 = np.array([a0], dtype=complex)
        sol = solve_ivp(rhs_scipy, (t[0], t[-1]), y0, t_eval=t, method=args.scipy_method, rtol=1e-30, atol=1e-30)
        a_scipy = sol.y[0]
        label = f"SciPy {args.scipy_method} |a|^2"
        results.append((label, a_scipy))

    # power v. time plot
    P_ref = np.abs(a_ref)**2
    plt.figure()
    plt.plot(t * 1e6, P_ref, label="Analytic", linewidth=2)
    for label, a in results:
        plt.plot(t * 1e6, np.abs(a)**2, "--", label=label)
    plt.xlabel("time [µs]")
    plt.ylabel("Circulating Power |a|² [arb]")
    plt.title(f"Cavity {args.mode} (Δ={delta:.2g} rad/s)")
    plt.legend()
    plt.tight_layout()

    # error plots
    plt.figure()
    tiny = 1e-30 # stops plotting errors due to semilogy if error = 0 
    for label, a in results:
        err = np.abs(a)**2 - P_ref
        plt.semilogy(t * 1e6, np.maximum(np.abs(err), tiny), label=label)
    plt.xlabel("time [µs]")
    plt.ylabel("Absolute error in |a|²")
    plt.title("ODE power error vs analytic")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Integral analytic
def gaussian_intensity(r, w):
    return np.exp(-2.0 * (r * r) / (w * w))

def clipping_fraction_analytic(a, w):
    return 1.0 - np.exp(-2.0 * a * a / (w * w))

# numerical functions
def clipping_fraction_numeric(a, w, n=2000, rule="simpson"):
    Rmax = 6.0 * w # prevents integration to infinity, full response is in here already
    def integrand(r):
        return 2.0 * np.pi * r * gaussian_intensity(r, w)
    
    # calling function type
    if rule == "riemann":
        num = riemann_midpoint(integrand, 0.0, a, n)
        den = riemann_midpoint(integrand, 0.0, Rmax, n)
    elif rule == "trapz":
        num = trapz(integrand, 0.0, a, n)
        den = trapz(integrand, 0.0, Rmax, n)
    elif rule == "simpson":
        num = simpson(integrand, 0.0, a, n)
        den = simpson(integrand, 0.0, Rmax, n)
    else:
        raise ValueError("rule must be 'riemann', 'trapz', or 'simpson'")

    return num / den

def run_integral(args):
    a_sel, w, n, rule, nsamp = args.a, args.w, args.n, args.rule, args.nsamp

    a_grid = np.linspace(0.0, 2.0 * w, 400)
    F_curve = 1.0 - np.exp(-2.0 * a_grid * a_grid / (w * w)) # analytic response

    a_samples = np.linspace(0.0, 2.0 * w, nsamp)

    def sample_rule(rule_name):
        return np.array([clipping_fraction_numeric(ai, w, n=n, rule=rule_name) for ai in a_samples])

    numeric_sets = []
    if rule == "all":
        numeric_sets.append(("riemann", sample_rule("riemann"), "x"))
        numeric_sets.append(("trapz",   sample_rule("trapz"),   "o"))
        numeric_sets.append(("simpson", sample_rule("simpson"), "s"))
    else:
        numeric_sets.append((rule, sample_rule(rule), "x"))

    F_ana_at_a = clipping_fraction_analytic(a_sel, w)
    chosen_rule = "simpson" if rule == "all" else rule # will only print simpson error if all is selected
    F_num_at_a = clipping_fraction_numeric(a_sel, w, n=n, rule=chosen_rule)
    rel_err = abs(F_num_at_a - F_ana_at_a) / (F_ana_at_a if F_ana_at_a != 0 else 1.0)

    # printing error taken at 1 point to compare
    print(f"Selected a = {a_sel:.6g} m, w = {w:.6g} m")
    print(f"Analytic F(a): {F_ana_at_a:.8f}")
    print(f"Numeric  F(a): {F_num_at_a:.8f}   (rule={chosen_rule}, n={n})")
    print(f"Rel. error   : {rel_err:.3e}")

    # plotting fraction power v radius
    plt.figure()
    plt.plot(a_grid * 1e3, F_curve, label="Analytic", linewidth=2)
    for name, F_vals, marker in numeric_sets:
        plt.scatter(a_samples * 1e3, F_vals, marker=marker, s=40, label=f"Numeric ({name})")
    plt.scatter([a_sel * 1e3], [F_num_at_a], marker="*", s=120,
                label=f"F(a) at a={a_sel*1e3:.2f} mm", zorder=5)
    plt.xlabel("Aperture radius a [mm]")
    plt.ylabel("Enclosed power fraction F(a)")
    plt.title("TEM00 clipping fraction")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()

    # error
    plt.figure()
    tiny = 1e-18
    F_curve_interp = 1.0 - np.exp(-2.0 * a_samples * a_samples / (w * w))
    for name, F_vals, marker in numeric_sets:
        err = np.maximum(np.abs(F_vals - F_curve_interp), tiny)
        plt.semilogy(a_samples * 1e3, err, marker=marker, linestyle="none", label=f"{name}")
    plt.xlabel("Aperture radius a [mm]")
    plt.ylabel("Absolute error |F_num - F_analytic|")
    plt.title(f"Integral error vs analytic (n={n})")
    plt.legend()
    plt.tight_layout()
    plt.show()

# interactive calls
def build_parser():
    p = argparse.ArgumentParser(description="Fabry–Perot cavity ODE + Gaussian clipping integral")
    sub = p.add_subparsers(dest="which", required=True)

    # ODE
    q = sub.add_parser("ode", help="Fabry–Perot cavity")
    q.add_argument("--mode", choices=["ringdown", "stepon"], default="ringdown")
    q.add_argument("--method", choices=["euler", "rk4", "semi", "scipy", "all"], default="all")
    q.add_argument("--scipy-method", choices=["RK45", "Radau", "BDF", "DOP853"], default="RK45",
                   help="SciPy solve_ivp method")
    q.add_argument("--tmax", type=float, default=300e-6)
    q.add_argument("--npts", type=int, default=2000)
    q.add_argument("--delta", type=float, default=0.0, help="detuning [rad/s]")
    q.add_argument("--a0", type=float, default=1.0, help="initial field for ringdown")
    q.add_argument("--s0", type=float, default=1.0, help="input field amplitude for step-on")
    # changing mirror params
    q.add_argument("--length", type=float, default=4000.0, help="cavity length [m]")
    q.add_argument("--T1", type=float, default=3e-6, help="ITM power transmissivity")
    q.add_argument("--T2", type=float, default=3e-6, help="ETM power transmissivity")
    q.add_argument("--Lrt", type=float, default=1e-6, help="round-trip loss")

    # Integral
    r = sub.add_parser("integral", help="Gaussian TEM00 clipping integral")
    r.add_argument("--a", type=float, default=5e-3, help="aperture radius [m]")
    r.add_argument("--w", type=float, default=3e-3, help="beam radius w [m]")
    r.add_argument("--n", type=int, default=2000, help="number of panels per integral")
    r.add_argument("--nsamp", type=int, default=25, help="number of aperture samples to plot")
    r.add_argument("--rule", choices=["riemann", "trapz", "simpson", "all"], default="simpson")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.which == "ode":
        run_ode(args)
    elif args.which == "integral":
        run_integral(args)
