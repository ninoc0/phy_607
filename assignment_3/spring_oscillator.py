#%%
## importing libraries
import numpy as np
import matplotlib.pyplot as plt
#%%
## constants
k = 2 # spring constant
x0 = 8 # initial position
m = 10 # mass
v0 = 0 # initial velocity
omega = np.sqrt(k/m) # angular freq
T = 2*np.pi/omega # period

#%%
# force vect
def force(x):
    return -k * x
# adding acceleration bc we use it a lot
def accel(x):
    return force(x) / m

# calcauting real solution
def analytic_solution(t, x0, v0):
    x = x0*np.cos(omega*t) + (v0/omega)*np.sin(omega*t)
    v = -x0*omega*np.sin(omega*t) + v0*np.cos(omega*t)
    return x, v

## calculating eucler explicit
def euler_explicit(x0, v0, ts, steps):
    x = [x0]
    v = [v0]
    for i in range(steps):
        a = accel(x[i])
        curr_x = x[i] + ts*v[i]
        curr_v = v[i] + ts*a
        x.append(curr_x)
        v.append(curr_v)
    return np.array(x), np.array(v)

## calculating euler sympletic
def euler_sympletic(x0, v0, ts, steps):
    x = [x0]
    v = [v0]
    for i in range(steps):
        curr_v = v[i] + ts*accel(x[i])
        curr_x = x[i] + ts*curr_v
        
        x.append(curr_x)
        v.append(curr_v)
    return np.array(x), np.array(v)

## second order rk method
def rk2(x0, v0, ts, steps):
    x = [x0]
    v = [v0]
    for i in range(steps):
        k1x = v[i]
        k1v = accel(x[i])

        x_mid = x[i] + 0.5*ts*k1x
        v_mid = v[i] + 0.5*ts*k1v

        k2x = v_mid
        k2v = accel(x_mid)

        curr_x = x[i] + 0.5*ts*k2x
        curr_v = v[i] + 0.5*ts*k2v

        x.append(curr_x)
        v.append(curr_v)
    return np.array(x), np.array(v)
#%% 
## run simulations
def simulate(periods):
    ts = T/100
    steps = int(periods * 100)
    t = np.linspace(0, periods*T, steps+1)

    x, v = analytic_solution(t,x0,v0)
    x_eu, v_eu = euler_explicit(x0,v0,ts,steps)
    x_es, v_es = euler_sympletic(x0, v0, ts, steps)
    x_rk, v_rk = rk2(x0, v0, ts, steps)

    return t, (x,v), (x_eu, v_eu), (x_es, v_es), (x_rk,v_rk)

#%%
# plotting functions
def plot_positions(periods):
    t, (x,v), (x_eu, v_eu), (x_es, v_es), (x_rk, v_rk)= simulate(periods)
    plt.figure(figsize=(10, 5))
    plt.plot(t/T, x, label="Analytic")
    plt.plot(t/T, x_eu,   label="Euler explicit")
    plt.plot(t/T, x_es, '--', color='r', label="Euler symplectic")
    plt.plot(t/T, x_rk, label="RK2")
    plt.xlabel("Time [periods]")
    plt.ylabel("Position [m]")
    plt.title(f"Simple Harmonic Oscilations ({periods} periods)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
def energies(position, velocity):
    x = position
    v = velocity
    return 0.5*m*v**2 + 0.5*k*x**2

def plot_energies(periods):
    t, (x, v), (x_eu, v_eu), (x_es, v_es), (x_rk, v_rk) = simulate(periods)
    E0 = energies(np.array([x0]), np.array([v0]))[0]

    plt.figure(figsize=(10, 5))
    plt.plot(t/T, energies(x, v)/E0, label="Analytic")
    plt.plot(t/T, energies(x_eu, v_eu)/E0,     label="Euler explicit")
    plt.plot(t/T, energies(x_es, v_es)/E0,   label="Euler symplectic")
    plt.plot(t/T, energies(x_rk, v_rk)/E0,   label="RK2")
    plt.xlabel("Time [periods]")
    plt.ylabel("Energy / E(0)")
    plt.title(f"Energy ({periods} periods)")
    plt.legend()
    plt.grid(alpha=0.3)

plot_positions(5)
plot_positions(15)
plot_energies(5)
plot_energies(15)

plt.show()
# %%
