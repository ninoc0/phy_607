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

#%%
## calculating for the real solution
def tot_force_vect(position):
    curr_x = position
    force = -k * curr_x
    return force 

def real(position, velocity, force, time):
    curr_x = position
    curr_v = velocity
    x_acceleration = force / m 

    new_x = curr_x*np.cos(np.sqrt(k/m)*time) + (curr_v/np.sqrt(k/m))*np.sin(np.sqrt(k/m)*time)
    new_v = -curr_x*np.sqrt(k/m)*np.sin(np.sqrt(k/m)*time) + curr_v*np.cos(np.sqrt(k/m)*time)

    position = new_x
    velocity = new_v
    return position , velocity

## calculating eucler explicit
def calc_energies(position, velocity):
    h = position
    x_vel = velocity

    # U = mgh
    pot = m * g * h
    # T = 1/2 mv**2
    kin = m * (x_vel**2) / 2

    tot = pot + kin
    return pot, kin, tot

def testing_timesteps(ts, pos_init, time):
    pos = pos_init
    vel = 0

    # stores data for graphing
    t_hist = [time]
    x_hist = [pos]
    pot_hist, kin_hist, tot_hist = [], [], []
    
    # running the code over and over until the height is 0 
    while time < 40:
        time +=ts
        f = tot_force_vect(pos)
        pos, vel = update_pos(pos, f, ts)
        pot, kin, tot = calc_energies(pos, vel)

        # updating the stored data
        t_hist.append(time)
        x_hist.append(pos)
        pot_hist.append(pot)
        kin_hist.append(kin)
        tot_hist.append(tot)
    return t_hist, x_hist, pot_hist, kin_hist, tot_hist

plt.figure()

# the trajectory gets to a higher precision and has more defined motion as it approaches 0
t_hist, x_hist, pot_hist, kin_hist, tot_hist = testing_timesteps(0.001, x, 0)
plt.plot(t_hist, x_hist, label="ts=0.001")

plt.xlabel("time")
plt.ylabel("y(m)")
plt.title("Spring")
plt.legend()
plt.grid()
# %%
