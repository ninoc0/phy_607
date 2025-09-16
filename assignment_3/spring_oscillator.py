#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
k = 2 # spring constant
x = 8 # initial position
m = 10 # mass
g = 9.8 # gravity

def tot_force_vect(position):
    curr_x = position

    #creating a force vector
    force = -k * curr_x

    return force 

def update_pos(position, force, time_step):
    curr_x = position
    x_force = force

    #calculate force x acceleration
    x_acceleration = x_force / m 

    new_vel_x = x_acceleration*time_step

    new_x = curr_x + new_vel_x * time_step

    #gives out new position and velocity
    position = new_x
    velocity = new_vel_x

    return position , velocity

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
