import numpy as np
import matplotlib.pyplot as plt

# setting initial conditions
m = 1000 # kg
time = 0
ts = 0.05
pos = [0,300]
vel = [10,-5]
g = 9.8 # m/s^2

const = 3 # wind resistnace constant

# sadman
def tot_force_vect(velocity):
    curr_x_vel = velocity[0]
    curr_y_vel = velocity[1]

    #calculate force x acceleration

    x_force = - const * curr_x_vel * abs(curr_x_vel) #alternates the sign with the direction of motion

    #calculate force y acceleration

    y_force = -m*g - const * curr_y_vel * abs(curr_y_vel) #alternates the sign with the direction of motion

    force = (x_force, y_force)

    return force    

# sadman
def update_pos(position, velocity, force, time_step):
    curr_x = position[0]
    curr_y = position[1]
    curr_x_vel = velocity[0]
    curr_y_vel = velocity[1]
    x_force = force[0]
    y_force = force[1]

    #calculate force x acceleration

    x_acceleration = x_force / m 

    new_vel_x = curr_x_vel + x_acceleration*time_step

    new_x = curr_x + new_vel_x * time_step
    #calculate force y acceleration

    y_acceleration = y_force / m 

    new_vel_y = curr_y_vel + y_acceleration*time_step

    new_y = curr_y + new_vel_y * time_step
    #gives out new position and velocity
     
    position = ( new_x , new_y)
    velocity = (new_vel_x , new_vel_y)

    return position , velocity


# nico
def calc_energies(position, velocity):
    h = position[1]
    x_vel = velocity[0]
    y_vel = velocity[1]

    pot = m * g * max(h,0)

    kin = m * (x_vel**2 + y_vel**2) / 2

    tot = pot + kin
    return pot, kin, tot

t_hist = [time]
x_hist = [pos[0]]
y_hist = [pos[1]]
pot_hist, kin_hist, tot_hist = [], [], []

# nico
while pos[1] > 0:
    time +=ts
    f = tot_force_vect(vel)
    pos, vel = update_pos(pos, vel, f, ts)
    pot, kin, tot = calc_energies(pos, vel)

    t_hist.append(time)
    x_hist.append(pos[0])
    y_hist.append(max(pos[1],0))
    pot_hist.append(pot)
    kin_hist.append(kin)
    tot_hist.append(tot)

plt.figure()
plt.plot(x_hist, y_hist)
plt.xlabel("x(m)")
plt.ylabel("y(m)")
plt.title("Trajectory of Cow")
plt.grid()

plt.show()
