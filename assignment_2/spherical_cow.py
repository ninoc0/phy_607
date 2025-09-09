import numpy as np
import matplotlib.pyplot as plt

# setting initial conditions
m = 1000 # kg
time = 0
pos = [0,1000]
vel = [1,100]
g = 9.8 # m/s^2

const = 4 # wind resistnace constant

# sadman
def tot_force_vect(velocity):
    curr_x_vel = velocity[0]
    curr_y_vel = velocity[1]

    #calculate force x acceleration
    x_force = - const * curr_x_vel * abs(curr_x_vel) #alternates the sign with the direction of motion

    #calculate force y acceleration
    y_force = -m*g - const * curr_y_vel * abs(curr_y_vel) #alternates the sign with the direction of motion

    #creating a force vector
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

    # U = mgh
    pot = m * g * max(h,0)
    # T = 1/2 mv**2
    kin = m * (x_vel**2 + y_vel**2) / 2

    tot = pot + kin
    return pot, kin, tot

# nico
def testing_timesteps(ts, pos_init, vel_init, time):
    pos = [pos_init[0], pos_init[1]]
    vel = [vel_init[0], vel_init[1]]

    # stores data for graphing
    t_hist = [time]
    x_hist = [pos[0]]
    y_hist = [pos[1]]
    pot_hist, kin_hist, tot_hist = [], [], []
    
    # running the code over and over until the height is 0 
    while pos[1] > 0:
        time +=ts
        f = tot_force_vect(vel)
        pos, vel = update_pos(pos, vel, f, ts)
        pot, kin, tot = calc_energies(pos, vel)

        # updating the stored data
        t_hist.append(time)
        x_hist.append(pos[0])
        y_hist.append(max(pos[1],0))
        pot_hist.append(pot)
        kin_hist.append(kin)
        tot_hist.append(tot)
    return t_hist, x_hist, y_hist, pot_hist, kin_hist, tot_hist

def analytic_solution(x0, y0, vx0, vy0):
    t = np.linspace(0, 28, 500)
    x = x0 + vx0*t
    y = y0 + vy0*t - 0.5*g*t**2
    y = np.maximum(y,0)
    return t,x,y

t_a,x_a,y_a = analytic_solution(pos[0], pos[1], vel[0], vel[1])

## graphing 0 drag analytic vs code
# graphing the trajectory
plt.figure()

# the trajectory gets to a higher precision and has more defined motion as it approaches 0
t_hist, x_hist, y_hist, pot_hist, kin_hist, tot_hist = testing_timesteps(0.001, pos, vel, time)
plt.plot(x_hist, y_hist, label="ts=0.001")
# the analytic solution fits the parabolic shape better and the cow makes it further before hitting the ground
plt.plot(x_a, y_a, "--", label="Analytic Soution")
plt.xlabel("x(m)")
plt.ylabel("y(m)")
plt.title("Trajectory of Cow")
plt.legend()
plt.grid()

# printing output file
f = open("output_ts001.out", "w")
f.write(f"X: {x_hist}\nY: {y_hist}\nTime: {t_hist}")

## changing time steps
plt.figure()
t_hist, x_hist, y_hist, pot_hist, kin_hist, tot_hist = testing_timesteps(0.1, pos, vel, time)
plt.plot(x_hist, y_hist, label="ts=0.1")
f = open("output_ts1.out", "w")
f.write(f"X: {x_hist}\nY: {y_hist}\nTime: {t_hist}")

t_hist, x_hist, y_hist, pot_hist, kin_hist, tot_hist = testing_timesteps(0.01, pos, vel, time)
plt.plot(x_hist, y_hist, label="ts=0.01")
f = open("output_ts01.out", "w")
f.write(f"X: {x_hist}\nY: {y_hist}\nTime: {t_hist}")

t_hist, x_hist, y_hist, pot_hist, kin_hist, tot_hist = testing_timesteps(0.001, pos, vel, time)
plt.plot(x_hist, y_hist, label="ts=0.001")

t_hist, x_hist, y_hist, pot_hist, kin_hist, tot_hist = testing_timesteps(0.0001, pos, vel, time)
plt.plot(x_hist, y_hist, label="ts=0.0001")

t_hist1, x_hist, y_hist, pot_hist, kin_hist, tot_hist = testing_timesteps(0.00001, pos, vel, time)
plt.plot(x_hist, y_hist, label="ts=0.00001")
f = open("output_ts00001.out", "w")
f.write(f"X: {x_hist}\nY: {y_hist}\nTime: {t_hist}")

plt.plot(x_a, y_a, "--", label="Analytic Soution")
f = open("output_analytic.out", "w")
f.write(f"X: {x_a}\nY: {y_a}\nTime: {t_a}")

plt.xlabel("x(m)")
plt.ylabel("y(m)")
plt.title("Trajectory of Cow")
plt.legend()
plt.grid()

t_hist, x_hist, y_hist, pot_hist, kin_hist, tot_hist = testing_timesteps(0.05, pos, vel, time)

# energy is not conserved due to the wind resistance, but since wind resistance is a function of velocity, we "lose" energy more as time progresses
plt.figure()
plt.plot(t_hist[1:], pot_hist, label="Potential")
plt.plot(t_hist[1:], kin_hist, label="Kinetic")
plt.plot(t_hist[1:], tot_hist, label="Total")
plt.xlabel("t (s)")
plt.ylabel("Energy (J)")
plt.title("Energy vs. Time")
plt.legend()
plt.grid()

plt.show()
