import numpy as np
# setting initial conditions
m = 1000 # kg
time = 0
ts = 0.1
pos = (0, 100) # inital position
v = (0, 0) # initial velocity
g = 9.8 # m/s
const = 3 # wind resistnace constant

# sadman
def tot_force_vect(position, velocity):
    position = pos
    height = position(1)
    velocity = np.sqrt(2 * g * height)

    wind = const * velocity**2
    grav = - m * g
    force = wind + grav
    a = 1
    return force

# sadman
def update_pos(position, velocity, force):
    position += 
    
    return position, velocity

# nico
def calc_forces():
    h = y
    pot = m * g * h
    kin = m * v**2 / 2
    tot = pot + kin
    return pot, kin, tot

# nico
for i in smth:
    time += ts
    f = tot_force_vect(pos, v)
    pos, vel = update_pos(pos, v, f)
    if pos(0) & pos(1) =
