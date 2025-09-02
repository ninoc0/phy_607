# setting initial conditions
m = 1000 # kg
time = 0
ts = 20
x = 0
y = 100 # height of cow
v = 0 # initial velocity
g = 9.8 # m/s

const = 3 # wind resistnace constant

def tot_force_vect(curr_x, curr_y, velocity):
    curr_x = x
    curr_y = y
    velocity = v

    wind = const * velocity**2
    grav = - m * g
    force = wind + grav
    return force 

def update_pos(init_pos, velocity, force, ts):
    init_x = x
    init_y = y
    velocity = v
    
    return x,y

def calc_forces():
    h = y
    pot = m * g * h
    kin = m * v**2 / 2
    tot = pot + kin
    return pot, kin, tot

