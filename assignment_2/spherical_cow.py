# setting initial conditions
m = 1000 # kg
time = 0
ts = 20
x = 0
y = 100 # height of cow
v = 0 # initial velocity
g = 9.8 # m/s

const = 3 # wind resistnace constant
<<<<<<< Updated upstream

def tot_force_vect(curr_x, curr_y, velocity):
    curr_x = x
    curr_y = y
    velocity = v
=======
theta = np.pi / 4 # angle of the cow falling down
# sadman
def tot_force_vect(position, velocity):
    #curr_x = position[0]
    #curr_y = position[1]
    curr_x_vel = velocity[0]
    curr_y_vel = velocity[1]
>>>>>>> Stashed changes

    #calculate force x acceleration

    x_force = - const * curr_x_vel * abs(curr_x_vel) #alternates the sign with the direction of motion

    #calculate force y acceleration

    y_force = -m*g - const * curr_y_vel * abs(curr_y_vel) #alternates the sign with the direction of motion

    force = (x_force, y_force)

    return force    
'''
    wind = const * velocity**2
    grav = - m * g
    force = wind + grav
<<<<<<< Updated upstream
    return force 

def update_pos(init_pos, velocity, force, ts):
    init_x = x
    init_y = y
    velocity = v
    
    return x,y

=======
    a = 1
    return force
'''
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

    new_x = curr_x + curr_x_vel * time_step

    new_vel_x = curr_x_vel + x_acceleration*time_step

    #calculate force y acceleration

    y_acceleration = y_force / m 

    new_y = curr_y + curr_y_vel * time_step

    new_vel_y = curr_y_vel + y_acceleration*time_step

    #gives out new position and velocity
     
    position = ( new_x , new_y)
    velocity = (new_vel_x , new_vel_y)

    return position , velocity

"""
# nico
>>>>>>> Stashed changes
def calc_forces():
    h = y
    pot = m * g * h
    kin = m * v**2 / 2
    tot = pot + kin
    return pot, kin, tot

<<<<<<< Updated upstream
=======
# nico
for i in smth:
    time += ts
    f = tot_force_vect(pos, v)
    pos, vel = update_pos(pos, v, f)
    if pos(0) & pos(1) =
"""
>>>>>>> Stashed changes
