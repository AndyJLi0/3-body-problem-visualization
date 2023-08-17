import scipy as sci
from scipy import constants
import numpy as np
import scipy.integrate

import matplotlib.pyplot as plt
from matplotlib import animation

# Non-dimensionality constants
m_nd = 1.989e+30  # kg #mass of the sun
r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
v_nd = 30000  # m/s #relative velocity of earth around the sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri

# Net constants
G = constants.G
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd

# Define masses
m1 = 2.1
m2 = 2.5

# Define initial position vectors
r1_init = [0.0, -0.2, 0.3]  # m
r2_init = [0.2, 0.2, 0.2]
r1_init = np.array(r1_init, dtype="float64")
r2_init = np.array(r2_init, dtype="float64")

# Define initial velocities
v1_init = [0.0, -0.2,  -0.3]  # m/s
v2_init = [-0.2, 0.2, 0.1]  # m/s
v1_init = np.array(v1_init, dtype="float64")
v2_init = np.array(v2_init, dtype="float64")

def two_body_function(pos_and_vel, time_const, gravity_const, m1, m2):
    r1 = pos_and_vel[:3]
    r2 = pos_and_vel[3:6]
    v1 = pos_and_vel[6:9]
    v2 = pos_and_vel[9:12]

    r = sci.linalg.norm(r1 - r2)

    dr1_by_dt = K2 * v1
    dr2_by_dt = K2 * v2
    dv1_by_dt = K1 * m2 * (r2 - r1) / r ** 3
    dv2_by_dt = K1 * m1 * (r1 - r2) / r ** 3

    r_derivatives = np.concatenate((dr1_by_dt, dr2_by_dt))
    v_derivatives = np.concatenate((dv1_by_dt, dv2_by_dt))

    return np.concatenate((r_derivatives, v_derivatives))


init_params = np.array([r1_init, r2_init, v1_init, v2_init])  # create array of initial params
init_params = init_params.flatten()  # flatten array to make it 1D
time_span = np.linspace(2, 15, 1500)  # 8 orbital periods and 500 points

two_body_sim = sci.integrate.odeint(two_body_function, init_params, time_span, args=(G, m1, m2))

r1_sol = two_body_sim[:, :3]
r2_sol = two_body_sim[:, 3:6]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Create new arrays for animation
r1_anim = r1_sol[::5, :].copy()
r2_anim = r2_sol[::5, :].copy()

# Set initial marker for planets, that is, blue and red circles at the initial positions
h1 = [ax.scatter(r1_anim[0, 0], r1_anim[0, 1], r1_anim[0, 2], color='deepskyblue', marker="o", s=80,
                 label="Star A")]
h2 = [ax.scatter(r2_anim[0, 0], r2_anim[0, 1], r2_anim[0, 2], color='r', marker="o", s=80,
                 label="Star B")]

# set texts
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
ax.set_zlabel("z", fontsize=14)
ax.set_title("Visualization of Orbits in a Binary Star System\n", fontsize=16)
ax.legend(loc="upper left", fontsize=14)


def update(i, head1, head2):
    # Remove old markers
    h1[0].remove()
    h2[0].remove()

    # Plotting the orbits (for every i, we plot from init pos to final pos)
    t1 = ax.plot(r1_anim[:i, 0], r1_anim[:i, 1], r1_anim[:i, 2], color='deepskyblue')
    t2 = ax.plot(r2_anim[:i, 0], r2_anim[:i, 1], r2_anim[:i, 2], color='r')

    # Plotting the current markers
    h1[0] = ax.scatter(r1_anim[i, 0], r1_anim[i, 1], r1_anim[i, 2], color='deepskyblue', marker="o", s=80)
    h2[0] = ax.scatter(r2_anim[i, 0], r2_anim[i, 1], r2_anim[i, 2], color='r', marker="o", s=80)
    return t1, t2, h1, h2


animate = animation.FuncAnimation(fig, update, frames=4000, interval=2, repeat=False, blit=False, fargs=(h1, h2))
plt.show()
