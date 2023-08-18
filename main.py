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
m1 = 1.1
m2 = 0.907
m3 = 1.425

# Define initial position vectors
r1_init = [-0.5, 1, 0]  # m
r2_init = [0.5, 0, 0.5]
r3_init = [0.2, 1, 1.5]  # m
r1_init = np.array(r1_init, dtype="float64")
r2_init = np.array(r2_init, dtype="float64")
r3_init = np.array(r3_init, dtype="float64")

# Define initial velocities
v1_init = [0.1, 0.1, 0]  # m/s
v2_init = [-0.05, 0, -0.1]
v3_init = [0, -0.01, 0]
v1_init = np.array(v1_init, dtype="float64")
v2_init = np.array(v2_init, dtype="float64")
v3_init = np.array(v3_init, dtype="float64")


def two_body_function(pos_and_vel, time_const, gravity_const, m1, m2, m3):
    r1 = pos_and_vel[:3]
    r2 = pos_and_vel[3:6]
    r3 = pos_and_vel[6:9]
    v1 = pos_and_vel[9:12]
    v2 = pos_and_vel[12:15]
    v3 = pos_and_vel[15:18]

    r12 = sci.linalg.norm(r2 - r1)
    r13 = sci.linalg.norm(r3 - r1)
    r23 = sci.linalg.norm(r3 - r2)

    dr1_by_dt = K2 * v1
    dr2_by_dt = K2 * v2
    dr3_by_dt = K2 * v3
    dv1_by_dt = K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3
    dv2_by_dt = K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3
    dv3_by_dt = K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3

    r_derivatives = np.concatenate((dr1_by_dt, dr2_by_dt))
    r_all_derivatives = np.concatenate((r_derivatives, dr3_by_dt))
    v_derivatives = np.concatenate((dv1_by_dt, dv2_by_dt))
    v_all_derivatives = np.concatenate((v_derivatives, dv3_by_dt))

    return np.concatenate((r_all_derivatives, v_all_derivatives))


init_params = np.array([r1_init, r2_init, r3_init, v1_init, v2_init, v3_init])  # create array of initial params
init_params = init_params.flatten()  # flatten array to make it 1D
time_span = np.linspace(0, 20, 500)  # 8 orbital periods and 500 points

three_body_sim = sci.integrate.odeint(two_body_function, init_params, time_span, args=(G, m1, m2, m3))

r1_sol = three_body_sim[:, :3]
r2_sol = three_body_sim[:, 3:6]
r3_sol = three_body_sim[:, 6:9]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Create new arrays for animation
r1_anim = r1_sol[::1, :].copy()
r2_anim = r2_sol[::1, :].copy()
r3_anim = r3_sol[::1, :].copy()

# Set initial marker for planets, that is, blue and red circles at the initial positions
h1 = [ax.scatter(r1_anim[0, 0], r1_anim[0, 1], r1_anim[0, 2], color='deepskyblue', marker="o", s=80,
                 label="Star A")]
h2 = [ax.scatter(r2_anim[0, 0], r2_anim[0, 1], r2_anim[0, 2], color='r', marker="o", s=80,
                 label="Star B")]
h3 = [ax.scatter(r3_anim[0, 0], r3_anim[0, 1], r3_anim[0, 2], color='goldenrod', marker="o", s=80,
                 label="Star C")]
# set texts
ax.set_xlabel("x", fontsize=14)
ax.set_ylabel("y", fontsize=14)
ax.set_zlabel("z", fontsize=14)
ax.set_title("Visualization of Orbits in a 3 Star System\n", fontsize=16)
ax.legend(loc="upper left", fontsize=14)
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_zlim(-2, 2)

def update(i, head1, head2, head3):
    # Remove old markers
    h1[0].remove()
    h2[0].remove()
    h3[0].remove()

    # Plotting the orbits (for every i, we plot from init pos to final pos)
    t1 = ax.plot(r1_anim[:i, 0], r1_anim[:i, 1], r1_anim[:i, 2], color='deepskyblue')
    t2 = ax.plot(r2_anim[:i, 0], r2_anim[:i, 1], r2_anim[:i, 2], color='r')
    t3 = ax.plot(r3_anim[:i, 0], r3_anim[:i, 1], r3_anim[:i, 2], color='goldenrod')

    # Plotting the current markers
    h1[0] = ax.scatter(r1_anim[i-1, 0], r1_anim[i-1, 1], r1_anim[i-1, 2], color='deepskyblue', marker="o", s=80)
    h2[0] = ax.scatter(r2_anim[i-1, 0], r2_anim[i-1, 1], r2_anim[i-1, 2], color='r', marker="o", s=80)
    h3[0] = ax.scatter(r3_anim[i-1, 0], r3_anim[i-1, 1], r3_anim[i-1, 2], color='goldenrod', marker="o", s=80)

    return t1, t2, t3, h1, h2, h3


animate = animation.FuncAnimation(fig, update, frames=800, interval=10, repeat=False, blit=False, fargs=(h1, h2, h3))
plt.show()
