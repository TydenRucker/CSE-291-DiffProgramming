import matplotlib.pyplot as plt
from matplotlib import animation
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import math
with open('examples/loma_code/free_particle_radial.py') as f:
    structs, lib = compiler.compile(f.read(),
                              target = 'c',
                              output_filename = '_code/free_particle_radial')
dHdr = lib.dHdr
dHdtheta = lib.dHdtheta
dHdp_r = lib.dHdp_r
dHdp_theta = lib.dHdp_theta
FreeParticleConfigRadial = structs['FreeParticleConfigRadial']

# Start at q = pi /4 and p = 0
r0 = 1
theta0 = 0.0
p_r0 = 5    # radial momentum (nonzero to spiral outward)
p_theta0 = 10  # initial momentum px, py
# Mass
m = 1.0
# time step: 0.01
ts = 0.01
# frame per second
fps = 20
k = 0
config = FreeParticleConfigRadial(mass = m,k = k)
dp_theta = []
def solver(t, r, theta,p_r,p_theta):
    # given a target time t and q/p, advances
    # q & p and output
    ct = 0
    while True:
        cur_ts = ts
        if ct + cur_ts >= t:
            cur_ts = t - ct
        # sympletic Euler: first advances p, then uses
        # p to advance q
        p_r_next = p_r - cur_ts * dHdr(r, theta, p_r, p_theta,config)
        p_theta_next = p_theta - cur_ts * dHdtheta(r, theta, p_r, p_theta,config)
        r_old = r 
        # Update coordinates using updated momenta
        r_next = r + cur_ts * dHdp_r(r, theta, p_r_next, p_theta_next,config)
        theta_next = theta + cur_ts * dHdp_theta(r, theta, p_r_next, p_theta_next,config)
        r = r_next
        theta = theta_next
        p_r = p_r_next
        p_theta = p_theta_next
        ct += ts
        if ct >= t:
            break
    return ct, r,theta,p_r,p_theta
# t = 0
# r = r0
# theta = theta0
# p_r = p_r0     # radial momentum (nonzero to spiral outward)
# p_theta = p_theta0
# for i in range(100):
#     dt,r, theta, p_r, p_theta = solver(1.0/(i+1), r, theta,p_r,p_theta)
# print(dp_theta)
def visualize(fps=60):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    point, = ax.plot([], [], 'bo', ms=10)
    trail_line, = ax.plot([], [], 'r-', alpha=0.5)  # create trail line once

    t = 0
    r = r0
    theta = theta0
    p_r = p_r0     # radial momentum (nonzero to spiral outward)
    p_theta = p_theta0

    trail_x, trail_y = [], []

    def animate(i):
        nonlocal t, r,theta,p_r,p_theta

        

        # Update x axis
        r = min(r,5)
        dt,r, theta, p_r, p_theta = solver(1.0/fps, r, theta,p_r,p_theta)
        
        print(r)
        x = r * math.cos(theta)
        y = r * math.sin(theta)

        trail_x.append(x)
        trail_y.append(y)
        t += dt

        point.set_data([x], [y])
        trail_line.set_data(trail_x, trail_y)  # update trail line data

        return point, trail_line,

    return animation.FuncAnimation(fig, animate, frames=400, interval=1000/fps, blit=True)

anim = visualize()
anim.save('particle_radial.mp4')
plt.show()
