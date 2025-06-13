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
import numpy as np
with open('examples/loma_code/free_particle_MultiWall.py') as f:
    structs, lib = compiler.compile(f.read(),
                              target = 'c',
                              output_filename = '_code/free_particle_MultiWall')
gradH = lib.gradH
FreeParticleConfig = structs['FreeParticleConfig']

# Start at q = pi /4 and p = 0
q0 = np.array([0.0,0.0], dtype = np.float32)  # initial position x, y
p0 = np.array([2.0,2.0], dtype = np.float32)  # initial momentum px, py
# Mass
m = 1.0
# time step: 0.01
ts = 0.0001
# frame per second
fps = 20

config = FreeParticleConfig(mass = m,wall_pos = 5,wall_pos2=-5,wall_stiffness=1e5,g=2.0)

def solver(t, q, p):
    # given a target time t and q/p, advances
    # q & p and output
    ct = 0
    while True:
        cur_ts = ts
        if ct + cur_ts >= t:
            cur_ts = t - ct
        # sympletic Euler: first advances p, then uses
        # p to advance q
        q_grad = np.zeros_like(q)
        p_grad = np.zeros_like(p)
        gradH(q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              config,
              q_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
              p_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),)
        next_p = p - cur_ts * q_grad
        next_q = q + cur_ts * p_grad
        q = next_q
        p = next_p
        ct += ts
        if ct >= t:
            break
    return ct, q, p

def visualize(fps=60):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.axvline(5)
    ax.axhline(-5)
    point, = ax.plot([], [], 'bo', ms=10)
    trail_line, = ax.plot([], [], 'r-', alpha=0.5)  # create trail line once

    t = 0
    q = q0
    p = p0
    
    trail_x, trail_y = [], []

    def animate(i):
        nonlocal t, q,p

        dt = 1.0 / fps

        # Update x axis
        dt_x, q, p = solver(dt, q, p)
        t += dt

        trail_x.append(q[0])
        trail_y.append(q[1])

        point.set_data([q[0]], [q[1]])
        trail_line.set_data(trail_x, trail_y)  # update trail line data

        return point, trail_line,

    return animation.FuncAnimation(fig, animate, frames=800, interval=1000/fps, blit=True)

anim = visualize()
anim.save('particle_rev_wall_3_w_g.mp4')
plt.show()
