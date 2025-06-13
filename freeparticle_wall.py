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
with open('examples/loma_code/free_particle_Wall.py') as f:
    structs, lib = compiler.compile(f.read(),
                              target = 'c',
                              output_filename = '_code/free_particle_wall')
dHdq = lib.dHdq
dHdp = lib.dHdp
FreeParticleConfig = structs['FreeParticleConfig']

# Start at q = pi /4 and p = 0
q0 = [0.0,0.0]  # initial position x, y
p0 = [1,1]  # initial momentum px, py
# Mass
m = 1.0
# time step: 0.01
ts = 0.0001
# frame per second
fps = 20

config = FreeParticleConfig(mass = m,wall_pos = 5,wall_stiffness=1e5)

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
        next_p = p - cur_ts * dHdq(q, p, config)
        next_q = q + cur_ts * dHdp(q, next_p, config)
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
    point, = ax.plot([], [], 'bo', ms=10)
    trail_line, = ax.plot([], [], 'r-', alpha=0.5)  # create trail line once

    t = 0
    qx,qy = q0
    px,py = p0
    
    trail_x, trail_y = [], []

    def animate(i):
        nonlocal t, qx, qy, px,py

        dt = 1.0 / fps

        # Update x axis
        dt_x, qx, px = solver(dt, qx, px)
        dt_y, qy, py = solver(dt, qy, py)
        t += dt

        trail_x.append(qx)
        trail_y.append(qy)

        point.set_data([qx], [qy])
        trail_line.set_data(trail_x, trail_y)  # update trail line data

        return point, trail_line,

    return animation.FuncAnimation(fig, animate, frames=800, interval=1000/fps, blit=True)

anim = visualize()
anim.save('particle_fwd_wall_bug.mp4')
plt.show()
