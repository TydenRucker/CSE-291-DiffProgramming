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
with open('examples/loma_code/free_particle_fwd.py') as f:
    structs, lib = compiler.compile(f.read(),
                              target = 'c',
                              output_filename = '_code/free_particle_fwd')
dHdq = lib.dHdq
dHdp = lib.dHdp
FreeParticleConfig = structs['FreeParticleConfig']

# Start at q = pi /4 and p = 0
q0 = [0.0, 0.0,0.0]  # initial position x, y
p0 = [0.0, 1.0,0.5]  # initial momentum px, py
# Mass
m = 1.0
# time step: 0.01
ts = 0.01
# frame per second
fps = 20

config = FreeParticleConfig(mass = m)

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
    fig = plt.figure() # Create a figure first
    ax = fig.add_subplot(111, projection='3d') # Then add the 3D subplot to it
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20,20)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    point, = ax.plot([], [],[], 'bo', ms=10)
    trail_line, = ax.plot([], [],[], 'r-', alpha=0.5)  # create trail line once

    t = 0
    qx, qy,qz = q0
    px, py,pz = p0

    trail_x, trail_y,trail_z = [], [],[]

    def animate(i):
        nonlocal t, qx, qy, px, py,qz,pz

        dt = 1.0 / fps

        # Update x axis
        dt_x, qx, px = solver(dt, qx, px)
        # Update y axis
        dt_y, qy, py = solver(dt, qy, py)
        #Update z axis
        dt_z, qz, pz = solver(dt, qz, pz)
        t += dt

        trail_x.append(qx)
        trail_y.append(qy)
        trail_z.append(qz)
        point.set_data([qx], [qy])
        point.set_3d_properties([qz])
        trail_line.set_data(trail_x, trail_y)
        trail_line.set_3d_properties(trail_z)# update trail line data

        return point, trail_line,

    return animation.FuncAnimation(fig, animate, frames=400, interval=1000/fps, blit=True)

anim = visualize()
anim.save('particle_fwd_3d.mp4')
plt.show()
