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
with open('examples/loma_code/free_particle_Wall_3d.py') as f:
    structs, lib = compiler.compile(f.read(),
                              target = 'c',
                              output_filename = '_code/free_particle_wall')
gradH = lib.gradH
FreeParticleConfig = structs['FreeParticleConfig']

# Start at q = pi /4 and p = 0
q0 = np.array([0.0,0.0,10.0], dtype = np.float32)  # initial position x, y
p0 = np.array([1.0,1.0,0.0], dtype = np.float32)  # initial momentum px, py
# Mass
m = 1.0
# time step: 0.01
ts = 0.0001
# frame per second
fps = 20
wall_pos = 0
config = FreeParticleConfig(mass = m,wall_pos = wall_pos,wall_stiffness=1e5,g=2.0)

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
    fig = plt.figure() # Create a figure first
    ax = fig.add_subplot(111, projection='3d') # Then add the 3D subplot to it
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-20,20)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    
    # Define the x and y coordinates
    x = np.linspace(-20, 20, 50)
    y = np.linspace(-20, 20, 50)

    # Create a meshgrid
    X, Y = np.meshgrid(x, y)

    # Set z-coordinate to 0
    Z = np.full_like(X,wall_pos)
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.5)
    point, = ax.plot([], [],[], 'ro', ms=10)
    trail_line, = ax.plot([], [],[],'b-', alpha=0.5)  # create trail line once

    t = 0
    q = q0
    p = p0
    
    trail_x, trail_y,trail_z = [], [],[]

    def animate(i):
        nonlocal t, q,p

        dt = 1.0 / fps

        # Update x axis
        dt_x, q, p = solver(dt, q, p)
        t += dt

        trail_x.append(q[0])
        trail_y.append(q[1])
        trail_z.append(q[2])
        point.set_data([q[0]], [q[1]])
        point.set_3d_properties([q[2]])
        trail_line.set_data(trail_x, trail_y)  # update trail line data
        trail_line.set_3d_properties(trail_z)
        return point, trail_line,

    return animation.FuncAnimation(fig, animate, frames=800, interval=1000/fps, blit=True)

anim = visualize()
anim.save('particle_rev_wall_3d_w_g.mp4')
plt.show()
