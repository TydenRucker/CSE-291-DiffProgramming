class FreeParticleConfig:
    mass: float
    wall_pos: float
    wall_pos2: float
    wall_stiffness: float
    g : float
def hamiltonian(q: In[Array[float]], p: In[Array[float]], c: In[FreeParticleConfig]) -> float:

    K : float  
    K = K + 0.5 * p[0] * p[0] / c.mass
    K = K + 0.5 * p[1] * p[1] / c.mass
    V : float = 0.0
    if c.wall_pos < q[0]:
        V = 0.5 * c.wall_stiffness * (q[0] - c.wall_pos) * (q[0] - c.wall_pos)
    if c.wall_pos2 > q[1]:
        V = 0.5 * c.wall_stiffness * (q[1] - c.wall_pos2) * (q[1] - c.wall_pos2)
    V = V + c.mass * c.g * q[1]
    return K + V
d_hamiltonian = rev_diff(hamiltonian)
def gradH(q : In[Array[float]], p : In[Array[float]], c : In[FreeParticleConfig],
          dq : Out[Array[float]], dp : Out[Array[float]]):
    d_c : FreeParticleConfig
    d_hamiltonian(q, dq, p, dp, c, d_c, 1.0)
# Example usage
# c = PolarFreeParticleConfig(mass=1.0)
# r, theta = 1.0, math.pi / 4
# p_r, p_theta = 2.0, 3.0