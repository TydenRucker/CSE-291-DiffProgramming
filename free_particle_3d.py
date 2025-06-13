class FreeParticleConfigRadial:
    mass: float
    k: float

def hamiltonian(p: In[Array[float]],q: In[Array[float]] c: In[FreeParticleConfigRadial]) -> float:

    K : float  = (0.5 / c.mass) * ((pow(p[0],2)) + ((pow(p[1],2)) / (pow(p[2],2))))
    return K
d_hamiltonian = fwd_diff(hamiltonian)
# Partial derivatives
def dHdp(p: In[Array[float]], c: In[FreeParticleConfigRadial]) -> float:
    d_p : Diff[Array[float]]
    d_p[0].val = p[0]
    d_p[0].dval = 1
    d_p[1].val = p[1]
    d_p[1].dval = 1
    d_p[2].val = p[2]
    d_p[2].dval = 1
    d_c : Diff[FreeParticleConfigRadial]
    d_c.mass.val = c.mass
    d_c.k.val = c.k
    return d_hamiltonian(d_p, d_q,d_c).dval  # should be 0 for a free particle
def dHdq(p: In[Array[float]],q: In[Array[float]] c: In[FreeParticleConfigRadial]) -> float:
    d_p : Diff[Array[float]]
    d_p[0].val = p[0]
    d_p[0].dval = 0
    d_p[1].val = p[1]
    d_p[1].dval = 0
    d_p[2].val = p[2]
    d_p[2].dval = 0
    d_q : Diff[Array[float]]
    d_q[0].val = q[0]
    d_q[0].dval = 1
    d_q[1].val = q[1]
    d_q[1].dval = 1
    d_q[2].val = q[2]
    d_q[2].dval = 1
    d_c : Diff[FreeParticleConfigRadial]
    d_c.mass.val = c.mass
    d_c.k.val = c.k
    return d_hamiltonian(d_p, d_q,d_c).dval  # should be 0 for a free particle
# Example usage
# c = PolarFreeParticleConfigRadial(mass=1.0)
# r, theta = 1.0, math.pi / 4
# p_r, p_theta = 2.0, 3.0