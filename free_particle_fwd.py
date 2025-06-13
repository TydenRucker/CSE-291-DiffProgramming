class FreeParticleConfig:
    mass: float

def hamiltonian(q: In[float], p: In[float], c: In[FreeParticleConfig]) -> float:

    K : float  = (p * p) / (2 * c.mass)
    return K
d_hamiltonian = fwd_diff(hamiltonian)
# Partial derivatives
def dHdp(q: In[float], p: In[float], c: In[FreeParticleConfig]) -> float:
    d_p : Diff[float]
    d_p.val = p
    d_p.dval = 1
    d_q : Diff[float]
    d_q.val = q
    d_q.dval = 0
    d_c : Diff[FreeParticleConfig]
    d_c.mass.val = c.mass
    return d_hamiltonian(d_q, d_p, d_c).dval

def dHdq(q: In[float], p: In[float], c: In[FreeParticleConfig]) -> float:
    d_p : Diff[float]
    d_p.val = p
    d_p.dval = 0
    d_q : Diff[float]
    d_q.val = q
    d_q.dval = 1
    d_c : Diff[FreeParticleConfig]
    d_c.mass.val = c.mass
    return d_hamiltonian(d_q, d_p, d_c).dval  # should be 0 for a free particle

# Example usage
# c = PolarFreeParticleConfig(mass=1.0)
# r, theta = 1.0, math.pi / 4
# p_r, p_theta = 2.0, 3.0