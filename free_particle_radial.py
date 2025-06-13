class FreeParticleConfigRadial:
    mass: float
    k: float

def hamiltonian(r: In[float], theta: In[float],p_r: In[float],p_theta: In[float], c: In[FreeParticleConfigRadial]) -> float:

    K : float  = (0.5 / c.mass) * ((pow(p_r,2)) + ((pow(p_theta,2)) / (pow(r,2))))
    U : float  = 0.5 * c.k * pow(r,2)
    return K + U
d_hamiltonian = fwd_diff(hamiltonian)
# Partial derivatives
def dHdr(r: In[float], theta: In[float],p_r: In[float],p_theta: In[float], c: In[FreeParticleConfigRadial]) -> float:
    d_r : Diff[float]
    d_r.val = r
    d_r.dval = 1
    d_theta : Diff[float]
    d_theta.val = theta
    d_theta.dval = 0
    d_p_r : Diff[float]
    d_p_r.val = p_r
    d_p_r.dval = 0
    d_p_theta : Diff[float]
    d_p_theta.val = p_theta
    d_p_theta.dval = 0
    d_c : Diff[FreeParticleConfigRadial]
    d_c.mass.val = c.mass
    d_c.k.val = c.k
    return d_hamiltonian(d_r, d_theta,d_p_r,d_p_theta, d_c).dval  # should be 0 for a free particle
def dHdtheta(r: In[float], theta: In[float],p_r: In[float],p_theta: In[float], c: In[FreeParticleConfigRadial]) -> float:
    d_r : Diff[float]
    d_r.val = r
    d_r.dval = 0
    d_theta : Diff[float]
    d_theta.val = theta
    d_theta.dval = 1
    d_p_r : Diff[float]
    d_p_r.val = p_r
    d_p_r.dval = 0
    d_p_theta : Diff[float]
    d_p_theta.val = p_theta
    d_p_theta.dval = 0
    d_c : Diff[FreeParticleConfigRadial]
    d_c.mass.val = c.mass
    d_c.k.val = c.k
    return d_hamiltonian(d_r, d_theta,d_p_r,d_p_theta, d_c).dval  # should be 0 for a free particle
def dHdp_r(r: In[float], theta: In[float],p_r: In[float],p_theta: In[float], c: In[FreeParticleConfigRadial]) -> float:
    d_r : Diff[float]
    d_r.val = r
    d_r.dval = 0
    d_theta : Diff[float]
    d_theta.val = theta
    d_theta.dval = 0
    d_p_r : Diff[float]
    d_p_r.val = p_r
    d_p_r.dval = 1
    d_p_theta : Diff[float]
    d_p_theta.val = p_theta
    d_p_theta.dval = 0
    d_c : Diff[FreeParticleConfigRadial]
    d_c.mass.val = c.mass
    d_c.k.val = c.k
    return d_hamiltonian(d_r, d_theta,d_p_r,d_p_theta, d_c).dval  # should be 0 for a free particle
def dHdp_theta(r: In[float], theta: In[float],p_r: In[float],p_theta: In[float], c: In[FreeParticleConfigRadial]) -> float:
    d_r : Diff[float]
    d_r.val = r
    d_r.dval = 0
    d_theta : Diff[float]
    d_theta.val = theta
    d_theta.dval = 0
    d_p_r : Diff[float]
    d_p_r.val = p_r
    d_p_r.dval = 0
    d_p_theta : Diff[float]
    d_p_theta.val = p_theta
    d_p_theta.dval = 1
    d_c : Diff[FreeParticleConfigRadial]
    d_c.mass.val = c.mass
    d_c.k.val = c.k
    return d_hamiltonian(d_r, d_theta,d_p_r,d_p_theta, d_c).dval  # should be 0 for a free particle
# Example usage
# c = PolarFreeParticleConfigRadial(mass=1.0)
# r, theta = 1.0, math.pi / 4
# p_r, p_theta = 2.0, 3.0