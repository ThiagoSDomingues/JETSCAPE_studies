import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import i0, k1

m_pi = 0.1396

def blastwave_integrand(r, pt, m, T, beta_s, n):
    beta = beta_s * r**n
    rho = np.arctanh(beta)
    mt = np.sqrt(m**2 + pt**2)
    xip = pt * np.sinh(rho) / T
    xim = mt * np.cosh(rho) / T
    return r * mt * i0(xip) * k1(xim)  # Normalized yield ~ integ * 2pi pt

def blastwave(pt, T, beta_s, n):
    res = np.zeros_like(pt)
    for i, pt_i in enumerate(pt):
        integ, _ = quad(blastwave_integrand, 0, 1, args=(pt_i, m_pi, T, beta_s, n))
        res[i] = integ
    return res

# Base params (from prior fit)
T0, beta_s0, n0 = 0.12, 0.70, 1.0
sigma_T, sigma_beta = 0.0012, 0.0098  # From ALICE v0 fits [web:30]
N_events = 1000  # MC samples

pt = np.linspace(0.3, 3.0, 30)

# Generate fluctuating params
T_fluc = np.random.normal(T0, sigma_T, N_events)
beta_fluc = np.random.normal(beta_s0, sigma_beta, N_events)

# Compute yields for all events (vectorized approx; full is N_events x len(pt))
yields = np.zeros((N_events, len(pt)))
for i in range(N_events):
    yields[i] = blastwave(pt, T_fluc[i], beta_fluc[i], n0)

mean_y = np.mean(yields, axis=0)
var_y = np.var(yields, axis=0)
v0_pt = np.sqrt(var_y) / mean_y

# Integrated v0 ~ average over pT (weighted)
v0_int = np.average(v0_pt, weights=mean_y * pt)  # Approx <v0>

# Scaling: compute <pT> from mean_y (as before)
mean_pt_num = np.trapz(pt * mean_y * pt, pt)  # ~ int p dN/dp dp
mean_pt_den = np.trapz(mean_y * pt, pt)
mean_pt = mean_pt_num / mean_pt_den
xt = pt / mean_pt

v0_scaled = v0_pt / v0_int

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(pt, v0_pt, 'b-', label='v_0(p_T)')
ax1.axhline(v0_int, color='r', ls='--', label=f'v_0 (int) = {v0_int:.4f}')
ax1.set_xlabel('p_T (GeV/c)'); ax1.set_ylabel('v_0'); ax1.legend()

ax2.plot(xt, v0_scaled, 'g-', linewidth=2, label='BW Model')
ax2.set_xlabel('x_T'); ax2.set_ylabel('v_0(p_T)/v_0'); ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 3.5)

plt.tight_layout()
plt.show()

print(f'Integrated v_0: {v0_int:.4f}, Mean p_T: {mean_pt:.3f} GeV/c')