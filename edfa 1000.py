import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

h = 6.626e-34  # Planck's constant (J*s)
c = 3.0e8      # Speed of light (m/s)

params = {
    'L': 20.0,                   # Fiber length (m)
    'Nt': 2.0e25,                 # Total Er3+ ion density (ions/m^3)
    'tau21': 10e-3,               # Metastable state lifetime (s)
    'radius': 2e-6,             # Fiber core radius (m)
    'lambda_p': 980e-9,           # Pump wavelength (m)
    'sigma_pa': 1.8e-25,          # Pump absorption cross-section (m^2)
    'Gamma_p': 0.9,               # Pump overlap factor
    'lambda_s': 1550e-9,          # Signal wavelength (m)
    'sigma_sa': 2.14e-25,          # Signal absorption cross-section (m^2)
    'sigma_se': 3.8e-25,          # Signal emission cross-section (m^2)
    'Gamma_s': 0.9 ,               # Signal overlap factor
    'alpha_p': 0.35,
    'alpha_s': 0.25,
}

params['A_core'] = np.pi * params['radius']**2
params['nu_p'] = c / params['lambda_p']
params['nu_s'] = c / params['lambda_s']
params['A21'] = 1 / params['tau21']

def edfa_model(z, P, params):
    P_p, P_s = P
    Nt, A21 = params['Nt'], params['A21']
    A_core = params['A_core']
    sigma_pa, sigma_sa, sigma_se = params['sigma_pa'], params['sigma_sa'], params['sigma_se']
    Gamma_p, Gamma_s = params['Gamma_p'], params['Gamma_s']
    alp_p, alp_s = params['alpha_p'], params['alpha_s'] 
    nu_p, nu_s = params['nu_p'], params['nu_s']
    


    I_p = max(P_p, 0) / A_core
    I_s = max(P_s, 0) / A_core

    R13 = (sigma_pa * I_p) / (h * nu_p)
    W12 = (sigma_sa * I_s) / (h * nu_s)
    W21 = (sigma_se * I_s) / (h * nu_s)

    den = R13 + W12 + W21 + A21 
    N2 = Nt * (R13 + W12) / den
    N1 = Nt - N2

    dPp_dz = (-Gamma_p * sigma_pa * N1 * P_p) - alp_p * P_p
    dPs_dz = (Gamma_s * (sigma_se * N2 - sigma_sa * N1) * P_s) - alp_s * P_s

    return [dPp_dz, dPs_dz]

P_p_in = 1000e-3  
P_s_in = 10e-3    # 1 mW
P0 = [P_p_in, P_s_in]
z_span = [0, params['L']]
z_eval = np.linspace(*z_span, 200)

solution = solve_ivp(edfa_model, z_span, P0, t_eval=z_eval, args=(params,), method='RK45')
if not solution.success:
    raise RuntimeError(solution.message)

z = solution.t
P_p_z, P_s_z = solution.y

P_p_out = P_p_z[-1]
P_s_out = P_s_z[-1]
Gain = P_s_out / P_s_in
Gain_dB = 10 * np.log10(Gain + 1e-30)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(z, P_s_z * 1e3, 'r-', lw=2.5, label='Signal (1550 nm)')
ax1.set_xlabel('Fiber Length (m)', fontsize=22)
ax1.set_ylabel('Signal Power (mW)', color='r', fontsize=22)
ax1.tick_params(axis='y', labelcolor='r',labelsize=18)
ax1.tick_params(axis='x', labelsize=18)
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
ax2.plot(z, P_p_z * 1e3, 'b--', lw=2.5, label='Pump (980 nm)')
ax2.set_ylabel('Pump Power (mW)', color='b', fontsize=22)
ax2.tick_params(axis='y', labelcolor='b', labelsize=18)
ax2.set_ylim(bottom=0)
textstr = (
    f"Input Pump power: {P_p_in*1e3:.2f} mW\n"
    f"Input Signal power: {P_s_in*1e3:.2f} mW\n"

)
props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray')
ax1.text(
    0.6, 0.4, textstr, transform=ax1.transAxes, fontsize=14,
    verticalalignment='baseline', bbox=props
)
fig.suptitle('EDFA Signal and Pump Power Evolution', fontsize=22, fontweight='bold')
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.85), fontsize= 15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
