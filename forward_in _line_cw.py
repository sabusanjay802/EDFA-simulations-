import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


h = 6.62607015e-34   # Planck's constant (J*s)
c = 299792458        # speed of light (m/s)
L = 5.0                      # fiber length [m]
Nt = 1.0e25                   # Er3+ ion density [m^-3]
tau = 10e-3                   # upper-state lifetime [s]

# Effective core area and overlap factors
A_eff = 12.56e-12                # effective area [m^2] (50 µm^2)
Gamma_p = 0.9                 # overlap factor for pump
Gamma_s = 0.9                 # overlap factor for signal

# Wavelengths (pump and signal)
lambda_p = 980e-9             # pump wavelength [m]
lambda_s = 1550e-9            # signal wavelength [m]
nu_p = c / lambda_p
nu_s = c / lambda_s

# Cross sections (typical order-of-magnitude values, in m^2)
sigma_ap = 1.8e-25            # pump absorption cross-section
sigma_ep = 3.15e-25            # pump emission cross-section (often small for 980)
sigma_as = 2.14e-25            # signal absorption cross-section
sigma_es = 3.80e-25            # signal emission cross-section

# Background loss (fiber intrinsic) [1/m]
alpha_p = 0.04                # pump loss 
alpha_s = 0.02                # signal loss

# Input (fixed) signal
P_sig_in = 10e-3   #p Power in Watts

# Spatial discretization options for solver
z_span = (0.0, L)

# --------------------------
# Helper functions
# --------------------------
def db_to_linear(dB):
    return 10**(dB / 10.0)

def linear_to_dBm(P_watt):
    # avoid log of zero
    P_watt = np.maximum(P_watt, 1e-25)
    return 10.0 * np.log10(P_watt) + 30.0

def compute_N2_from_photon_flux(phi_p, phi_s):
    W_p_abs = Gamma_p * sigma_ap * phi_p
    W_p_ems = Gamma_p * sigma_ep * phi_p
    W_s_abs = Gamma_s * sigma_as * phi_s
    W_s_ems = Gamma_s * sigma_es * phi_s

    A = W_p_abs + W_s_abs
    B = 1.0 / tau + W_p_ems + W_s_ems

    # avoid division by zero
    denom = (A + B)
    # If denom is zero (extreme), set N2 to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        N2 = Nt * (A / denom)
    # Clip between 0 and Nt
    N2 = np.clip(N2, 0.0, Nt)
    return N2

# --------------------------
# ODE system
# --------------------------
def edfa_odes(z, y):
    """
    y[0] : P_pump(z) [W]
    y[1] : P_signal(z) [W]
    returns dP_p/dz, dP_s/dz
    """
    Pp = max(y[0], 0.0)
    Ps = max(y[1], 0.0)

    # Photon fluxes [photons / (m^2 * s)]
    # phi = P / (h * nu * A_eff)
    phi_p = Pp / (h * nu_p * A_eff)
    phi_s = Ps / (h * nu_s * A_eff)

    # Steady-state excited population N2
    N2 = compute_N2_from_photon_flux(phi_p, phi_s)
    N1 = Nt - N2

    # Net gain/abs coefficients (1/m)
    # For pump: negative sign typically because pump is absorbed (we model net change)
    # dPp/dz = -Gamma_p*(sigma_ap*N1 - sigma_ep*N2)*Pp - alpha_p*Pp
    dPp_dz = -Gamma_p * (sigma_ap * N1 - sigma_ep * N2) * Pp - alpha_p * Pp

    # For signal: amplification/absorption
    # dPs/dz =  Gamma_s*(sigma_es*N2 - sigma_as*N1)*Ps - alpha_s*Ps
    dPs_dz =  Gamma_s * (sigma_es * N2 - sigma_as * N1) * Ps - alpha_s * Ps

    return [dPp_dz, dPs_dz]

# --------------------------
# Sweep over pump powers and solve ODE for each
# --------------------------
pump_mw_array = np.linspace(1.0, 500.0, 60)   # pump powers in mW to sweep (1 mW .. 500 mW)
pump_w_array = pump_mw_array * 1e-3          # convert to Watts

Pout_signal_list = []

for Pp_in in pump_w_array:
    # initial conditions: forward co-directional => both injected at z=0
    y0 = [Pp_in, P_sig_in]

    # Solve ODEs
    sol = solve_ivp(edfa_odes, z_span, y0, method='RK45', atol=1e-8, rtol=1e-6)

    Pp_out = float(sol.y[0, -1])
    Ps_out = float(sol.y[1, -1])
    Pout_signal_list.append(Ps_out)

# Convert to numpy arrays
Pout_signal_array = np.array(Pout_signal_list)

# Convert outputs to dBm for plotting
Pout_signal_dBm = linear_to_dBm(Pout_signal_array)

# --------------------------
# Plotting
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(pump_mw_array, Pout_signal_dBm, marker='o', linewidth=1.6)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("Input Pump Power (mW)", fontsize=22)
plt.ylabel("Output Signal Power (dBm)", fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title(f"Forward (Co-directional) EDFA: Output Signal power vs Pump power (L={L} m,  Sig_Power ={P_sig_in} W)",fontsize=19)
plt.tight_layout()
plt.show()

# Also print a small table for reference
for pmw, pout_w, pout_dbm in zip(pump_mw_array[::10], Pout_signal_array[::10], Pout_signal_dBm[::10]):
    print(f"Pump: {pmw:6.1f} mW -> Signal out: {pout_w*1e3:7.3f} mW   ({pout_dbm:6.2f} dBm)")
