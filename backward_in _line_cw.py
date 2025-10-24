import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


h = 6.62607015e-34   # Planck's constant (J*s)
c = 299792458        # speed of light (m/s)
L = 5.0                      # fiber length [m]
Nt = 1.0e25                   # Er3+ ion density [m^-3]
tau = 10e-3                   # upper-state lifetime [s]

A_eff = 12.56e-12             # effective area [m^2] (50 µm^2)
Gamma_p = 0.9                 # overlap factor for pump
Gamma_s = 0.9                 # overlap factor for signal

lambda_p = 980e-9             # pump wavelength [m]
lambda_s = 1550e-9            # signal wavelength [m]
nu_p = c / lambda_p
nu_s = c / lambda_s

sigma_ap = 1.8e-25            # pump absorption cross-section [m^2]
sigma_ep = 3.15e-25           # pump emission cross-section [m^2]
sigma_as = 2.14e-25           # signal absorption cross-section [m^2]
sigma_es = 3.80e-25           # signal emission cross-section [m^2]

alpha_p = 0.04                # pump background loss [1/m]
alpha_s = 0.02                # signal background loss [1/m]

# Input signal (fixed) (injected at z = 0)
P_sig_in = 10e-3   # Watts

# z mesh for solver
z_mesh = np.linspace(0.0, L, 200)

# --------------------------
# Utility functions
# --------------------------
def linear_to_dBm(P_watt):
    P_watt = np.maximum(P_watt, 1e-25)
    return 10.0 * np.log10(P_watt) + 30.0

def compute_N2_from_photon_flux(phi_p, phi_s):
    """
    phi_p, phi_s can be scalars or numpy arrays (same shape).
    Returns N2 (same shape) clipped to [0, Nt].
    """
    W_p_abs = Gamma_p * sigma_ap * phi_p
    W_p_ems = Gamma_p * sigma_ep * phi_p
    W_s_abs = Gamma_s * sigma_as * phi_s
    W_s_ems = Gamma_s * sigma_es * phi_s

    A = W_p_abs + W_s_abs
    B = 1.0 / tau + W_p_ems + W_s_ems

    # avoid division by zero
    denom = (A + B)
    with np.errstate(divide='ignore', invalid='ignore'):
        N2 = Nt * (A / denom)
    N2 = np.nan_to_num(N2, nan=0.0, posinf=Nt, neginf=0.0)
    N2 = np.clip(N2, 0.0, Nt)
    return N2

# --------------------------
# ODE system for solve_bvp
# y[0] = P_p(z) (backward pump flowing toward decreasing z, injected at z=L)
# y[1] = P_s(z) (signal flowing toward increasing z, injected at z=0)
#
# Derivatives when using z increasing from 0->L:
#  - pump: dPp/dz = +Gamma_p*(sigma_ap*N1 - sigma_ep*N2)*Pp + alpha_p*Pp
#    (positive because pump power increases with z as z -> L where pump is injected)
#  - signal: dPs/dz = Gamma_s*(sigma_es*N2 - sigma_as*N1)*Ps - alpha_s*Ps
# --------------------------
def edfa_bvp_odes(z, y):
    # y shape: (2, m)
    Pp = y[0]
    Ps = y[1]

    # prevent negative or zero values inside photon flux calc
    Pp_safe = np.maximum(Pp, 0.0)
    Ps_safe = np.maximum(Ps, 0.0)

    # photon fluxes (photons / (m^2 * s))
    phi_p = Pp_safe / (h * nu_p * A_eff)
    phi_s = Ps_safe / (h * nu_s * A_eff)

    N2 = compute_N2_from_photon_flux(phi_p, phi_s)
    N1 = Nt - N2

    # pump derivative: positive sign (pump injected at z=L)
    # dPp/dz = + [Gamma_p*(sigma_ap*N1 - sigma_ep*N2) + alpha_p] * Pp
    dPp_dz = ( Gamma_p * (sigma_ap * N1 - sigma_ep * N2) + alpha_p ) * Pp

    # signal derivative: same as forward case
    dPs_dz = ( Gamma_s * (sigma_es * N2 - sigma_as * N1) - alpha_s ) * Ps

    return np.vstack((dPp_dz, dPs_dz))

# --------------------------
# Boundary conditions for solve_bvp
# For backward pump:
#  - Pp(L) = Pp_in   (pump injected at z=L)
#  - Ps(0) = P_sig_in (signal injected at z=0)
# bc returns residuals: [Pp(0) - ? , Ps(L) - ?]   (ordered as desired)
# --------------------------
def make_bc(pump_power_in_watt):
    def bc(ya, yb):
        # ya: y at z=0, yb: y at z=L
        Pp_at_0 = ya[0]
        Ps_at_L = yb[1]
        # residuals: Pp(L) should equal pump_power_in_watt -> yb[0] - pump
        #            Ps(0) should equal P_sig_in -> ya[1] - P_sig_in
        return np.array([ yb[0] - pump_power_in_watt,  # Pp(L) - Pp_in
                          ya[1] - P_sig_in ])         # Ps(0) - P_sig_in
    return bc

# --------------------------
# Sweep over pump powers and solve BVP
# --------------------------
pump_mw_array = np.linspace(1.0, 500.0, 60)   # pump powers in mW to sweep (1 mW .. 500 mW)
pump_w_array = pump_mw_array * 1e-3          # convert to Watts

Pout_signal_list = []

# initial guess arrays for solve_bvp (used as starting point for each pump)
# We'll reuse and update the guess to accelerate convergence
Pp_guess_global = None
Ps_guess_global = None

for idx, Pp_in in enumerate(pump_w_array):
    # initial guess for the solution y_guess on z_mesh
    if Pp_guess_global is None:
        # basic initial guess: pump small at z=0 -> linear rise to Pp_in at z=L
        Pp_guess = np.linspace(1e-9, Pp_in, z_mesh.size)
        # signal guess: start as constant equal to input signal
        Ps_guess = np.full_like(z_mesh, P_sig_in)
    else:
        # use previous converged solution as initial guess (helps convergence)
        Pp_guess = np.interp(z_mesh, prev_z, prev_sol[0])
        Ps_guess = np.interp(z_mesh, prev_z, prev_sol[1])

    y_guess = np.vstack((Pp_guess, Ps_guess))

    bc_fun = make_bc(Pp_in)

    try:
        sol = solve_bvp(edfa_bvp_odes, bc_fun, z_mesh, y_guess, tol=1e-4, max_nodes=5000)
    except Exception as e:
        print(f"[pump {Pp_in*1e3:.1f} mW] solve_bvp raised exception: {e}. Skipping this pump point.")
        Pout_signal_list.append(np.nan)
        continue

    if not sol.success:
        # Try to relax tolerances once if failed
        sol2 = solve_bvp(edfa_bvp_odes, bc_fun, z_mesh, y_guess, tol=1e-3, max_nodes=10000)
        if sol2.success:
            sol = sol2
        else:
            print(f"[pump {Pp_in*1e3:.1f} mW] solve_bvp failed to converge (message: {sol.message}).")
            Pout_signal_list.append(np.nan)
            # keep previous guess for next pump
            prev_z = z_mesh.copy()
            prev_sol = y_guess.copy()
            Pp_guess_global = prev_sol[0]
            Ps_guess_global = prev_sol[1]
            continue

    # store solution to speed up next initial guess
    prev_z = sol.x.copy()
    prev_sol = sol.y.copy()
    Pp_guess_global = prev_sol[0]
    Ps_guess_global = prev_sol[1]

    # Signal output at z = L (last point)
    Ps_out = float(sol.y[1, -1])
    Pout_signal_list.append(Ps_out)

# Convert to arrays for plotting
Pout_signal_array = np.array(Pout_signal_list)
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
plt.title(f"Backward-pumped EDFA: Output Signal power vs Pump power (L={L} m, Sig_in={P_sig_in} W)", fontsize=19)
plt.tight_layout()
plt.show()

# Print a compact table for a few pump points
for pmw, pout_w, pout_dbm in zip(pump_mw_array[::10], Pout_signal_array[::10], Pout_signal_dBm[::10]):
    if np.isnan(pout_w):
        print(f"Pump: {pmw:6.1f} mW -> Solver failed / no result.")
    else:
        print(f"Pump: {pmw:6.1f} mW -> Signal out: {pout_w*1e3:7.3f} mW   ({pout_dbm:6.2f} dBm)")
