import numpy as np
import matplotlib.pyplot as plt

h = 6.62607015e-34      # Planck constant (J*s)
c = 299792458.0         # speed of light (m/s)

L = 5.0                 # fiber length (m)
Nz = 500                # spatial steps
dz = L / Nz
z = np.linspace(0, L, Nz + 1)

Nt = 1.0e25             # Er3+ concentration (m^-3)
tau = 10e-3             # upper-state lifetime (s)
Gamma = 0.9             # overlap factor
A_eff = 12.56e-12         # effective area (m^2)

# Wavelengths and frequencies
lambda_s = 1530e-9
lambda_p = 980e-9
nu_s = c / lambda_s
nu_p = c / lambda_p

# Cross sections (typical order-of-magnitude values, in m^2)
sigma_ap = 1.8e-25          # pump absorption cross-section
sigma_ep = 3.15e-25         # pump emission cross-section (often small for 980)
sigma_as = 2.14e-25         # signal absorption cross-section
sigma_es = 3.80e-25         # signal emission cross-section

# Losses (m^-1)
alpha_p = 0.4 / 4.343e3   # 0.4 dB/km
alpha_s = 0.2 / 4.343e3   # 0.2 dB/km
# -------------------------------
# 3. Helper functions
# -------------------------------
def photon_flux(P, nu, Aeff):
    return P / (h * nu * Aeff)

def solve_N2(Pp_f, Pp_b, Ps_f, Ps_b):
    """Compute steady-state excited-state population fraction."""
    Phi_p = (Pp_f + Pp_b) / (h * nu_p * A_eff)
    Phi_s = (Ps_f + Ps_b) / (h * nu_s * A_eff)
    
    # Handle potential division by zero or invalid values if powers are zero
    if (Phi_p + Phi_s) == 0:
        return 0.0 # No pumping, no excited state
        
    R_abs = sigma_ap * Phi_p + sigma_as * Phi_s
    R_em = sigma_ep * Phi_p + sigma_es * Phi_s
    
    denominator = (R_abs + R_em + 1 / tau)
    if denominator == 0:
        return 0.0 # Avoid division by zero
        
    return Nt * R_abs / denominator

# -------------------------------
# 4. Simulation setup
# -------------------------------
P_s_in = 10e-3      # Signal input (10 mW)
pump_powers = np.linspace(10e-3, 500e-3, 70)  # 10 mW to 500 mW (in Watts)
output_signal = [] # Will store output power in Watts

# -------------------------------
# 5. Simulation loop
# -------------------------------
print("Running EDFA simulation...")
for P_p_in in pump_powers:
    # Initialize fields
    Pp_f = np.zeros(Nz + 1)
    Pp_b = np.zeros(Nz + 1)
    Ps_f = np.zeros(Nz + 1)
    Ps_b = np.zeros(Nz + 1)

    # Boundary conditions
    Pp_f[0] = P_p_in      # forward pump at z=0
    Pp_b[-1] = P_p_in     # backward pump at z=L
    Ps_f[0] = P_s_in
    Ps_b[-1] = 0.0

    # Iterative bidirectional solver
    for _ in range(50):  # 50 iterations typically enough
        # Forward sweep (i from 0 to Nz-1)
        for i in range(Nz):
            N2 = solve_N2(Pp_f[i], Pp_b[i], Ps_f[i], Ps_b[i])
            N1 = Nt - N2
            g_s = Gamma * (sigma_es * N2 - sigma_as * N1)
            g_p = Gamma * (sigma_ep * N2 - sigma_ap * N1)
            Ps_f[i + 1] = Ps_f[i] + (g_s - alpha_s) * Ps_f[i] * dz
            Pp_f[i + 1] = Pp_f[i] + (g_p - alpha_p) * Pp_f[i] * dz
        
        # Backward sweep (i from Nz to 1)
        for i in range(Nz, 0, -1):
            N2 = solve_N2(Pp_f[i], Pp_b[i], Ps_f[i], Ps_b[i])
            N1 = Nt - N2
            g_s = Gamma * (sigma_es * N2 - sigma_as * N1)
            g_p = Gamma * (sigma_ep * N2 - sigma_ap * N1)
            # Note: The sign is tricky. A positive (g_s - alpha_s)
            # should *increase* the power as we go from z=L to z=0.
            # So dP/dz = -(g-a)P for backward.
            # P(i-1) = P(i) - [-(g-a)P(i)]*dz = P(i) + (g-a)P(i)*dz
            Ps_b[i - 1] = Ps_b[i] + (g_s - alpha_s) * Ps_b[i] * dz
            Pp_b[i - 1] = Pp_b[i] + (g_p - alpha_p) * Pp_b[i] * dz

        # Fix boundary pumps again
        Pp_f[0] = P_p_in
        Pp_b[-1] = P_p_in
        Ps_f[0] = P_s_in
        Ps_b[-1] = 0.0


    # Record output signal (power in Watts at z=L)
    output_signal.append(Ps_f[-1])
print("Simulation complete.")

# -------------------------------
# 6. Plot results (in dBm)
# -------------------------------

# Convert output power from Watts to dBm
# P(dBm) = 10 * log10( P(W) / 1e-3 W )
output_signal_W = np.array(output_signal)
output_signal_dBm = 10 * np.log10(output_signal_W / 1e-3)

plt.figure(figsize=(8,5))
# Plot Pump Power (mW) vs Output Signal Power (dBm)
plt.plot(pump_powers * 1e3, output_signal_dBm, 'o-', lw=2, color='blue')
plt.xlabel('Input Pump Power per End (mW)',fontsize=22)
plt.ylabel('Output Signal Power (dBm)',fontsize=22) # <-- Changed label
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('Bidirectional Pumped EDFA\nOutput Signal Power (dBm) vs Pump Power',fontsize=22) # <-- Changed title
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# 7. Print summary (in dBm)
# -------------------------------

# Calculate input signal in dBm for gain calculation
P_s_in_dBm = 10 * np.log10(P_s_in / 1e-3)
print("\n" + "="*60)
print(f"Input Signal: {P_s_in * 1e3:.1f} mW ({P_s_in_dBm:.2f} dBm)")
print("="*60)

# Use the numpy array 'output_signal_W' created for plotting
for Pin, Pout_W in zip(pump_powers, output_signal_W):
    # Convert output power (Watts) to dBm
    Pout_dBm = 10 * np.log10(Pout_W / 1e-3)
    
    # Calculate gain: Gain(dB) = P_out(dBm) - P_in(dBm)
    Gain_dB = Pout_dBm - P_s_in_dBm
    
    print(f"Pump: {Pin*1e3:6.1f} mW -> Signal out: {Pout_dBm:8.3f} dBm "
          f"(Gain = {Gain_dB:6.2f} dB)")
print("="*60)
