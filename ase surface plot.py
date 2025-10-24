import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Physical Constants ---
h = 6.626e-34  # Planck's constant (J·s)
c = 3.0e8      # Speed of light (m/s)

# --- EDFA Parameters ---
L = 5.0          # Fiber length (m)
N_total = 1.0e25 # Total Er³⁺ ion concentration (ions/m³)

# --- Wavelength Configuration ---
wavelengths_nm = np.linspace(1520, 1575, 400)
wavelengths = wavelengths_nm * 1e-9
frequencies = c / wavelengths

# --- Cross-Section Models ---
def absorption_cross_section(wl):
    return 6.5e-25 * np.exp(-((wl - 1530e-9) / 8e-9)**2)

def emission_cross_section(wl):
    peak_1 = 6.0e-25 * np.exp(-((wl - 1532e-9) / 9e-9)**2)
    peak_2 = 3.5e-25 * np.exp(-((wl - 1555e-9) / 18e-9)**2)
    return peak_1 + peak_2

sigma_a = absorption_cross_section(wavelengths)
sigma_e = emission_cross_section(wavelengths)

# --- Range of Population Inversions (representing pump levels) ---
inversions = np.linspace(0.4, 0.9, 25)  # from low to high pump

# --- Prepare Storage for ASE Results ---
P_ase_dBm_surface = np.zeros((len(inversions), len(wavelengths)))

# --- Compute ASE Spectrum for Each Inversion ---
for i, n2 in enumerate(inversions):
    n1 = 1 - n2
    gamma = N_total * (sigma_e * n2 - sigma_a * n1)
    G = np.exp(np.clip(gamma * L, -35, 35))

    denom = (sigma_e * n2 - sigma_a * n1)
    n_sp = (sigma_e * n2) / (denom + 1e-30)
    n_sp[denom <= 0] = 1

    P_ase_psd = 2 * n_sp * h * frequencies * (G - 1)
    delta_lambda = wavelengths[1] - wavelengths[0]
    resolution_bandwidth_hz = (c / wavelengths**2) * delta_lambda
    P_ase_watts = P_ase_psd * resolution_bandwidth_hz

    P_ase_dBm_surface[i, :] = 10 * np.log10(P_ase_watts / 1e-3 + 1e-12)

# --- Create Meshgrid for Plotting ---
W, Inv = np.meshgrid(wavelengths_nm, inversions)

# --- Plot: 3D Surface ---
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(W, Inv, P_ase_dBm_surface, cmap='viridis', linewidth=0, antialiased=True)

ax.set_title('EDFA ASE Spectrum Surface', fontsize=20)
ax.set_xlabel('Wavelength (nm)', fontsize=20)
ax.set_ylabel('Population Inversion (n₂)', fontsize=20)
ax.set_zlabel('ASE Power (dBm)', fontsize=20)
ax.view_init(elev=30, azim=-130)  # Adjust angle for better view
cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.set_label('ASE Power (dBm)', fontsize=18)
plt.show()
