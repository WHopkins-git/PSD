"""
Physical constants and conversion factors
"""

import numpy as np

# Fundamental constants
SPEED_OF_LIGHT = 2.998e8  # m/s
ELECTRON_MASS_KEV = 510.999  # keV/c²
AVOGADRO = 6.022e23  # mol^-1
ELEMENTARY_CHARGE = 1.602e-19  # C

# Unit conversions
KEV_TO_MEV = 1e-3
MEV_TO_KEV = 1e3
CM_TO_M = 1e-2
M_TO_CM = 1e2

# Compton scattering
def compton_edge_energy(photon_energy_keV):
    """
    Calculate Compton edge energy

    Parameters:
    -----------
    photon_energy_keV : float
        Incident photon energy

    Returns:
    --------
    edge_energy_keV : float
        Maximum electron energy (Compton edge)
    """
    alpha = photon_energy_keV / ELECTRON_MASS_KEV
    edge_energy = photon_energy_keV * (2 * alpha) / (1 + 2 * alpha)
    return edge_energy


def compton_scatter_angle(incident_keV, scattered_keV):
    """Calculate scatter angle from energies"""
    cos_theta = 1 - ELECTRON_MASS_KEV * (1/scattered_keV - 1/incident_keV)
    return np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi


# Scintillator physics
ANTHRACENE_LIGHT_YIELD = 17400  # photons/MeV

def relative_to_absolute_light_yield(relative_percent):
    """Convert relative light yield (% anthracene) to absolute (photons/MeV)"""
    return relative_percent / 100 * ANTHRACENE_LIGHT_YIELD
