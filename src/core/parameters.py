"""
Parameters based on final FNAL results
https://arxiv.org/abs/2506.03069 & https://arxiv.org/pdf/2402.15410
"""

import numpy as np
from dataclasses import dataclass

# from typing import Optional
# from enum import Enum
# 
# class ReferenceFrame(Enum):
#     """Reference frame options for calculations"""
#     REST = "rest"
#     LAB = "lab"

@dataclass
class PhysicsParameters:
    """
    Physics parameters 
    """
    # Fundamental constants
    c: float = 299792458  # Speed of light in a vacuum m/s
    e: float = 1.602176634e-19  # Elementary charge [C]
    q: float = +1.0  # In units of elementary charge
    h: float = 6.62607015e-34 # Planck constant [Js]
    hbar: float = h / (2 * np.pi)  # Reduced Planck constant [Js]
    m_mu: float = 105.6583745  # Muon mass [MeV/c^2] (PDG 2024)
    tau_mu: float = 2.197e-6  # Muon rest frame lifetime seconds (PDG 2024)

    @property
    def m_mu_kg(self) -> float:
        """Muon mass in kg"""
        return self.m_mu * 1e6 * self.e / (self.c**2)
    
@dataclass
class MuonParameters:
    """
    Muon parameters for g-2 experiment
    """
    physics: PhysicsParameters

    # g-2 parameters 
    a_mu: float = 1165920715e-12  # Anomalous magnetic moment
    a_mu_uncertainty: float = 145e-12  # a_mu uncertainty in a_mu
    a_mu_relative_uncertainty: float = 124e-9  # Relative uncertainty in ppb
    magic_momentum: float = 3.094  # GeV/c
    
    @property
    def magic_gamma(self) -> float:
        """Magic Lorentz factor calculated from a_mu"""
        return np.sqrt(1 + 1/self.a_mu)
    
    @property
    def magic_beta(self) -> float:
        """Magic beta (v/c) calculated from magic_gamma"""
        return np.sqrt(1 - 1/self.magic_gamma**2)
    
    @property
    def g_factor(self) -> float:
        """Calculate g-factor from anomalous magnetic moment"""
        return 2.0 * (1 + self.a_mu)
    
    @property
    def g_factor_uncertainty(self) -> float:
        """Calculate g-factor uncertainty from a_mu uncertainty"""
        return 2.0 * self.a_mu_uncertainty
    
    @property
    def muon_magnetic_moment(self) -> float:
        """Magnetic moment in J/T
        mu = g * (|q|e/2m) * (hbar/2) for spin-1/2
        """
        return (self.g_factor * abs(self.physics.q) * self.physics.e * self.physics.hbar) / (4 * self.physics.m_mu_kg)

@dataclass
class PrecessionParameters:
    """Spin precession parameters (muon rest frame)"""
    physics: PhysicsParameters
    muons: MuonParameters

    def cyclotron_frequency(self, B_field: float) -> float:
        """Cyclotron frequency in rest frame [rad/s]"""
        return self.physics.e / self.physics.m_mu_kg * B_field
        
    def anomalous_precession_frequency(self, B_field: float) -> float:
        """Anomalous precession frequency in rest frame [rad/s]"""
        return self.muons.a_mu * self.cyclotron_frequency(B_field)
    
    def edm_tilt_angle(self, edm_magnitude_ecm: float, B_field: float) -> float:
        """
        Calculate EDM-induced tilt angle in rest frame [radians]
        Uses formula: tan δ = (2m_μcβ d_μ)/(Qeℏa_μ)
        """
        # Convert EDM to SI units  
        d_mu_si = edm_magnitude_ecm * self.physics.e * 1e-2  # Cm
        
        # Apply rest frame formula (no gamma factors)
        numerator = (2 * self.physics.m_mu_kg * self.physics.c * 
                    self.muons.magic_beta * d_mu_si)
        denominator = (self.physics.q * self.physics.e * 
                      self.physics.hbar * self.muons.a_mu)
        
        return numerator / denominator  # Small angle: tan(delta) ~= delta
    
@dataclass
class RingGeometry:
    """Storage ring geometry parameters"""
    
    # Main ring dimensions
    radius: float = 7.112  # metres (major radius R0)
    
    # Beam aperture (defined by collimators)
    minor_radius: float = 0.045  # metres 
    
    # Storage volume
    @property
    def storage_cross_section(self) -> float:
        """Cross-sectional area of storage volume"""
        return np.pi * self.minor_radius**2
    
    @property
    def storage_volume(self) -> float:
        """Total storage volume (toroidal)"""
        return 2 * np.pi * self.radius * self.storage_cross_section

@dataclass
class MagneticField:
    """Magnetic field configuration"""
    ring: RingGeometry
    nominal_field: float = 1.45  # Tesla (confirmed in multiple sources)
    field_uncertainty: float = 0.000034  # Tesla (34 ppb from final paper)
    field_homogeneity: float = 70e-9  # 70 ppb target precision