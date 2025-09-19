import yaml
from pathlib import Path
from typing import Dict
import numpy as np
from dataclasses import is_dataclass
from scipy.spatial.transform import Rotation as R

# Import dataclasses
from parameters import PhysicsParameters, MuonParameters, PrecessionParameters, RingGeometry, MagneticField

class Simulation:
    def __init__(self, param_path: str = "config/sim_params.yaml"):
        """
        Muon EDM simulation framework
        """ 
        self.physics = PhysicsParameters()
        self.muon = MuonParameters(self.physics)
        self.precession = PrecessionParameters(self.physics, self.muon)
        self.ring = RingGeometry()
        self.b_field = MagneticField(self.ring)

        # Key angular frequencies
        # Note that omega_a is frame-independent at magic momentum
        self.omega_c = self.precession.cyclotron_frequency(self.b_field.nominal_field)
        self.omega_a = self.precession.anomalous_precession_frequency(self.b_field.nominal_field)

        # Placeholder for results
        # self._results = None

        # Print and dump configuration in one pass
        print("\n--- Configuration ---")
        param_dict = self._print_and_collect(self)

        # Build target path under root/config
        param_path = "config/simulator_config.yaml"
        root_dir = Path(__file__).resolve().parents[1]
        abs_path = root_dir / param_path

        # Ensure directory exists
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Dump config to YAML
        self._dump_yaml(param_dict, abs_path)

    def _print_and_collect(self, obj, indent=0):
        """Print dataclass fields and return a dict"""
        pad = "  " * indent
        cls_name = obj.__class__.__name__
        print(f"{pad}{cls_name}:")

        data = {}
        for key, value in vars(obj).items():
            if is_dataclass(value):
                print(f"{pad}  {key}:")
                data[key] = self._print_and_collect(value, indent + 2)
            else:
                print(f"{pad}  {key}: {value}")
                data[key] = value
        return data

    def _dump_yaml(self, config_dict: dict, path: Path):
        """Write the collected dict to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(config_dict, f, sort_keys=False)
        print(f"\nConfig dumped to {path}")

    ################################################################################
    # Run simulation
    ################################################################################
    def run(self,
            n_muons: int = 1000,
            t_max: float = 50e-6,
            time_steps: int = 100,
            edm_mag: float = 5.4e-18,
            backgrounds: Dict[str, float] = {"Bz_n0": 1.0, "Bz_n1": 1.0, "Br_n0": 1.0}
            ) -> Dict:
        
        t = np.linspace(0, t_max, time_steps)
        dt = t[1] - t[0]
        T_g2 = 2 * np.pi / self.omega_a
        t_mod = np.mod(t, T_g2)
        
        # EDM tilt
        edm_tilt = self.precession.edm_tilt_angle(edm_mag, self.b_field.nominal_field)
        edm_tilt_lab = edm_tilt / self.muon.magic_gamma
        
        # Results 
        results = { 
            # Arrays, initialised with zeros 
            't': t, 'phi_c': np.zeros((time_steps, n_muons)), 'phi_a': np.zeros((time_steps, n_muons)), 
            'sx': np.zeros((time_steps, n_muons)), 'sy': np.zeros((time_steps, n_muons)), 'sz': np.zeros((time_steps, n_muons)),
            'x': np.zeros((time_steps, n_muons)), 'y': np.zeros((time_steps, n_muons)),
            # Constants
            'dt': dt, 'T_g2': T_g2, 't_mod': t_mod, 'omega_a': self.omega_a, 'n_muons': n_muons, 'edm_tilt': edm_tilt, 'edm_tilt_lab': edm_tilt_lab
        }
        
        def ppm_to_omega(ppm):
            """Helper to scale precession according to ppm tilt"""
            return self.omega_a * (ppm * 1e-6)
        
        # Loop over muons
        for muon_idx in range(n_muons):
            # Initial spin: along momentum (z-direction/longitudinal)
            s = np.array([0.0, 0.0, 1.0])
            
            # Loop over time for this muon
            for i, time in enumerate(t):
                phi_c = self.omega_c * time
                phi_a = self.omega_a * time
                x = self.ring.radius * np.cos(phi_c)
                y = self.ring.radius * np.sin(phi_c)
                
                # Build Omega vector (rad/s)
                
                # Main g-2 precession: precession about vertical dipole field
                Omega = np.array([0.0, self.omega_a, 0.0])
                
                # EDM contribution tilts the precession axis away from the radial (x) direction.
                if edm_mag != 0.0:
                    Omega[0] -= self.omega_a * edm_tilt
                
                # Background fields modify the effective field direction:
                
                if "Br_n0" in backgrounds:
                    # Radial field component tilts the precession axis in radially, 
                    # so it adds a component to Omega[0].
                    # FIXME: is it positive or negative? 
                    Omega[0] -= ppm_to_omega(backgrounds["Br_n0"])
                
                if "Bz_n0" in backgrounds:
                    # Uniform longitudinal field N=0 tilts the precession axis
                    # along the z-direction, so it adds a component to Omega[2].
                    Omega[2] += ppm_to_omega(backgrounds["Bz_n0"])
                
                if "Bz_n1" in backgrounds:
                    # Longitudinal field N=0 adds an azimuthally-dependent
                    # component along the z-axis.
                    Omega[2] += ppm_to_omega(backgrounds["Bz_n1"]) * np.cos(phi_c)
                
                # Rotate spin vector by angle: theta = |Omega|*dt about axis n = Omega/|Omega|
                Omega_mag = np.linalg.norm(Omega)

                if Omega_mag > 0:
                    n = Omega / Omega_mag
                    theta = Omega_mag * dt
                    # Create a rotation object using the axis-angle representation
                    rot = R.from_rotvec(theta * n)
                    # Apply the rotation to the spin vector 
                    s = rot.apply(s)
                    # 
                    # Rodrigues' rotation formula (same thing):
                    # s = (s * np.cos(theta)
                    #     + np.cross(n, s) * np.sin(theta)
                    #     + n * (np.dot(n, s)) * (1 - np.cos(theta)))

                # Renormalise 
                s = s / np.linalg.norm(s)

                # Store results (rest frame)
                results['phi_a'][i, muon_idx] = phi_a
                results['phi_c'][i, muon_idx] = phi_c
                results['sx'][i, muon_idx] = s[0]
                results['sy'][i, muon_idx] = s[1]
                results['sz'][i, muon_idx] = s[2]
                results['x'][i, muon_idx] = x
                results['y'][i, muon_idx] = y
                
        return results