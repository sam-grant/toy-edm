import yaml
from pathlib import Path
from typing import Dict
import numpy as np

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

        # Key angular frequencies (rest frame)
        # Note that omega_a is frame-independent at magic momentum
        self.omega_c = self.precession.cyclotron_frequency(self.b_field.nominal_field)
        self.omega_a = self.precession.anomalous_precession_frequency(self.b_field.nominal_field)

        # Placeholder for results
        self._results = None

        # Print and dump configuration in one pass
        print("\n--- Configuration ---")
        param_dict = self._print_and_collect(self)

        # Build target path under root/config
        param_path = "config/simulator_config.yaml"
        root_dir = Path(__file__).resolve().parents[1]
        abs_path = root_dir / param_path

        # Ensure directory exists
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Dump to YAML
        self._dump_yaml(param_dict, abs_path)

    def _print_and_collect(self, obj, indent=0):
        """Recursively pretty-print dataclass fields and return a dict."""
        from dataclasses import is_dataclass

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
        
        # results dict
        results = { 't': t, 'phi_c': np.zeros((time_steps, n_muons)), 'phi_a': np.zeros((time_steps, n_muons)), 
                    'sx': np.zeros((time_steps, n_muons)), 'sy': np.zeros((time_steps, n_muons)), 'sz': np.zeros((time_steps, n_muons)),
                    'x': np.zeros((time_steps, n_muons)), 'y': np.zeros((time_steps, n_muons)),
                    'theta_y': np.zeros((time_steps, n_muons)),
                    't_mod': t_mod, 'omega_a': self.omega_a, 'n_muons': n_muons, 'edm_tilt': edm_tilt, 'edm_tilt_lab': edm_tilt_lab,
                    'backgrounds': backgrounds }
        
        def ppm_to_omega(ppm):
            return self.omega_a * (ppm * 1e-6)
        
        # Loop over muons
        for muon_idx in range(n_muons):
            # Initial spin: along momentum (z-direction/longitudinal)
            s = np.array([0.0, 0.0, 1.0])
            
            # Loop over time for this muon
            for i, time in enumerate(t):
                phi_c = self.omega_c * time #% 2*np.pi
                phi_a = self.omega_a * time # % 2*np.pi
                x = self.ring.radius * np.cos(phi_c)
                y = self.ring.radius * np.sin(phi_c)
                
                # Build Omega vector (rad/s)
                # This represents the effective magnetic field direction (times frequency)
                
                # Main g-2 precession: vertical field causes precession about y
                Omega = np.array([0.0, self.omega_a, 0.0])
                
                # The EDM contribution, similar to a radial field, tilts the precession
                # axis toward the radial (x) direction.
                if edm_mag != 0.0:
                    Omega[0] -= self.omega_a * edm_tilt_lab
                
                # Background fields modify the effective field direction:
                
                if "Br_n0" in backgrounds:
                    # Radial field component (Br) tilts the precession axis
                    # radially, so it adds a component to Omega[0].
                    Omega[0] -= ppm_to_omega(backgrounds["Br_n0"])
                
                if "Bz_n0" in backgrounds:
                    # Uniform longitudinal field (Bz) tilts the precession axis
                    # along the z-direction, so it adds a component to Omega[2].
                    Omega[2] += ppm_to_omega(backgrounds["Bz_n0"])
                
                if "Bz_n1" in backgrounds:
                    # Dipole longitudinal field (Bz) adds a position-dependent
                    # component along the z-axis.
                    Omega[2] += ppm_to_omega(backgrounds["Bz_n1"]) * np.cos(phi_c)
                
                # Rotate spin vector by angle theta = |Omega|*dt about axis n = Omega/|Omega|
                Omega_mag = np.linalg.norm(Omega)
                if Omega_mag > 0:
                    n = Omega / Omega_mag
                    theta = Omega_mag * dt
                    # Rodrigues' rotation formula:
                    s = (s * np.cos(theta)
                        + np.cross(n, s) * np.sin(theta)
                        + n * (np.dot(n, s)) * (1 - np.cos(theta)))
                
                # Renormalize (for numerical stability)
                s = s / np.linalg.norm(s)
                
                # Store results
                results['phi_a'][i, muon_idx] = phi_a
                results['phi_c'][i, muon_idx] = phi_c
                results['sx'][i, muon_idx] = s[0]
                results['sy'][i, muon_idx] = s[1]
                results['sz'][i, muon_idx] = s[2]
                results['x'][i, muon_idx] = x
                results['y'][i, muon_idx] = y
                s_mag = np.linalg.norm(s)
                results['theta_y'][i, muon_idx] = np.arcsin(s[1] / s_mag)
        
        return results
    # def run(self,
    #         n_muons: int = 1000,
    #         t_max: float = 50e-6,
    #         time_steps: int = 100,
    #         edm_mag: float = 5.4e-18,
    #         backgrounds: Dict[str, float] = {"Bz_n0": 1.0, "Bz_n1": 1.0, "Br_n0": 1.0}
    #         ) -> Dict:

    #     t = np.linspace(0, t_max, time_steps)
    #     dt = t[1] - t[0]
    #     T_g2 = 2 * np.pi / self.omega_a
    #     t_mod = np.mod(t, T_g2)

    #     # EDM tilt
    #     edm_tilt = self.precession.edm_tilt_angle(edm_mag, self.b_field.nominal_field)
    #     edm_tilt_lab = edm_tilt / self.muon.magic_gamma

    #     # results dict 
    #     results = { 't': t, 'phi_c': np.zeros((time_steps, n_muons)), 'phi_a': np.zeros((time_steps, n_muons)), 
    #                 'sx': np.zeros((time_steps, n_muons)), 'sy': np.zeros((time_steps, n_muons)), 'sz': np.zeros((time_steps, n_muons)),
    #                 'x': np.zeros((time_steps, n_muons)), 'y': np.zeros((time_steps, n_muons)),
    #                 'theta_y': np.zeros((time_steps, n_muons)),
    #                 't_mod': t_mod, 'omega_a': self.omega_a, 'n_muons': n_muons, 'edm_tilt': edm_tilt, 'edm_tilt_lab': edm_tilt_lab,
    #                 'backgrounds': backgrounds }

    #     def ppm_to_omega(ppm):
    #         return self.omega_a * (ppm * 1e-6)

    #     # Loop over muons
    #     for muon_idx in range(n_muons):
    #         # Initial spin: along momentum (z-direction)
    #         s = np.array([0.0, 0.0, 1.0])
            
    #         # For radial field: it causes a static tilt of the spin away from z
    #         # The spin precesses around the tilted axis
    #         if "Br_n0" in backgrounds and backgrounds["Br_n0"] != 0:
    #             # Radial field tilts the effective B field
    #             # The tilt angle is proportional to Br/By
    #             tilt_angle = backgrounds["Br_n0"] * 1e-6  # ppm to radians (small angle)
    #             # Tilt the initial spin in the x-z plane
    #             s = np.array([np.sin(tilt_angle), 0.0, np.cos(tilt_angle)])
            
    #         # Loop over time for this muon
    #         for i, time in enumerate(t):
    #             phi_c = self.omega_c * time
    #             phi_a = self.omega_a * time
    #             x = self.ring.radius * np.cos(phi_c)
    #             y = self.ring.radius * np.sin(phi_c)

    #             # Build Omega vector (rad/s)
    #             Omega = np.array([0.0, 0.0, 0.0])
                
    #             # Main g-2 precession about vertical (y) axis
    #             Omega[1] = self.omega_a
                
    #             # EDM contribution -> precession about radial (x) axis
    #             if edm_mag != 0.0:
    #                 Omega[0] = self.omega_a * edm_tilt_lab
                
    #             # Background field effects:
    #             if "Bz_n0" in backgrounds:
    #                 # Uniform longitudinal field -> changes precession frequency
    #                 # This modifies the y-component of Omega
    #                 Omega[1] *= (1 + backgrounds["Bz_n0"] * 1e-6)
                
    #             if "Bz_n1" in backgrounds:
    #                 # Longitudinal field with cos(phi) modulation
    #                 Omega[1] *= (1 + backgrounds["Bz_n1"] * 1e-6 * np.cos(phi_c))
                
    #             # Note: Br effect was handled by tilting the initial spin
    #             # It doesn't add to Omega directly
                
    #             # Rotate spin vector
    #             Omega_mag = np.linalg.norm(Omega)
    #             if Omega_mag > 0:
    #                 n = Omega / Omega_mag
    #                 theta = Omega_mag * dt
    #                 # Rodrigues' rotation formula:
    #                 s = (s * np.cos(theta)
    #                     + np.cross(n, s) * np.sin(theta)
    #                     + n * (np.dot(n, s)) * (1 - np.cos(theta)))
                
    #             # Renormalize to ensure unit length
    #             s = s / np.linalg.norm(s)
                
    #             # Store results
    #             results['phi_a'][i, muon_idx] = phi_a
    #             results['phi_c'][i, muon_idx] = phi_c
    #             results['sx'][i, muon_idx] = s[0]
    #             results['sy'][i, muon_idx] = s[1]
    #             results['sz'][i, muon_idx] = s[2]
    #             results['x'][i, muon_idx] = x
    #             results['y'][i, muon_idx] = y
    #             results['theta_y'][i, muon_idx] = np.arcsin(s[1])

    #     return results

    # def run(self,
    #         n_muons: int = 1000,
    #         t_max: float = 50e-6,
    #         time_steps: int = 100,
    #         edm_mag: float = 5.4e-18,
    #         backgrounds: Dict[str, float] = {"Bz_n0": 1.0, "Bz_n1": 1.0, "Br_n0": 1.0}
    #         ) -> Dict:

    #     t = np.linspace(0, t_max, time_steps)
    #     dt = t[1] - t[0]
    #     T_g2 = 2 * np.pi / self.omega_a
    #     t_mod = np.mod(t, T_g2)

    #     # EDM tilt
    #     edm_tilt = self.precession.edm_tilt_angle(edm_mag, self.b_field.nominal_field)
    #     edm_tilt_lab = edm_tilt / self.muon.magic_gamma

    #     # results dict 
    #     results = { 't': t, 'phi_c': np.zeros((time_steps, n_muons)), 'phi_a': np.zeros((time_steps, n_muons)), 
    #                 'sx': np.zeros((time_steps, n_muons)), 'sy': np.zeros((time_steps, n_muons)), 'sz': np.zeros((time_steps, n_muons)),
    #                 'x': np.zeros((time_steps, n_muons)), 'y': np.zeros((time_steps, n_muons)),
    #                 'theta_y': np.zeros((time_steps, n_muons)),
    #                 't_mod': t_mod, 'omega_a': self.omega_a, 'n_muons': n_muons, 'edm_tilt': edm_tilt, 'edm_tilt_lab': edm_tilt_lab,
    #                 'backgrounds': backgrounds }

    #     def ppm_to_omega(ppm):
    #         return self.omega_a * (ppm * 1e-6)

    #     # Loop over muons
    #     for muon_idx in range(n_muons):
    #         # Initial spin: along momentum direction (z-direction/tangential)
    #         s = np.array([0.0, 0.0, 1.0])  # pointing along beam/tangential
            
    #         # Loop over time for this muon
    #         for i, time in enumerate(t):
    #             phi_c = self.omega_c * time
    #             phi_a = self.omega_a * time
    #             x = self.ring.radius * np.cos(phi_c)
    #             y = self.ring.radius * np.sin(phi_c)

    #             # Build Omega vector (rad/s)
    #             # Main g-2 precession about vertical (y) axis - CORRECT!
    #             Omega = np.array([0.0, self.omega_a, 0.0])

    #             # EDM contribution -> precession about radial (x) axis - CORRECT!
    #             if edm_mag != 0.0:
    #                 Omega[0] += self.omega_a * edm_tilt_lab

    #             # Backgrounds:
    #             if "Br_n0" in backgrounds:
    #                 # Radial field (Bx) -> cross product gives rotation about z axis
    #                 # τ = μ × B, so μ(along z initially) × Bx gives rotation about y? 
    #                 # Actually need to think about what Br means...
    #                 # If Br is radial field, it causes vertical tilt
    #                 Omega[2] += ppm_to_omega(backgrounds["Br_n0"])  # rotation about z

    #             if "Bz_n0" in backgrounds:
    #                 # Longitudinal field (Bz) -> rotation about x (radial)
    #                 Omega[0] += ppm_to_omega(backgrounds["Bz_n0"])

    #             if "Bz_n1" in backgrounds:
    #                 # Longitudinal field varying with position
    #                 Omega[0] += ppm_to_omega(backgrounds["Bz_n1"]) * np.cos(phi_c)

    #             # Rotate spin vector
    #             Omega_mag = np.linalg.norm(Omega)
    #             if Omega_mag > 0:
    #                 n = Omega / Omega_mag
    #                 theta = Omega_mag * dt
    #                 # Rodrigues' rotation formula:
    #                 s = (s * np.cos(theta)
    #                     + np.cross(n, s) * np.sin(theta)
    #                     + n * (np.dot(n, s)) * (1 - np.cos(theta)))

    #             # Renormalize
    #             s = s / np.linalg.norm(s)

    #             # Store results
    #             results['phi_a'][i, muon_idx] = phi_a
    #             results['phi_c'][i, muon_idx] = phi_c
    #             results['sx'][i, muon_idx] = s[0]
    #             results['sy'][i, muon_idx] = s[1]
    #             results['sz'][i, muon_idx] = s[2]
    #             results['x'][i, muon_idx] = x
    #             results['y'][i, muon_idx] = y
    #             results['theta_y'][i, muon_idx] = np.arcsin(s[1])  # vertical component

    #     return results
    # def run(self,
    #         n_muons: int = 1000,
    #         t_max: float = 50e-6,
    #         time_steps: int = 100,
    #         edm_mag: float = 5.4e-18,
    #         backgrounds: Dict[str, float] = {"Bz_n0": 1.0, "Bz_n1": 1.0, "Br_n0": 1.0}
    #         ) -> Dict:

    #     t = np.linspace(0, t_max, time_steps)
    #     dt = t[1] - t[0]
    #     T_g2 = 2 * np.pi / self.omega_a
    #     t_mod = np.mod(t, T_g2)

    #     # EDM tilt
    #     edm_tilt = self.precession.edm_tilt_angle(edm_mag, self.b_field.nominal_field)
    #     edm_tilt_lab = edm_tilt / self.muon.magic_gamma

    #     # results dict 
    #     results = { 't': t, 'phi_c': np.zeros((time_steps, n_muons)), 'phi_a': np.zeros((time_steps, n_muons)), 
    #                 'sx': np.zeros((time_steps, n_muons)), 'sy': np.zeros((time_steps, n_muons)), 'sz': np.zeros((time_steps, n_muons)),
    #                 'x': np.zeros((time_steps, n_muons)), 'y': np.zeros((time_steps, n_muons)),
    #                 'theta_y': np.zeros((time_steps, n_muons)),
    #                 't_mod': t_mod, 'omega_a': self.omega_a, 'n_muons': n_muons, 'edm_tilt': edm_tilt, 'edm_tilt_lab': edm_tilt_lab,
    #                 'backgrounds': backgrounds }

    #     # Interpret backgrounds: precession frequencies scale with B, so
    #     # delta_omega = omega_a * (dB/B) = omega_a * ppm*1e-6
    #     def ppm_to_omega(ppm):
    #         return self.omega_a * (ppm * 1e-6)

    #     # Loop over muons
    #     for muon_idx in range(n_muons):
    #         # Initial spin for this muon: lying in x-z plane, along x
    #         # You could randomize this later if desired
    #         s = np.array([0.0, 0.0, 1.0])  # shape (3,)

    #         # Loop over time for this muon
    #         for i, time in enumerate(t):
    #             phi_c = self.omega_c * time
    #             phi_a = self.omega_a * time
    #             x = self.ring.radius * np.cos(phi_c)
    #             y = self.ring.radius * np.sin(phi_c)

    #             # Build Omega vector (rad/s)
    #             # main g-2 precession about y:
    #             # dS/dt = Omega x S
    #             Omega = np.array([0.0, self.omega_a, 0.0])

    #             # EDM contribution -> effective precession about x.
    #             if edm_mag != 0.0:
    #                 Omega[0] += self.omega_a * edm_tilt_lab   # adds rotation about x

    #             # Backgrounds:
    #             if "Br_n0" in backgrounds:
    #                 # radial field -> rotation about x
    #                 Omega[0] += ppm_to_omega(backgrounds["Br_n0"])

    #             if "Bz_n0" in backgrounds:
    #                 # longitudinal field -> rotation about z (uniform)
    #                 Omega[2] += ppm_to_omega(backgrounds["Bz_n0"])

    #             if "Bz_n1" in backgrounds:
    #                 # dipole longitudinal field varying like cos(phi_c) -> contributes to Omega_z
    #                 Omega[2] += ppm_to_omega(backgrounds["Bz_n1"]) * np.cos(phi_c)

    #             # rotate s by angle theta = |Omega|*dt about axis n = Omega/|Omega|
    #             Omega_mag = np.linalg.norm(Omega)
    #             if Omega_mag > 0:
    #                 n = Omega / Omega_mag
    #                 theta = Omega_mag * dt
    #                 # Rodrigues' rotation formula:
    #                 s = (s * np.cos(theta)
    #                     + np.cross(n, s) * np.sin(theta)
    #                     + n * (np.dot(n, s)) * (1 - np.cos(theta)))

    #             # Renormalize to maintain unit length (optional, for numerical stability)
    #             s = s / np.linalg.norm(s)

    #             # Store results for this specific muon
    #             results['phi_a'][i, muon_idx] = phi_a
    #             results['phi_c'][i, muon_idx] = phi_c
    #             results['sx'][i, muon_idx] = s[0]
    #             results['sy'][i, muon_idx] = s[1]
    #             results['sz'][i, muon_idx] = s[2]
    #             results['x'][i, muon_idx] = x
    #             results['y'][i, muon_idx] = y
    #             s_mag = np.linalg.norm(s)
    #             results['theta_y'][i, muon_idx] = np.arcsin(s[1] / s_mag)

    #     return results

    # def run(self,
    #         n_muons: int = 1000,
    #         t_max: float = 50e-6,
    #         time_steps: int = 100,
    #         edm_mag: float = 5.4e-18,
    #         backgrounds: Dict[str, float] = {"Bz_n0": 1.0, "Bz_n1": 1.0, "Br_n0": 1.0}
    #         ) -> Dict:

    #     t = np.linspace(0, t_max, time_steps)
    #     dt = t[1] - t[0]
    #     T_g2 = 2 * np.pi / self.omega_a
    #     t_mod = np.mod(t, T_g2)

    #     # EDM tilt
    #     edm_tilt = self.precession.edm_tilt_angle(edm_mag, self.b_field.nominal_field)
    #     edm_tilt_lab = edm_tilt / self.muon.magic_gamma

    #     # results dict 
    #     results = { 't': t, 'phi_c': np.zeros((time_steps, n_muons)), 'phi_a': np.zeros((time_steps, n_muons)), 
    #                 'sx': np.zeros((time_steps, n_muons)), 'sy': np.zeros((time_steps, n_muons)), 'sz': np.zeros((time_steps, n_muons)),
    #                 'x': np.zeros((time_steps, n_muons)), 'y': np.zeros((time_steps, n_muons)),
    #                 'theta_y': np.zeros((time_steps, n_muons)),
    #                 't_mod': t_mod, 'omega_a': self.omega_a, 'n_muons': n_muons, 'edm_tilt': edm_tilt, 'edm_tilt_lab': edm_tilt_lab,
    #                 'backgrounds': backgrounds }

    #     # Interpret backgrounds: user gave ppm (dB/B). Precession frequencies scale with B, so
    #     # delta_omega = omega_a * (dB/B) = omega_a * ppm*1e-6
    #     def ppm_to_omega(ppm):
    #         return self.omega_a * (ppm * 1e-6)

    #     # # Initial phases
    #     # # Completely deterministic 
    #     # phi_a = 0 
    #     # phi_c = 0

    #     # initial spin: lying in x-z plane, along z
    #     # start spin in rest frame 
    #     s = np.array([1, 0, 0]) # shape (3,)

    #     for i, time in enumerate(t):
    #         phi_c = self.omega_c * time
    #         phi_a = self.omega_a * time
    #         x = self.ring.radius * np.cos(phi_c)
    #         y = self.ring.radius * np.sin(phi_c)

    #         # Build Omega vector (rad/s)
    #         # main g-2 precession about y:
    #         # dS/dt = Omega x S
    #         Omega = np.array([0.0, self.omega_a, 0.0])

    #         # EDM contribution -> effective precession about x.
    #         # For a small tilt angle of the precession plane, a reasonable small-frequency
    #         # approx is omega_edm ~ omega_a * edm_tilt_lab
    #         if edm_mag != 0.0:
    #             Omega[0] += self.omega_a * edm_tilt_lab   # adds rotation about x

    #         # Backgrounds:
    #         # The fields do not really modify the precession vector though?
    #         if "Br_n0" in backgrounds:
    #             # radial field -> rotation about x
    #             Omega[0] += ppm_to_omega(backgrounds["Br_n0"])

    #         if "Bz_n0" in backgrounds:
    #             # longitudinal field -> rotation about z (uniform)
    #             Omega[2] += ppm_to_omega(backgrounds["Bz_n0"])

    #         if "Bz_n1" in backgrounds:
    #             # dipole longitudinal field varying like cos(phi_c) -> contributes to Omega_z
    #             Omega[2] += ppm_to_omega(backgrounds["Bz_n1"]) * np.cos(phi_c)

    #         # rotate s by angle theta = |Omega|*dt about axis n = Omega/|Omega|
    #         Omega_mag = np.linalg.norm(Omega)
    #         if Omega_mag > 0:
    #             n = Omega / Omega_mag
    #             theta = Omega_mag * dt
    #             # Rodrigues' rotation formula:
    #             s = (s * np.cos(theta)
    #                 + np.cross(n, s) * np.sin(theta)
    #                 + n * (np.dot(n, s)) * (1 - np.cos(theta)))
                
    #         # if Omega_mag > 0:
    #         #     n = Omega / Omega_mag
    #         #     theta = Omega_mag * dt
                
    #         #     # Build rotation matrix using axis-angle formula
    #         #     K = np.array([[0, -n[2], n[1]],
    #         #                 [n[2], 0, -n[0]],
    #         #                 [-n[1], n[0], 0]])  # skew-symmetric matrix
                
    #         #     R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    #         #     s = np.dot(R, s)
    #         # else Omega_mag == 0 => s unchanged

    #         # Renormalize to maintain unit length
    #         # s = s / np.linalg.norm(s)

    #         # store
    #         results['phi_a'][i] = phi_a
    #         results['phi_c'][i] = phi_c
    #         results['sx'][i] = s[0]
    #         results['sy'][i] = s[1]
    #         results['sz'][i] = s[2]
    #         results['x'][i] = x
    #         results['y'][i] = y
    #         s_mag = np.linalg.norm(s)
    #         results['theta_y'][i] = np.arcsin(s[1] / s_mag)

    #     return results

    ################################################################################

        # Alternatively, store results in the simulator object
        # This feels correct, although in practice it makes life harder
        # self._results = results

    # @property 
    # def results(self):
    #     """Access the most recent simulation results"""
    #     if self._results is None:
    #         print("results is None. Call Simulation().run() first.")
    #         return None
    #     return self._results
    
    ################################################################################


    # def run_batch(self, configs: Dict[str, Dict]) -> Dict[str, Dict]:
    #     """Simple batch runner - just the configs dict"""
    #     results = {}
    #     for name, config in configs.items():
    #         results[name] = self.run(**config)
    #     return results
    
    # def compare_backgrounds(self, strength: float = 1.0) -> Dict[str, Dict]:
    #     """Most common use case"""
    #     configs = {
    #         'no_bkg': {'backgrounds': {"Br_n0": 0.0, "Bz_n0": 0.0, "Bz_n1": 0.0}},
    #         'br_n0': {'backgrounds': {"Br_n0": strength, "Bz_n0": 0.0, "Bz_n1": 0.0}},
    #         #  ... etc
    #     }
    #     return self.run_batch(configs)
    


    # Frame transformations (not currently used)
    # def _boost_to_lab_frame(self, rest_results: Dict) -> Dict:
    #     """Boost rest frame results to lab frame"""
        
    #     lab_results = rest_results.copy()
        
    #     # Time dilation: lab time = gamma times rest time
    #     lab_results['times'] = rest_results['times'] * self.muon.magic_gamma
        
    #     # Recalculate phases for lab frame frequencies
    #     # Note: anomalous precession frequency stays the same at magic momentum
    #     lab_results['phi_cyclotron'] = (rest_results['phi_cyclotron'] / 
    #                                   self.muon.magic_gamma)
    #     # phi_apin phases do not change because omega_a is frame-independent
        
    #     # Spatial coordinates and spin components do not change
    #     # (Lorentz transformation in transverse directions is identity)
        
    #     # Mark as lab frame
    #     lab_results['frame'] = ReferenceFrame.LAB
        
        # return lab_results
    
    # def get_lab_frequencies(self) -> Dict[str, float]:
    #     """Get lab frame frequencies for reference"""
    #     return {
    #         'omega_c_lab': self.omega_c_rest / self.muon.magic_gamma,
    #         'omega_a_lab': self.omega_a_rest,  # Same as rest frame at magic momentum
    #         'omega_c_rest': self.omega_c_rest,
    #         'omega_a_rest': self.omega_a_rest
    #     }