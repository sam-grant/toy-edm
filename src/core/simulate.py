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
    # Simulation methods
    ################################################################################

    def run(self, 
            n_muons: int = 1000, 
            t_max: float = 50e-6, 
            time_steps: int = 100, 
            edm_mag: float = 5.4e-18,
            backgrounds: Dict[str, float] = {"Bz_n0": 1.0, "Bz_n1": 1.0, "Br_n0": 1.0}
            ) -> Dict:
        """
        Run EDM simulation with optional background fields
        
        Args:
            backgrounds: Dict with background field types and strengths in ppm
                        e.g. {"Br_n0": 1.0, "Bz_n1": 2.0}
        """
        print(f"\nSimulating {n_muons} muons for {t_max*1e6:.1f} us with EDM={edm_mag} ecm")
        print(f"Backgrounds [ppm]: {backgrounds}")
        
        # Times
        t = np.linspace(0, t_max, time_steps)
        T_g2 = 2 * np.pi / self.omega_a
        t_mod = np.mod(t, T_g2)
        
        # Calculate EDM tilt in rest frame
        edm_tilt = self.precession.edm_tilt_angle(edm_mag, self.b_field.nominal_field)
        edm_tilt_lab = edm_tilt / self.muon.magic_gamma 
        
        # Initialise arrays
        results = {
            't': t,
            't_mod': t_mod, 
            'T_g2': T_g2,
            'phi_c': np.zeros((time_steps, n_muons)),
            'phi_a': np.zeros((time_steps, n_muons)),
            'x': np.zeros((time_steps, n_muons)),
            'y': np.zeros((time_steps, n_muons)),
            'sx': np.zeros((time_steps, n_muons)),
            'sy': np.zeros((time_steps, n_muons)),
            'sz': np.zeros((time_steps, n_muons)),
            'theta_y': np.zeros((time_steps, n_muons)),
            'omega_a': self.omega_a,
            'omega_c': self.omega_c,
            'magic_gamma': self.muon.magic_gamma,
            'n_muons': n_muons,
            'edm_mag': edm_mag,
            'edm_tilt': edm_tilt,
            'edm_tilt_lab': edm_tilt_lab,
            'backgrounds': backgrounds
        }

        # background_corrections = {
        #     "Br_n0": 0.0,
        #     "Bz_n0": 0.0, 
        #     "Bz_n1": 0.0
        # }
        # # backgrounds["delta_tilt"] = 0.0

        # # Calculate background field effects
        # for bg_field, strength_ppm in backgrounds.items():
        #     if bg_field == "Br_n0":
        #         # Radial field N=0: uniform radial field
        #         # Creates EDM-like tilt that oscillates IN-PHASE with omega_eta (EDM)
        #         # This mimics an EDM signal and is a major systematic
        #         delta_tilt = strength_ppm * 1e-6 # Convert ppm to radians
        #         background_corrections["Br_n0"] += delta_tilt
                
        #     elif bg_field == "Bz_n0":
        #         # Longitudinal field N=0 (monopole): uniform longitudinal field  
        #         # Creates tilt in z-direction, oscillates OUT-OF-PHASE with omega_eta (EDM)
        #         # Should not impact EDM
        #         delta_tilt = strength_ppm * 1e-6 # Convert ppm to radians
        #         background_corrections["Bz_n0"] += delta_tilt
                
        #     elif bg_field == "Bz_n1":
        #         # Longitudinal field N=1 (dipole): field varies as cos(phi) around ring
        #         # Creates z-direction tilt that varies with position
        #         # Can induce oscillation in-phase with omega_eta (EDM)
        #         delta_tilt = strength_ppm * 1e-6 # Convert ppm to radians
        #         background_corrections["Bz_n1"] += delta_tilt
        
        # Time evolution (always in rest frame)
        for i, time in enumerate(t):
            # Cyclotron motion
            phi_c = self.omega_c * time
            x = self.ring.radius * np.cos(phi_c)
            y = self.ring.radius * np.sin(phi_c)
            
            # Spin precession  
            phi_a = self.omega_a * time
            
            # Rest frame spin components            
            s_vec = np.array([
                np.cos(phi_a),
                0.0,
                np.sin(phi_a)
            ]) # shape=(3, n_muons)
                  
            ################################################################################

            # Apply EDM tilt if non-zero
            if edm_mag != 0:
                # EDM tilts plane radially (rotates spin vector around z-axis)
                # s_vec_org = s_vec.copy  # Save original vector
                # s_vec[0] -= edm_tilt * s_vec_org[1] # sy rotates into sx (reduced at +-pi/2)
                s_vec[1] += (np.sin(phi_a) * edm_tilt) # sx rotates into sy (increased at +-pi/2)
                # s_vec[2] = s_vec[2] # unchanged

            # if edm_mag != 0:
            #     s_vec[0] = np.cos(edm_tilt) * np.cos(phi_a)
            #     s_vec[1] = np.sin(edm_tilt) * np.cos(phi_a) 
            #     s_vec[2] = s_vec[2] # unchanged

            # if edm_mag != 0:
            #     # EDM tilts plane radially (rotates spin vector around z-axis)
            #     s_vec_org = s_vec.copy()  # Save original vector
            #     s_vec[0] -= edm_tilt * s_vec_org[1] # sy rotates into sx (reduced at +-pi/2)
            #     s_vec[1] += edm_tilt * s_vec_org[0] # sx rotates into sy (increased at +-pi/2)
            #     s_vec[2] = s_vec[2] # unchanged
                
            ################################################################################

            # Apply backgrounds 
            if "Br_n0" in backgrounds.keys():
                # Radial field N=0: constant background around ring
                # Tilts precession plane (rotates spin vector around z-axis)
                bkg_tilt = backgrounds["Br_n0"]
                # print(backgrounds["Br_n0"])
                s_vec_org = s_vec.copy() # Save original vector 
                s_vec[0] -= bkg_tilt * s_vec_org[1] # sy rotates into sx (reduced at +-pi/2)
                s_vec[1] += bkg_tilt * s_vec_org[0] # sx rotates into sy (increased at +-pi/2)
                s_vec[2] = s_vec[2] # unchanged

            if "Bz_n0" in backgrounds.keys():                  
                # Longitudinal field N=0: constant background around ring
                # Pitches the precession plane forward: (rotates vector around x-axis) 
                bkg_tilt = backgrounds["Bz_n0"]
                s_vec_org = s_vec.copy() # Save original vector
                s_vec[0] = s_vec[0] # unchanged
                s_vec[1] += bkg_tilt * s_vec_org[2] # sz rotates into sy (increased at +-pi/2)
                s_vec[2] -= bkg_tilt * s_vec_org[1] # sy rotates into sz (decreased at +-pi/2)

            if "Bz_n1" in backgrounds.keys():
                # Longitudinal field N=1: dipole longitudinal field varying as cos(phi_c)
                # Pitches precession plane forward/backward depending on azimuthal position
                bkg_tilt = backgrounds["Bz_n1"] 
                s_vec_org = s_vec.copy() # Save original vector
                dipole = np.cos(phi_c)  # Varies around ring azimuth
                s_vec[0] = s_vec[0] # unchanged
                s_vec[1] += bkg_tilt * dipole * s_vec_org[2] # sz rotates into sy (increased at +-pi/2)
                s_vec[2] -= bkg_tilt * dipole * s_vec_org[1] # sy rotates into sz (decreased at +-pi/2)

            ################################################################################
            
            # Vertical angle
            # spin_mag = np.sqrt(s_vec[0]**2 + s_vec[1]**2 + s_vec[2]**2)
            s_mag = np.linalg.norm(s_vec, axis=0)
            theta_y = np.arcsin(s_vec[1] / s_mag)
            
            # Store results
            results['phi_c'][i] = phi_c
            results['phi_a'][i] = phi_a
            results['x'][i] = x
            results['y'][i] = y
            results['sx'][i] = s_vec[0]
            results['sy'][i] = s_vec[1]
            results['sz'][i] = s_vec[2]
            results['theta_y'][i] = theta_y

        # Return results
        return results
    
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