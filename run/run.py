# Preamble and imports
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load custom style
plt.style.use('../src/utils/edm.mplstyle')

# Add the src subdirectories to Python path
project_root = Path("..").resolve()
sys.path.append(str(project_root / "src" / "core"))
sys.path.append(str(project_root / "src" / "utils"))

# Import internal modules
from simulate import Simulation
from plotting import Plotter

print("Muon EDM toy simulation framework")

print("=" * 40)
print(f"Project root: {project_root}")

# Create the spin precession simulator
simulation = Simulation()

# Create plotter
plotter = Plotter()

############################
# Small EDM, no backgrounds
############################

# Run the simulation
config = {
    'n_muons': 1,           # Number of muons to simulate (deterministic at present, one is as good as a million!)
    't_max': 50e-6,         # Total simulation time in seconds
    'time_steps': int(5e3), # Number of time steps in the simulation
    'edm_mag': 1e-20,       # EDM magnitude in ecm 
    'backgrounds': {"Bz_n0": 0.0, "Bz_n1": 0.0, "Br_n0": 0.0} # Background field strengths in ppm
}       
results = simulation.run(**config)

# Make plots 
dir="small_edm_no_bkg"
plotter.spin_3d(results, out_path=f'../img/{dir}/plt_spin_3d.png')
plotter.wiggle_modulo(results, out_path=f'../img/{dir}/gr_1x2_wiggle_modulo.png')
plotter.sy_modulo(results, out_path=f'../img/{dir}/gr_sy_modulo.png')
plotter.spin_phase_summary(results, out_path=f'../img/{dir}/gr_3x2_spin_phase_summary.png')