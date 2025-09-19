# Muon electric dipole moment (EDM) toy model

Simulation of muon spin precession in the Fermilab Muon g-2 storage ring with a non-zero muon EDM and magnetic field contributions away from the main vertical dipole.

>Note: This is a work in progress.

## Setup

The setup script should create a virtual Python environment with all the dependencies required to run, based on the `pyproject.toml` file. 

```bash
./setup.sh
```

## Usage

Run the simulation:
```bash
cd run
python run.py
```

## Code

**Core:**

```bash
src/core/parameters.py # Input parameters 
src/core/simulate.py   # Simulation framework 
```

**Utils:**

```bash
src/utils/edm.mplstyle
src/utils/plotting.py 
```

The toy model suite generates plots in the `img/` directory illustrating spin components under the influence of an EDM and different magnetic field components. 

## Physics 

The simulation models muon spin precession in the Muon **g-2** storage ring. The key physics may be found [here](docs/key_physics.pdf).