import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter

class Plotter:
    """Class for plotting simulation results"""
    
    def __init__(self):
        # Style file path handling
        self.style_path = Path(__file__).parent / "edm.mplstyle"
        # Check if style file exists 
        if self.style_path.exists():
            plt.style.use(str(self.style_path))
        else:
            print(f"Warning: Style file not found at {self.style_path}")
        # Some member variables 
        self.color="#C41E3A" # one colour for everything, for now
        self.s = 5 # scatter plot size 
    
    def wiggle_modulo(self, results, title: Optional[str] = None, 
                      show: Optional[bool] = True, out_path: Optional[str] = None):
        """Plot wiggle vs modulo time"""
        # Create a figure with 1x2 subplots (1 row, 2 columns)
        fig, ax = plt.subplots(1, 2, figsize=(2*6.4, 4.8))
        # Loop through muons
        for muon_idx in range(results['n_muons']): 
            # Left: sin(phi_a) vs t_mod 
            ax[0].scatter(
                results['t_mod'] * 1e6, # s->us
                np.sin(results['phi_a'][:, muon_idx]),
                color=self.color, s=self.s 
            )
            # Right: sin(phi_a) vs sin(phi_c)
            ax[1].scatter(
                np.sin(results['phi_c'][:, muon_idx]),
                np.sin(results['phi_a'][:, muon_idx]),
                color=self.color, s=self.s 
            )
        # Format
        ax[0].set_xlabel(r'$\mathrm{mod}(t, T_{a})$ [$\mu$s]')
        ax[0].set_ylabel(r'$\sin{\phi_a}$')
        # 
        ax[1].set_xlabel(r'$\sin{\phi_c}$')
        ax[1].set_ylabel(r'$\sin{\phi_a}$')
        # Optional figure title
        if title:
            plt.suptitle(title)
        # Save
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path)
            print(f"\tWrote {out_path}")
        # Display
        if show:
            plt.show()

    def sy_modulo(self, results, title: Optional[str] = None, 
                    show: Optional[bool] = True, out_path: Optional[str] = None):
        """ Vertical polarisation vs time modulo T_a"""
        # Create a figure 
        fig, ax = plt.subplots() 
        for muon_idx in range(results['n_muons']):
            ax.scatter(
                results['t_mod']*1e6, # s->us
                results['sy'][:,muon_idx], 
                color=self.color, s=self.s
            )
        # Format
        ax.set_xlabel(r'$\text{mod}(t, T_{a})$ [$\mu$s]')
        ax.set_ylabel(r'$S_{y}$')
        # Add rest frame tilt to plot
        delta_rest_mrad = results["edm_tilt"] * 1e3 # rad->mrad
        ax.text(0.95, 0.95, rf'$\delta^* = {delta_rest_mrad:.2f}$ mrad', 
                transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right')
        # Optional figure title
        if title:
            plt.suptitle(title)
        # Save
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path)
            print(f"\tWrote {out_path}")
        # Display
        if show:
            plt.show()

    def spin_phase_summary(self, results, title: Optional[str] = None, 
                           show: Optional[bool] = True, out_path: Optional[str] = None):
        """ Spin components versus phase"""
        # Create a figure with 3x2 subplots (3 rows, 2 columns)
        fig, ax = plt.subplots(3, 2, figsize=(2*6.4, 3*4.8))

        # --- Top Left Panel: sin(φ_s) for many muons ---
        for muon_idx in range(results['n_muons']):
            # sx vs sin(phi_a)
            ax[0, 0].scatter(
                np.sin(results['phi_a'][:, muon_idx]),
                results['sx'][:, muon_idx],
                color=self.color, s=self.s 
            )
            # sx vs sin(phi_c)
            ax[0, 1].scatter(
                np.sin(results['phi_c'][:, muon_idx]),
                results['sx'][:, muon_idx],
                color=self.color, s=self.s 
            )
            # sy vs sin(phi_a)
            ax[1, 0].scatter(
                np.sin(results['phi_a'][:, muon_idx]),
                results['sy'][:, muon_idx],
                color=self.color, s=self.s 
            )
            # sy vs sin(phi_c)
            ax[1, 1].scatter(
                np.sin(results['phi_c'][:, muon_idx]),
                results['sy'][:, muon_idx],
                color=self.color, s=self.s 
            )
            # sz vs sin(phi_a)
            ax[2, 0].scatter(
                np.sin(results['phi_a'][:, muon_idx]),
                results['sz'][:, muon_idx],
                color=self.color, s=self.s 
            )
            # sz vs sin(phi_c)
            ax[2, 1].scatter(
                np.sin(results['phi_c'][:, muon_idx]),
                results['sz'][:, muon_idx],
                color=self.color, s=self.s 
            )
        # Format
        ax[0, 0].set_xlabel(r'$\sin{\phi_a}$')
        ax[0, 0].set_ylabel(r'$S_{x}$')
        ax[0, 1].set_xlabel(r'$\sin{\phi_c}$')
        ax[0, 1].set_ylabel(r'$S_{x}$')
        #
        ax[1, 0].set_xlabel(r'$\sin{\phi_a}$')
        ax[1, 0].set_ylabel(r'$S_{y}$')
        ax[1, 1].set_xlabel(r'$\sin{\phi_c}$')
        ax[2, 1].set_ylabel(r'$S_{y}}$')
        #
        ax[2, 0].set_xlabel(r'$\sin{\phi_a}$')
        ax[2, 0].set_ylabel(r'$S_{z}$')
        ax[2, 1].set_xlabel(r'$\sin{\phi_c}$')
        ax[2, 1].set_ylabel(r'$S_{z}$')
        # Optional figure title
        if title:
            plt.suptitle(title)
        # Save
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path)
            print(f"\tWrote {out_path}")
        # Display
        if show:
            plt.show()


    def spin_3d(self, results, title: Optional[str] = None, 
                show: Optional[bool] = True, out_path: Optional[str] = None):
        """ Plot spin in 3D"""
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) 
        # --- Spin trajectories ---
        for muon_idx in range(results['n_muons']):
            ax.plot(
                results['sx'][:, muon_idx], 
                results['sz'][:, muon_idx], 
                results['sy'][:, muon_idx], 
                color=self.color, linewidth=1.5
            )
        # Format
        ax.set_xlabel(r'$S_{x}$') 
        ax.set_ylabel(r'$S_{z}$') 
        ax.set_zlabel(r'$S_{y}$') 
        # Set viewing angle (elevation, azimuth)
        # ax.view_init(elev=20, azim=130)  
        # Optional figure title
        if title:
            plt.suptitle(title)
        # Save
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path)
            print(f"\tWrote {out_path}")
        # Display
        if show:
            plt.show()




