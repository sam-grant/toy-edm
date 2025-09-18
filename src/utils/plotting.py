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
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) # , figsize=(10, 8))
        # --- Spin trajectories ---
        for muon_idx in range(results['n_muons']):
            ax.plot(
                results['sz'][:, muon_idx], 
                results['sx'][:, muon_idx], 
                results['sy'][:, muon_idx], 
                color=self.color, linewidth=1.5
            )
        # Format
        ax.set_xlabel(r'$S_{z}$') 
        ax.set_ylabel(r'$S_{x}$') 
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


    def spin_3d_overlay(self, results_dict, title: Optional[str] = None, 
                show: Optional[bool] = True, out_path: Optional[str] = None):
        """ Plot spin in 3D with improved layout and legend positioning"""
        
        # Create figure with larger margins
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Adjust subplot position to leave more room for legend
        # [left, bottom, width, height] as fractions of figure size
        # ax.set_position([0.1, 0.1, 0.65, 0.8])
        
        # Plot trajectories
        for label, results in results_dict.items():
            # --- Spin trajectories ---
            for muon_idx in range(results['n_muons']):
                ax.plot(
                    results['sz'][:, muon_idx], 
                    results['sx'][:, muon_idx], 
                    results['sy'][:, muon_idx], 
                    label=label if muon_idx == 0 else "",  # Only label first muon per config
                    alpha=0.7, linewidth=1.5
                )
        
        # Format axes
        ax.set_xlabel(r'$S_{z}$', labelpad=10) 
        ax.set_ylabel(r'$S_{x}$', labelpad=10) 
        ax.set_zlabel(r'$S_{y}$', labelpad=10) 
        
        # Position legend outside the plot area
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        
        # Set viewing angle (elevation, azimuth)
        # ax.view_init(elev=20, azim=130)  

        # ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2,2), useMathText=True)
        
        # Optional figure title with more padding
        if title:
            fig.suptitle(title) # , fontsize=14, y=0.95)
        
        # # Adjust layout to prevent clipping
        # plt.tight_layout()
        
        # Save with bbox_inches='tight' to include legend
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path) # , bbox_inches='tight') #, dpi=300, facecolor='white')
            print(f"\tWrote {out_path}")
        
        # Display
        if show:
            plt.show()
            
    # def spin_3d_overlay(self, results_dict, title: Optional[str] = None, 
    #             show: Optional[bool] = True, out_path: Optional[str] = None):
    #     """ Plot spin in 3D"""
    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) 
    #     #
    #     for label, results in results_dict.items():
    #         # --- Spin trajectories ---
    #         for muon_idx in range(results['n_muons']):
    #             ax.plot(
    #                 results['sx'][:, muon_idx], 
    #                 results['sz'][:, muon_idx], 
    #                 results['sy'][:, muon_idx], 
    #                 linewidth=1.5,
    #                 label=label,
    #                 alpha=0.7
    #             )
    #     # Format
    #     ax.set_xlabel(r'$S_{x}$') 
    #     ax.set_ylabel(r'$S_{z}$') 
    #     ax.set_zlabel(r'$S_{y}$') 
    #     ax.legend(loc="upper right")
    #     # Set viewing angle (elevation, azimuth)
    #     # ax.view_init(elev=20, azim=130)  
    #     # Optional figure title
    #     if title:
    #         plt.suptitle(title)
    #     # Save
    #     if out_path:
    #         Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    #         plt.savefig(out_path)
    #         print(f"\tWrote {out_path}")
    #     # Display
    #     if show:
    #         plt.show()


