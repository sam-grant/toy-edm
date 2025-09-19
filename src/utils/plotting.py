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
        self.s = 10 # scatter plot size 
    
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

    # def sy_modulo(self, results, title: Optional[str] = None, 
    #                 show: Optional[bool] = True, out_path: Optional[str] = None):
    #     """ Vertical polarisation vs time modulo T_a"""
    #     # Create a figure 
    #     fig, ax = plt.subplots() 
    #     for muon_idx in range(results['n_muons']):
    #         ax.scatter(
    #             results['phi_a'], # *1e6, # s->us
    #             results['sy'][:,muon_idx], 
    #             color=self.color, s=self.s
    #         )
    #     # Format
    #     ax.set_xlabel(r'$phi_{a}')
    #     ax.set_ylabel(r'$S_{y}$')
    #     # Add rest frame tilt to plot
    #     delta_rest_mrad = results["edm_tilt"] * 1e3 # rad->mrad
    #     if delta_rest_mrad != 0:
    #         ax.text(0.95, 0.95, rf'$\delta^* = {delta_rest_mrad:.1f}$ mrad', 
    #                 transform=ax.transAxes, 
    #                 verticalalignment='top', horizontalalignment='right')
    #     # Math formatting
    #     ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2), useMathText=True)
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

    def sy_modulo(self, results, title: Optional[str] = None, 
                show: Optional[bool] = True, out_path: Optional[str] = None):
        """ Vertical polarisation vs time modulo T_a"""
        # Create a figure 
        fig, ax = plt.subplots() 
        for muon_idx in range(results['n_muons']):
            ax.scatter(
                results['t_mod'] * 1e6,  # s->us 
                results['sy'][:,muon_idx], 
                color=self.color, s=self.s
            )
        # Format
        ax.set_xlabel(r'$\mathrm{mod}(t, T_{a})$ [$\mu$s]')
        ax.set_ylabel(r'$S_{y}$')
        # Add rest frame tilt to plot
        delta_rest_mrad = results["edm_tilt"] * 1e3 # rad->mrad
        if delta_rest_mrad != 0:
            ax.text(0.95, 0.95, rf'$\delta^* = {delta_rest_mrad:.1f}$ mrad', 
                    transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right')
        # Math formatting
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2), useMathText=True)
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


    # def sy_modulo(self, results, title: Optional[str] = None,
    #             show: Optional[bool] = True, out_path: Optional[str] = None):
    #     """Plot vertical polarisation vs (t mod T_a) and vs sin(phi_a) in a 1x2 frame."""

    #     # Create a 1x2 subplot
    #     fig, ax = plt.subplots(1, 2, figsize=(2*6.4, 4.8))

    #     # Loop over muons
    #     for muon_idx in range(results['n_muons']):
    #         # Left: S_y vs t_mod
    #         ax[0].scatter(
    #             results['t_mod'] * 1e6,  # s → µs
    #             results['sy'][:, muon_idx],
    #             s=self.s, color=self.color
    #         )

    #         # Right: S_y vs sin(phi_a)
    #         ax[1].scatter(
    #             np.sin(results['phi_a']),
    #             results['sy'][:, muon_idx],
    #             s=self.s, color=self.color
    #         )

    #     # Rest-frame EDM tilt (only annotate once per config)
    #     delta_rest_mrad = results["edm_tilt"] * 1e3  # rad→mrad
    #     if delta_rest_mrad != 0:
    #         ax[1].text(
    #             0.95, 0.95,
    #             rf'$\delta^* = {delta_rest_mrad:.1f}$ mrad',
    #             transform=ax[1].transAxes,
    #             verticalalignment='top', horizontalalignment='right'
    #         )

    #     # Formatting left panel
    #     ax[0].set_xlabel(r'$\mathrm{mod}(t, T_{a})$ [$\mu$s]')
    #     ax[0].set_ylabel(r'$S_{y}$')
    #     ax[0].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2), useMathText=True)
    #     ax[0].legend()

    #     # Formatting right panel
    #     ax[1].set_xlabel(r'$\sin(\phi_{a})$')
    #     ax[1].set_ylabel(r'$S_{y}$')
    #     ax[1].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2), useMathText=True)

    #     # Optional title
    #     if title:
    #         fig.suptitle(title)

    #     # Save
    #     if out_path:
    #         Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    #         plt.savefig(out_path)
    #         print(f"\tWrote {out_path}")

    #     # Show
    #     if show:
    #         plt.show()

    def sy_modulo_overlay(self, results_dict, title: Optional[str] = None, 
                        show: Optional[bool] = True, out_path: Optional[str] = None):
        """
        Overlay many vertical polarisation plots:
        Left:  S_y vs (t mod T_a)
        Right: S_y vs sin(phi_a)
        """
        # Create a 1x2 subplot
        fig, ax = plt.subplots(1, 2, figsize=(2*6.4, 4.8), sharey=True)

        # Plot each set of results from the dictionary
        for i, (label, results) in enumerate(results_dict.items()):
            for muon_idx in range(results['n_muons']):
                # Left: S_y vs t_mod
                ax[0].scatter(
                    results['t_mod'] * 1e6,  # s → µs
                    results['sy'][:, muon_idx],
                    label=label if muon_idx == 0 else "",
                    s=self.s
                )

                # Right: S_y vs sin(phi_a)
                ax[1].scatter(
                    np.sin(results['phi_a']),
                    results['sy'][:, muon_idx],
                    label=label if muon_idx == 0 else "",
                    s=self.s
                )

            # # Rest-frame EDM tilt (annotate once per config, on right panel)
            # delta_rest_mrad = results["edm_tilt"] * 1e3  # rad→mrad
            # if delta_rest_mrad != 0:
            #     ax[1].text(
            #         0.95, 0.95,
            #         rf'$\delta^* = {delta_rest_mrad:.1f}$ mrad',
            #         transform=ax[1].transAxes,
            #         verticalalignment='top', horizontalalignment='right'
            #     )

        # Format left panel
        ax[0].set_xlabel(r'$\mathrm{mod}(t, T_{a})$ [$\mu$s]')
        ax[0].set_ylabel(r'$S_{y}$')
        ax[0].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2), useMathText=True)
        ax[0].legend()

        # Format right panel
        ax[1].set_xlabel(r'$\sin(\phi_{a})$')
        ax[1].set_ylabel(r'$S_{y}$')
        ax[1].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2), useMathText=True)
        ax[1].legend()

        # Tight layout
        plt.tight_layout()

        # Optional figure title
        if title:
            fig.suptitle(title)

        # Save
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path)
            print(f"\tWrote {out_path}")

        # Show
        if show:
            plt.show()

    # def sy_modulo_overlay(self, results_dict, title: Optional[str] = None, 
    #                      show: Optional[bool] = True, out_path: Optional[str] = None):
    #     """
    #     Overlay many vertical polarisation vs time modulo T_a plots
    #     """
    #     # Create a figure
    #     fig, ax = plt.subplots() # figsize=(6.4*1.25, 4.8*1.25))
        
    #     # Plot each set of results from the dictionary
    #     for i, (label, results) in enumerate(results_dict.items()):
            
    #         for muon_idx in range(results['n_muons']):
    #             ax.scatter(
    #                 results['t_mod'] * 1e6,  # s -> us
    #                 results['sy'][:, muon_idx],
    #                 label=label if muon_idx == 0 else "",  # Label once per configuration
    #                 s=self.s
    #             )
        
    #     # Format
    #     ax.set_xlabel(r'$\text{mod}(t, T_{a})$ [$\mu$s]')
    #     ax.set_ylabel(r'$S_{y}$')
        
    #     # Add legend
    #     ax.legend() # ncols=2) # , fontsize=16) # , bbox_to_anchor=(0.0, -0.2) )
        
    #     # Math formatting
    #     ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2), useMathText=True)
        
    #     # # Use plt.tight_layout() to automatically adjust subplot parameters
    #     # 
    #     ylims = ax.get_ylim()
    #     ax.set_ylim(-ylims[1]*(1+1), ylims[1]*(1+1))

    #     plt.tight_layout()

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
        #
        ax[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(-2,2), useMathText=True)
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
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) # , figsize=(10, 8))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
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
        ax.legend(bbox_to_anchor=(1.1, 1.0)) # Hack to extend the frame
        ax.ticklabel_format(style='scientific', axis='z', scilimits=(-2,2), useMathText=True)

        plt.tight_layout()

        # Optional figure title
        if title:
            plt.suptitle(title)
        # Save
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, bbox_inches='tight')
            print(f"\tWrote {out_path}")
        # Display
        if show:
            plt.show()

    # def spin_3d_grid(self, results_dict, main_title: Optional[str] = None, 
    #                 show: Optional[bool] = True, out_path: Optional[str] = None):
    #     """
    #     Plots a 2x2 grid of 3D spin trajectories.

    #     Args:
    #         results_dict (Dict[str, Dict]): A dictionary where keys are plot labels
    #                                         and values are simulation result dictionaries.
    #                                         Must contain exactly 4 entries.
    #         panel_titles (List[str]): A list of 4 titles for each panel.
    #         main_title (Optional[str]): The main title for the entire figure.
    #         show (Optional[bool]): Whether to display the plot.
    #         out_path (Optional[str]): Path to save the figure.
    #     """
    #     # Create a 2x2 subplot grid
    #     fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), subplot_kw={"projection": "3d"})
        
    #     # Flatten the axes array for easy iteration
    #     axes = axes.flatten()
        
    #     # Iterate through the results and plot each panel
    #     for i, (label, results) in enumerate(results_dict.items()):
    #         ax = axes[i]
            
    #         # --- Spin trajectories ---
    #         for muon_idx in range(results['n_muons']):
    #             ax.plot(
    #                 results['sz'][:, muon_idx], 
    #                 results['sx'][:, muon_idx], 
    #                 results['sy'][:, muon_idx], 
    #                 linewidth=1.5,
    #                 color=self.color,
    #                 label=""
    #             )

    #         # Set panel title
    #         ax.set_title(label, fontsize=16)
            
    #         # Format axes
    #         ax.set_xlabel(r'$S_{z}$', labelpad=10) 
    #         ax.set_ylabel(r'$S_{x}$', labelpad=10) 
    #         ax.set_zlabel(r'$S_{y}$', labelpad=10)
            
    #         # Set tick label format
    #         ax.ticklabel_format(style='scientific', axis='z', scilimits=(-2,2), useMathText=True)
            
    #         # Add legend (hack to stop clipping)
    #         # ax.legend(bbox_to_anchor=(1.4, 0.7))
            
    #     # Set the main title for the entire figure
    #     if main_title:
    #         fig.suptitle(main_title, fontsize=16, y=0.95)
        
    #     # Adjust subplot parameters for a tight layout
    #     plt.tight_layout()
        
    #     # Save the plot
    #     if out_path:
    #         Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    #         plt.savefig(out_path, bbox_inches='tight')
    #         print(f"\tWrote {out_path}")
            
    #     # Display the plot
    #     if show:
    #         plt.show()


    def spin_3d_overlay(self, results_dict, title: Optional[str] = None, 
                show: Optional[bool] = True, out_path: Optional[str] = None):
        """ Plot spin in 3D with improved layout and legend positioning"""
        
        # Create figure with larger margins
        fig = plt.figure(figsize=(6.4*100, 4.8))
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
                    # alpha=0.7 # , linewidth=1.5
                )
        
        # Format axes
        ax.set_xlabel(r'$S_{z}$', labelpad=10) 
        ax.set_ylabel(r'$S_{x}$', labelpad=10) 
        ax.set_zlabel(r'$S_{y}$', labelpad=10) 
        
        # Position legend outside the plot area
        ax.legend(bbox_to_anchor=(1.8, 0.7)) # , loc='upper left')
        
        # Set viewing angle (elevation, azimuth)
        # ax.view_init(elev=20, azim=130)  

        ax.ticklabel_format(style='scientific', axis='z', scilimits=(-2,2), useMathText=True)
        
        # Optional figure title with more padding
        if title:
            fig.suptitle(title) # , fontsize=14, y=0.95)
        
        plt.tight_layout()

        # Save with bbox_inches='tight' to include legend
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, bbox_inches='tight') #, dpi=300, facecolor='white')
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


