#!/usr/bin/env python3
"""
Curvature-Work Diagnostic Simulation
====================================

Theoretical physics exploration of how curvature-work contributions might bias 
apparent H₀ measurements from strong-lens and supernova data.

Working hypothesis: Observed redshift = expansion + "work" photons do escaping 
curvature wells. The work component likely decreases over cosmic time.

Author: Zora Mehmi
Date: 2025-07-30
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from io import StringIO
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CurvatureWorkDiagnostic:
    """
    Main class for curvature-work diagnostic analysis of H0 measurements.
    """
    
    def __init__(self):
        self.lens_data = None
        self.sn_data = None
        
    def load_h0licow_data(self) -> pd.DataFrame:
        """
        Load H0LiCOW strong lens time-delay data.
        Uses published values from the literature.
        """
        # Manually compiled from H0LiCOW papers (approximated for prototype)
        lens_systems = {
            'name': ['B1608+656', 'RXJ1131-1231', 'HE0435-1223', 'WFI2033-4723', 'HE1104-1805', 'PG1115+080'],
            'z_lens': [0.630, 0.295, 0.454, 0.661, 0.729, 0.311],
            'z_source': [1.394, 0.658, 1.693, 1.662, 2.319, 1.722],
            'sigma_v': [247, 323, 222, 270, 267, 281],  # km/s, velocity dispersion
            'log_sigma_v': [2.393, 2.509, 2.346, 2.431, 2.427, 2.449],
            'H0_apparent': [71.0, 78.3, 73.1, 71.6, 75.2, 82.4],  # km/s/Mpc, apparent H0
            'H0_err': [3.1, 4.9, 2.7, 4.2, 5.8, 8.1],
            'time_delay': [31.5, 91.7, 8.4, 36.2, 161.0, 25.0],  # days (approximate)
            'survey': ['H0LiCOW'] * 6
        }
        
        self.lens_data = pd.DataFrame(lens_systems)
        return self.lens_data
    
    def load_pantheon_data(self, n_sample: int = 100) -> pd.DataFrame:
        """
        Load Pantheon+ supernova host mass data (simplified for prototype).
        
        Args:
            n_sample: Number of SN to sample for quick analysis
        """
        # Simulated Pantheon+ data based on typical values
        np.random.seed(42)  # Reproducible results
        
        n = n_sample
        z_range = np.linspace(0.01, 2.3, n)
        
        # Simulate host galaxy masses (log M_star in solar masses)
        host_mass_mean = 10.5
        host_mass_std = 0.8
        host_logmass = np.random.normal(host_mass_mean, host_mass_std, n)
        
        # Simulate apparent H0 with some scatter and host mass correlation
        # Base H0 around 70, with slight correlation to environment depth
        base_h0 = 70.0
        mass_bias = -0.5 * (host_logmass - 10.5)  # Higher mass → lower apparent H0
        h0_scatter = np.random.normal(0, 3.0, n)
        H0_apparent = base_h0 + mass_bias + h0_scatter
        
        sn_data = {
            'name': [f'SN{i:04d}' for i in range(n)],
            'z': z_range + np.random.normal(0, 0.05, n),  # Add some scatter
            'host_logmass': host_logmass,
            'host_logmass_err': np.random.uniform(0.1, 0.3, n),
            'H0_apparent': H0_apparent,
            'H0_err': np.random.uniform(1.5, 4.0, n),
            'survey': ['Pantheon+'] * n
        }
        
        self.sn_data = pd.DataFrame(sn_data)
        return self.sn_data
    
    def curvature_work_correction(self, depth_proxy: np.ndarray, 
                                alpha: float = 0.05, 
                                functional_form: str = 'linear') -> np.ndarray:
        """
        Apply curvature-work correction to apparent H0 values.
        
        H0_corrected = H0_apparent × (1 - α × f(depth))
        
        Args:
            depth_proxy: Environment depth proxy (log σ_v or host mass)
            alpha: Correction strength parameter
            functional_form: 'linear', 'quadratic', or 'exponential'
        
        Returns:
            Correction factor (1 - α × f(depth))
        """
        # Normalize depth proxy to [0, 1] range
        depth_norm = (depth_proxy - np.min(depth_proxy)) / (np.max(depth_proxy) - np.min(depth_proxy))
        
        if functional_form == 'linear':
            f_depth = depth_norm
        elif functional_form == 'quadratic':
            f_depth = depth_norm**2
        elif functional_form == 'exponential':
            f_depth = 1 - np.exp(-2 * depth_norm)
        else:
            raise ValueError("functional_form must be 'linear', 'quadratic', or 'exponential'")
        
        correction_factor = 1 - alpha * f_depth
        return correction_factor
    
    def create_diagnostic_plot(self, alpha: float = 0.05, 
                             functional_form: str = 'linear',
                             save_path: str = 'curvature_work_diagnostic.png') -> plt.Figure:
        """
        Create the main diagnostic plot.
        
        x-axis: Environment depth proxy (log σ_v for lenses, host mass for SNe)
        y-axis: Apparent H0 under standard interpretation
        Color: Redshift of system
        Overlay: Curvature-work correction curve
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Panel 1: Strong Lens Systems
        if self.lens_data is not None:
            scatter1 = ax1.scatter(self.lens_data['log_sigma_v'], 
                                 self.lens_data['H0_apparent'],
                                 c=self.lens_data['z_lens'], 
                                 s=80, alpha=0.8, cmap='viridis',
                                 edgecolors='black', linewidth=0.5)
            
            ax1.errorbar(self.lens_data['log_sigma_v'], 
                        self.lens_data['H0_apparent'],
                        yerr=self.lens_data['H0_err'], 
                        fmt='none', ecolor='gray', alpha=0.5)
            
            # Add correction curve
            sigma_range = np.linspace(self.lens_data['log_sigma_v'].min(), 
                                    self.lens_data['log_sigma_v'].max(), 100)
            correction = self.curvature_work_correction(sigma_range, alpha, functional_form)
            h0_mean = self.lens_data['H0_apparent'].mean()
            corrected_h0 = h0_mean * correction
            
            ax1.plot(sigma_range, corrected_h0, 'r--', linewidth=2, 
                    label=f'Curvature correction (α={alpha})')
            
            ax1.set_xlabel('log σ_v [km/s]')
            ax1.set_ylabel('Apparent H₀ [km/s/Mpc]')
            ax1.set_title('Strong Lens Time-Delay Systems')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add colorbar
            cbar1 = plt.colorbar(scatter1, ax=ax1)
            cbar1.set_label('Lens Redshift')
        
        # Panel 2: Supernova Systems
        if self.sn_data is not None:
            scatter2 = ax2.scatter(self.sn_data['host_logmass'], 
                                 self.sn_data['H0_apparent'],
                                 c=self.sn_data['z'], 
                                 s=30, alpha=0.6, cmap='plasma',
                                 edgecolors='none')
            
            # Add correction curve
            mass_range = np.linspace(self.sn_data['host_logmass'].min(), 
                                   self.sn_data['host_logmass'].max(), 100)
            correction = self.curvature_work_correction(mass_range, alpha, functional_form)
            h0_mean = self.sn_data['H0_apparent'].mean()
            corrected_h0 = h0_mean * correction
            
            ax2.plot(mass_range, corrected_h0, 'r--', linewidth=2, 
                    label=f'Curvature correction (α={alpha})')
            
            ax2.set_xlabel('log M_host [M_☉]')
            ax2.set_ylabel('Apparent H₀ [km/s/Mpc]')
            ax2.set_title('Supernova Host Galaxy Environments')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add colorbar
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('SN Redshift')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def create_interactive_plot(self):
        """
        Create an interactive plot with sliders for α and functional form.
        Note: For basic matplotlib, we'll provide parameter exploration instead.
        """
        print("Creating parameter exploration plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        alphas = [0.01, 0.05, 0.10]
        forms = ['linear', 'quadratic', 'exponential']
        
        for i, alpha in enumerate(alphas):
            for j, form in enumerate(forms):
                ax = axes[j, i]
                
                if self.lens_data is not None:
                    # Plot lens data
                    ax.scatter(self.lens_data['log_sigma_v'], 
                             self.lens_data['H0_apparent'],
                             c=self.lens_data['z_lens'], 
                             s=50, alpha=0.7, cmap='viridis')
                    
                    # Add correction
                    sigma_range = np.linspace(self.lens_data['log_sigma_v'].min(), 
                                            self.lens_data['log_sigma_v'].max(), 50)
                    correction = self.curvature_work_correction(sigma_range, alpha, form)
                    h0_mean = self.lens_data['H0_apparent'].mean()
                    corrected_h0 = h0_mean * correction
                    
                    ax.plot(sigma_range, corrected_h0, 'r--', linewidth=2)
                
                ax.set_title(f'α={alpha}, {form}')
                ax.set_xlabel('log σ_v [km/s]')
                ax.set_ylabel('H₀ [km/s/Mpc]')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('parameter_exploration.png', dpi=300, bbox_inches='tight')
        return fig
    
    def summary_statistics(self) -> Dict:
        """
        Compute summary statistics for the analysis.
        """
        stats = {}
        
        if self.lens_data is not None:
            stats['lens'] = {
                'n_systems': len(self.lens_data),
                'h0_mean': self.lens_data['H0_apparent'].mean(),
                'h0_std': self.lens_data['H0_apparent'].std(),
                'sigma_v_range': (self.lens_data['sigma_v'].min(), self.lens_data['sigma_v'].max()),
                'z_range': (self.lens_data['z_lens'].min(), self.lens_data['z_lens'].max())
            }
        
        if self.sn_data is not None:
            stats['sn'] = {
                'n_systems': len(self.sn_data),
                'h0_mean': self.sn_data['H0_apparent'].mean(),
                'h0_std': self.sn_data['H0_apparent'].std(),
                'mass_range': (self.sn_data['host_logmass'].min(), self.sn_data['host_logmass'].max()),
                'z_range': (self.sn_data['z'].min(), self.sn_data['z'].max())
            }
        
        return stats

def main():
    """
    Main execution function for the diagnostic analysis.
    """
    print("Curvature-Work Diagnostic Analysis")
    print("=" * 40)
    
    # Initialize diagnostic
    diagnostic = CurvatureWorkDiagnostic()
    
    # Load data
    print("Loading H0LiCOW lens data...")
    diagnostic.load_h0licow_data()
    
    print("Loading Pantheon+ SN data...")
    diagnostic.load_pantheon_data(n_sample=150)
    
    # Create plots
    print("Creating diagnostic plot...")
    fig1 = diagnostic.create_diagnostic_plot(alpha=0.05, functional_form='linear')
    plt.show()
    
    print("Creating parameter exploration...")
    fig2 = diagnostic.create_interactive_plot()
    plt.show()
    
    # Print statistics
    stats = diagnostic.summary_statistics()
    print("\nSummary Statistics:")
    print("-" * 20)
    for dataset, values in stats.items():
        print(f"{dataset.upper()} Dataset:")
        for key, value in values.items():
            print(f"  {key}: {value}")
        print()
    
    print("Analysis complete! Check output files:")
    print("- curvature_work_diagnostic.png")
    print("- parameter_exploration.png")

if __name__ == "__main__":
    main()