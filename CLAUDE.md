# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a curvature-work diagnostic simulation for theoretical physics research exploring how curvature-work contributions might bias apparent Hubble constant (H₀) measurements from strong gravitational lensing and supernova data.

**Core Hypothesis**: Observed redshift = cosmological expansion + photon energy loss from escaping gravitational wells, where the work component decreases over cosmic time.

**Key Theoretical Model**: H₀_corrected = H₀_apparent × (1 - α × f(environment_depth))

## Architecture

### Core Components

- **CurvatureWorkDiagnostic class**: Main analysis engine containing data loading, correction calculations, and visualization methods
- **Data Sources**: 
  - H0LiCOW strong lens time-delay systems (6 systems with velocity dispersions, redshifts, H₀ measurements)
  - Pantheon+ supernova host galaxy mass data (currently simulated, needs replacement with real data)
- **Correction Models**: Linear, quadratic, and exponential functional forms with adjustable α parameter
- **Visualization Pipeline**: Dual-panel plots showing environment depth vs apparent H₀ with correction overlays

### Data Structure

Strong lens systems track: name, z_lens, z_source, sigma_v, log_sigma_v, H0_apparent, H0_err, time_delay
Supernova systems track: name, z, host_logmass, host_logmass_err, H0_apparent, H0_err

Environment depth proxies: log σ_v (velocity dispersion) for lenses, log M_host for supernovae

## Development Commands

### Running Analysis
```bash
# Execute main diagnostic analysis
python3 curvature_work_diagnostic.py

# Run Jupyter notebook for interactive exploration
jupyter notebook curvature_work_analysis.ipynb
```

### Dependencies
Required packages: numpy, matplotlib, pandas, jupyter
Install with: `pip3 install numpy matplotlib pandas jupyter --user`

### Output Files
- `curvature_work_diagnostic.png`: Main two-panel diagnostic plot
- `parameter_exploration.png`: Grid showing parameter sensitivity (α values × functional forms)

## Key Implementation Details

### Curvature-Work Correction Function
The correction applies to environment depth proxies normalized to [0,1] range:
- Linear: f(depth) = depth_normalized  
- Quadratic: f(depth) = depth_normalized²
- Exponential: f(depth) = 1 - exp(-2 × depth_normalized)

### Data Integration Priority
1. Replace simulated Pantheon+ data with real host galaxy masses from PantheonPlusSH0ES/DataRelease GitHub repository
2. Add TDCOSMO lens systems beyond the current H0LiCOW sample
3. Implement redshift evolution in correction models

### Theoretical Extensions
Future development should incorporate:
- Kretschmann scalar threshold for null propagation loss
- Global curvature field energy storage
- Cosmic time dependence (shallower potential wells at later epochs)

## Research Context

This is collaborative work with Eric Henning for first observational tests against strong-lens time-delay systems (H0LiCOW/TDCOSMO) and supernova host environments. The goal is quantifying potential systematic biases in H₀ measurements due to unaccounted gravitational work effects.

Target timeline: Working prototype by 2025-07-26 for theoretical physics exploration.