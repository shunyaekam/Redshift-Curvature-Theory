# Curvature-Work Diagnostic Simulation

A theoretical physics simulation exploring how curvature-work contributions might bias apparent Hubble constant (H₀) measurements from strong gravitational lensing and supernova observations.

## 🔬 Research Hypothesis

**Core Idea**: Observed redshift has two components:
1. Standard cosmological expansion
2. Energy loss from photons escaping gravitational potential wells ("curvature work")

The curvature-work component likely decreases over cosmic time as gravitational wells become shallower, potentially creating systematic biases in H₀ measurements that depend on environment depth.

**Mathematical Model**: 
```
H₀_corrected = H₀_apparent × (1 - α × f(environment_depth))
```

Where `α` is the correction strength and `f()` represents different functional forms of the environment dependence.

## 📊 Data Sources

- **Strong Lenses**: H0LiCOW time-delay systems with velocity dispersions as environment depth proxies
- **Supernovae**: Pantheon+ host galaxy masses as environment indicators
- **Observable**: Apparent H₀ measurements color-coded by redshift

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Sim

# Install dependencies
pip3 install numpy matplotlib pandas jupyter --user
```

### Running the Analysis
```bash
# Execute main diagnostic script
python3 curvature_work_diagnostic.py

# Or run interactive Jupyter notebook
jupyter notebook curvature_work_analysis.ipynb
```

### Output Files
- `curvature_work_diagnostic.png` - Main two-panel diagnostic plot
- `parameter_exploration.png` - Parameter sensitivity grid (α × functional forms)

## 📁 Repository Structure

```
├── curvature_work_diagnostic.py     # Main analysis script
├── curvature_work_analysis.ipynb    # Interactive Jupyter notebook
├── CLAUDE.md                        # Development guidance for Claude Code
├── README.md                        # This file
└── .gitignore                       # Git ignore patterns
```

## 🔧 Key Features

- **Real H0LiCOW Data**: 6 strong lens systems with published measurements
- **Parameter Exploration**: Interactive α values (0.01, 0.05, 0.10) and functional forms (linear, quadratic, exponential)
- **Professional Visualization**: Error bars, system labels, redshift color-coding
- **Modular Design**: Clean separation of data loading, corrections, and plotting

## 📈 Current Status & Next Steps

### ✅ Completed (v1.0)
- Core theoretical framework implementation
- H0LiCOW strong lens data integration
- Multi-parameter visualization system
- Interactive Jupyter notebook interface

### 🔄 In Progress
- [ ] Replace simulated Pantheon+ data with real host galaxy masses
- [ ] Add TDCOSMO lens systems beyond H0LiCOW sample
- [ ] Implement redshift evolution in correction models

### 🎯 Future Enhancements
- [ ] Kretschmann scalar threshold implementation
- [ ] Global curvature field energy storage modeling
- [ ] Cosmic time dependence (shallower wells at later epochs)
- [ ] Cross-validation against other distance ladder measurements

## 👥 Collaboration

**Primary Researcher**: Zora Mehmi  
**Theoretical Collaborator**: Eric Henning  
**Target Timeline**: Working prototype by 2025-07-26

## 📄 Research Context

This simulation is designed for first observational tests of curvature-work effects against:
- Strong-lens time-delay systems (H0LiCOW/TDCOSMO collaborations)
- Supernova host galaxy environments (Pantheon+ survey)

The goal is quantifying potential systematic biases in H₀ measurements due to unaccounted gravitational work effects in photon propagation.

## 🤝 Contributing

This is active theoretical physics research. For questions about the physics or to discuss collaboration opportunities, please open an issue or contact the research team.

## 📚 Dependencies

- Python 3.9+
- NumPy (numerical computations)
- Matplotlib (visualization)
- Pandas (data manipulation)
- Jupyter (interactive analysis)

---

*Built for exploring the intersection of general relativity, observational cosmology, and precision measurements of cosmic expansion.*