# Constraints on Geometric Redshift Models

**Abstract**

This repository contains the data, source code, and theoretical documentation for the "Curvature-Work" research initiative. This project investigates the hypothesis that cosmological redshift contains a component derived from photon energy loss in regions of extreme spacetime curvature (`z_cw`), in addition to standard cosmological expansion (`z_exp`). The primary objective is to rigorously test this hypothesis against observational data to determine if geometric modifications to General Relativity can resolve the Hubble Tension.

Current analysis (September 2025) utilizing the Pantheon+ Type Ia supernova dataset and a Kretschmann-scalar-gated model yields a null result ($\alpha \approx 0$). This outcome constrains the magnitude of any algebraic, curvature-dependent redshift modification in the late universe to be negligible. Consequently, research is shifting toward non-linear dynamical frameworks (Ray Sheet PDEs) to investigate potential effects in micro-physical regimes near event horizons.

## Repository Organization

*   **`Documentation/`**: Contains the theoretical framework, analysis methodology, and the current draft of the research paper (`findings.md`).
*   **`CurvatureWorkH0Diagnostic/`**: Holds the Python simulation code, Bayesian inference scripts, and validation protocols used to generate the constraints.
*   **`Reference papers/`**: A collection of foundational literature regarding the Hubble Tension, General Relativity, and null geodesic propagation.

## Research Objectives

1.  **Observational Constraints**: Systematically test geometric redshift models against state-of-the-art cosmological datasets (Pantheon+, TDCOSMO) to place upper limits on deviations from standard General Relativity.
2.  **Theoretical Distinction**: Formulate models that explicitly preserve the null geodesic in flat Minkowski spacetime, distinguishing this framework from global "Tired Light" theories which are observationally disfavored.
3.  **Dynamical Modeling**: Develop a non-linear Partial Differential Equation (PDE) framework to model photon propagation through specific geometric fields (e.g., the "Depth" field $\Phi$) in strong-gravity regimes, moving beyond static scalar approximations.

## Current Status

**Phase 1: Static Scalar Constraints (Completed)**
The initial phase tested a model where the curvature-work effect scales linearly with the depth of the host galaxy's gravitational potential, gated by the local Kretschmann scalar.
*   **Methodology**: Bayesian MCMC inference.
*   **Result**: The coupling parameter $\alpha$ was found to be statistically consistent with zero ($0.0002 \pm 0.1516$).
*   **Conclusion**: There is no evidence for a macroscopic, static curvature-work effect in galactic environments.

**Phase 2: Dynamical Ray Sheet Analysis (In Progress)**
Research is currently focused on deriving the third-order weakly nonlinear partial differential operator required to model the interaction between the null congruence and the curvature depth field near compact objects (Sgr A*, M87*).

## Authors

*   **Eric Henning** (Theoretical Lead, Southern New Hampshire University)
*   **Aryan Singh** (Computational Lead, The Open University)

## License

This project is released for academic transparency and reproducibility. All rights reserved by the authors regarding the theoretical formulations and original methodologies described herein.
