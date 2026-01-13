# Diagnostic Analysis Pipeline

**Date:** September 2025
**Subject:** Computational Methodology for Curvature-Work Constraints

## 1. Scientific Context

The diagnostic pipeline (`curvature_work_diagnostic.py`) was developed to empirically test the "Curvature-Work" hypothesis. This hypothesis posits that the observed redshift of a photon is a function of both the cosmological scale factor and the gravitational potential depth of the emission source. If valid, objects in deeper gravitational wells (such as massive elliptical galaxies) would exhibit an excess redshift relative to objects in shallower wells, potentially biasing measurements of the Hubble constant ($H_0$). 

## 2. Methodology

The analysis employs a Bayesian Markov Chain Monte Carlo (MCMC) approach to fit a modified cosmological model to observational data.

### 2.1 Data Ingestion and Proxies
The pipeline integrates two primary datasets, utilizing astrophysical proxies to estimate the gravitational curvature depth:
*   **Strong Lensing (TDCOSMO)**: The stellar velocity dispersion ($\sigma_v$) is used as a proxy for the depth of the gravitational potential well.
*   **Type Ia Supernovae (Pantheon+)**: The stellar mass of the host galaxy ($M_{stellar}$) serves as the curvature proxy.

Both proxies are normalized to a dimensionless "Environment Depth" scale ranging from 0 to 1 to facilitate cross-comparison.

### 2.2 Theoretical Model
The distance modulus $\mu$ is modeled as the sum of the standard $\Lambda$CDM prediction and a curvature-dependent correction:

$$ \mu_{theory}(z, \text{Depth}) = \mu_{\Lambda CDM}(z, H_0, \Omega_m) + 5 \log_{10}(1 - \alpha \cdot f(\text{Depth})) $$

Where:
*   $\alpha$ represents the strength of the curvature-work coupling.
*   $f(\text{Depth})$ is a sigmoid gating function designed to activate only in high-curvature regions, thereby satisfying General Relativity constraints in the weak-field limit.

### 2.3 Statistical Inference
Parameter estimation is performed using the `emcee` ensemble sampler. The likelihood function compares the observed distance moduli from the Pantheon+ sample against the theoretical predictions. The free parameters in the fit are the Hubble constant ($H_0$), the matter density ($\Omega_m$), and the curvature-work parameter ($\alpha$). 

## 3. Interpretation of Results

The primary output of the diagnostic is the posterior probability distribution for $\alpha$.

*   **Detection**: A statistically significant deviation of $\alpha$ from zero ($>3\sigma$) would indicate evidence for a curvature-dependent redshift component.
*   **Constraint**: A result consistent with zero implies that the proposed effect is negligible within the sensitivity of current data.

The current analysis yields $\alpha ≈ 0$, which serves as a strong constraint on the theory. It implies that any geometric redshift effect must be restricted to regimes not probed by galactic-scale metrics, motivating the subsequent shift to micro-physical modeling near event horizons.