# Theoretical Framework: Curvature-Work Theory

**Version:** 3.1
**Date:** September 2025

## 1. Core Hypothesis

The Curvature-Work Theory proposes a modification to the standard interpretation of cosmological redshift based on the principle of **Emergent Null Propagation**.

### 1.1 Emergent Null Propagation
In standard General Relativity (GR), the condition for light propagation, $ds^2 = 0$, is fundamental and invariant. We propose that this condition is emergent, resulting from the interaction between the photon and a background geometric field $\Phi$.
*   **Weak Field Limit**: In regions of low spacetime curvature (e.g., intergalactic space), the interaction is negligible, and the standard null geodesic is preserved.
*   **Strong Field Limit**: In regions of extreme curvature, the background field $\Phi$ acquires a significant gradient. Photons must expend energy to maintain the null condition against this gradient.

### 1.2 Redshift Decomposition
Consequently, the observed redshift $z_{obs}$ is defined as the composite of kinematic expansion and geometric work:

$$ z_{obs} = (1 + z_{exp})(1 + z_{cw}) - 1 $$

Where $z_{exp}$ is the standard cosmological redshift due to metric expansion, and $z_{cw}$ is the redshift induced by the "curvature work" performed by the photon.

## 2. Mathematical Formulation

### 2.1 The Geometric Field
The field $\Phi(w)$ is postulated to exist within a higher-dimensional embedding space ($w$). The interaction is "gated" by the local spacetime curvature, ensuring compatibility with solar-system tests of GR.

### 2.2 The Gating Mechanism
The curvature work component is active only when the local curvature exceeds a critical threshold. We utilize the Kretschmann scalar ($K$) as the invariant measure of curvature:

$$ K = R_{abcd}R^{abcd} $$

The interaction term is modeled as:

$$ z_{cw} \propto \alpha \cdot \Theta(K - K_{crit}) $$

Where $\Theta$ represents a smooth step function (sigmoid) and $\alpha$ is the coupling constant constrained by observation.

## 3. Observational Constraints

Recent Bayesian analysis using the Pantheon+ supernova dataset has placed strict constraints on this model. The coupling constant $\alpha$ was found to be statistically indistinguishable from zero ($\alpha \approx 0.0002 \pm 0.15$).

**Implication:** This result indicates that the curvature-work effect is not triggered by the macroscopic curvature characteristic of galactic halos ($K \sim 10^{-13} m^{-2}$). If the effect exists, the critical threshold $K_{crit}$ must correspond to significantly higher curvature regimes, likely those found only in the immediate vicinity of compact objects (black holes and neutron stars).

## 4. Future Development: Non-Linear Optics

Driven by the constraints established above, the theoretical focus has shifted to **Non-Linear Geometric Optics**.
We are currently deriving a third-order weakly nonlinear Partial Differential Equation (PDE) to describe the propagation of the null congruence on a "Ray Sheet" embedded in the $\Phi$ field. This framework aims to model specific optical phenomena (such as null congruence instability) near the photon sphere of supermassive black holes (e.g., Sgr A*, M87*).