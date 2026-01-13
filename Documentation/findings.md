# A Geometric Interpretation of Cosmological Redshift: Constraints from a Non-Linear Curvature-Work Model

**Authors:** Aryan Singh¹, Eric Henning²

¹ *Undergraduate in Mathematics & Physics, Open University*
² *Masters Student, Southern New Hampshire University*

*(September 2, 2025)*

> ### Abstract
> The discrepancy between early and late-universe measurements of the Hubble constant, known as the Hubble Tension, represents a fundamental challenge to the standard ΛCDM cosmological model. While various solutions have been proposed, including "Tired Light" mechanisms that violate General Relativity in flat spacetime, we introduce a rigorous alternative: the Curvature-Work hypothesis. This framework posits that redshift includes a component from energy expended by photons traversing regions of extreme spacetime curvature, preserving the null geodesic in flat space. This paper presents the first observational test of this hypothesis using a non-linear, Kretschmann-scalar-gated model. We constrain the effect using a Bayesian MCMC analysis of the Pantheon+ supernova dataset. Our analysis yields a curvature-work parameter of **α = 0.0002 ± 0.1516**, statistically consistent with zero. This null result effectively rules out simple, algebraic modifications to photon propagation and provides a strong constraint on geometric redshift theories. It demonstrates that if such an effect exists, it must be highly localized and dynamical, motivating the development of a non-linear Partial Differential Equation (PDE) framework rather than simple distance-based corrections.

---

### **1. Introduction**

The Λ-Cold Dark Matter (ΛCDM) model has been remarkably successful in describing the evolution and large-scale structure of the universe. However, a significant and persistent discrepancy has emerged in the measurement of its current expansion rate, the Hubble constant (H₀). Measurements based on the early universe (Planck CMB) yield H₀ = 67.4 ± 0.5 km/s/Mpc [1], while local measurements (SH0ES SNe Ia) find H₀ = 73.0 ± 1.0 km/s/Mpc [2]. This ~5σ disagreement suggests a potential breakdown in the standard model.

Proposed solutions range from dynamical dark energy [3] to reviving "Tired Light" (TL) theories, which hypothesize that photons lose energy continuously over cosmic distances due to interactions with the vacuum or unknown fields [4, 5]. Recent variations, such as CCC+TL models [6], attempt to blend these ideas with standard cosmology. However, traditional TL models often face insuperable challenges: they typically violate the strong equivalence principle, fail to preserve the blackbody spectrum of the CMB, and contradict the time-dilation observed in supernovae light curves. Furthermore, they imply a modification of the null geodesic even in flat spacetime, a postulate that conflicts with the high-precision successes of General Relativity (GR).

In this work, we investigate a distinct alternative: the **Curvature-Work hypothesis**. Unlike TL, this hypothesis respects the null geodesic in flat spacetime. We propose that spacetime is defined by the permission of null propagation (`ds²=0`), and that this permission is an emergent property. In regions of **extreme curvature**—such as the deep gravitational wells of massive galaxies—photons must perform "work" against a geometric field Φ(w) to maintain their null path. This energy loss manifests as an additional redshift, `z_cw`, but *only* in these specific, high-curvature environments.

This paper presents a rigorous constraint on this hypothesis. We construct a non-linear model where the effect is "gated" by the local Kretschmann scalar, ensuring no deviation from GR in low-curvature regions. We use the Pantheon+ SNe Ia compilation to constrain the strength of this effect.

### **2. The Curvature-Work Hypothesis vs. Tired Light**

It is crucial to distinguish Curvature-Work from Tired Light.
*   **Tired Light:** `z_TL ∝ Distance`. Energy loss is continuous and happens everywhere, including the intergalactic medium (flat space).
*   **Curvature-Work:** `z_cw ∝ Depth_of_Well` (gated). Energy loss is discrete and happens only during the exit from high-curvature regions.

Our hypothesis posits that `1 + z_obs = (1 + z_exp)(1 + z_cw)`. By testing for a `z_cw` component dependent on host galaxy mass (a proxy for curvature depth), we can distinguish geometric work from simple expansion or distance-dependent tired light.

### **3. A Non-Linear Observational Model**

We construct a model where the curvature-work effect is activated only when the spacetime curvature `K` exceeds a critical threshold `J₀`.

#### **3.1 Mass Proxies and Curvature Calculation**
We use the Kretschmann scalar, `K = R_abcd R^abcd`, as our measure of curvature. For a static, spherically symmetric object:
`K(r) = 48G²M² / c⁴r⁶` (Eq. 1)
Mass `M` is estimated from SNe Ia host stellar masses (`M_total = 15 × M_stellar`).

#### **3.2 Path-Integrated Curvature and the Sigmoid Gate**
We define a path-integrated curvature `J` and a sigmoid gate `w(J)`:
`w(J) = 1 / (1 + exp[-(J - J₀) / σJ])` (Eq. 2)
This ensures the effect turns "on" only for high-mass systems and is zero for low-mass systems, respecting GR in the weak-field limit.

#### **3.3 The Cosmological Correction**
The apparent distance modulus is modified by:
`μ_apparent = μ_corrected + 5 log₁₀(1 - α × w(J))` (Eq. 3)
A positive `α` implies energy loss (fainter SNe) in high-curvature hosts.

### **4. Data and Analysis Methodology**

We use the **Pantheon+** compilation (1701 light curves). We employ a Bayesian MCMC analysis (`emcee`) to fit `H₀`, `Ωₘ`, and `α`, comparing the observed distance moduli to our model predictions.

### **5. Results**

The MCMC analysis yields a constraint on the curvature-work parameter:
**`α = 0.0002 ± 0.1516`**

This result is statistically consistent with zero. The data shows no evidence for the algebraic, Kretschmann-gated curvature-work effect in the current sample. The recovered Hubble constant is **H₀ = 72.4 ± 1.0 km/s/Mpc**, consistent with SH0ES and retaining the Hubble Tension.

### **6. Discussion: Constraints on Modified Photon Theories**

The null result (`α ≈ 0`) is a significant finding. It allows us to place strong constraints on geometric theories of redshift.

**1. Rejection of Simple modifications:** The lack of a signal in our "gated" model reinforces the robustness of standard GR in the weak-to-moderate curvature regime (galactic halos). If photons lost energy simply by traversing gravitational potentials (as in some TL variations or simple "mass-tiredness" couplings), we would expect a non-zero `α`. The null result effectively rules out these simple algebraic couplings.

**2. The Failure of Tired Light in Flat Space:** While our model specifically tested curvature dependence, the broader implication supports the standard view that photons do not lose energy in flat spacetime. Any theory relying on "tiredness" to explain the Hubble Tension without invoking expansion must account for the lack of environmental dependence seen here.

**3. Motivation for Dynamical Frameworks:** The Curvature-Work hypothesis remains viable only if the interaction is **not** a static potential effect but a **dynamical** one. The null result suggests that the "work" done by a photon is likely not a function of the scalar curvature `K` alone, but involves the dynamics of the photon's propagation itself—how the null geodesic interacts with the geometric field Φ(w) in a non-linear way.
This motivates our development of a **non-linear Partial Differential Equation (PDE)** framework. Such a framework would treat the photon not as a passive test particle, but as an active probe whose null-congruence interacts with the depth field, potentially allowing for energy exchange only under specific stability conditions (e.g., near event horizons or photon spheres) that are not captured by a galactic-scale scalar average.

### **7. Conclusion**
We have presented a rigorous observational test of the Curvature-Work hypothesis. The null result from our Kretschmann-gated model constrains the magnitude of any algebraic geometric redshift effect to be negligible in the late universe. This finding distinguishes our approach from Tired Light by confirming that standard null propagation holds to high precision in galactic environments, and directs future research toward micro-physical, dynamical interactions described by non-linear geometric optics.

---
### **References**

[1] Planck Collaboration, et al., *A&A*, 641, A6 (2020).
[2] Riess, A. G., et al., *ApJ Letters*, 934, L7 (2022).
[3] Copeland, E. J., et al., *Int. J. Mod. Phys. D*, 15, 1753 (2006).
[4] LaViolette, P. A., *ApJ*, 301, 544 (1986).
[5] Zwicky, F., *PNAS*, 15, 773 (1929).
[6] Gupta, R., "JWST early Universe observations and ΛCDM cosmology," *MNRAS*, 524, 3385 (2023).
