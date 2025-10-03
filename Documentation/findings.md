# A Geometric Interpretation of Cosmological Redshift: Constraints from a Non-Linear Curvature-Work Model

**Authors:** Aryan Singh¹, Eric Henning²

¹ *Undergraduate in Mathematics & Physics, Open University*
² *Masters Student, Southern New Hampshire University*

*(September 2, 2025)*

> ### Abstract
> The discrepancy between early and late-universe measurements of the Hubble constant, known as the Hubble Tension, represents a fundamental challenge to the standard ΛCDM cosmological model. We introduce a novel theoretical framework, the Curvature-Work hypothesis, which posits that a component of observed redshift originates from the energy expended by photons to traverse regions of high spacetime curvature. This paper develops the first observational test of this hypothesis using a specific, non-linear model based on a Kretschmann scalar threshold. The model proposes that the curvature-work effect is activated via a sigmoid "gate" function only in the strong gravitational fields of massive galaxies. We constrain the strength of this effect, parameterized by α, using a Bayesian MCMC analysis of the Pantheon+ supernova dataset, with priors informed by TDCOSMO strong lensing data. Our analysis yields a best-fit value for the curvature-work parameter of **α = 0.0002 ± 0.1516**. This result is statistically consistent with zero, indicating that we find no evidence for this specific formulation of the Curvature-Work effect in the current data. The model consequently does not resolve the Hubble Tension, recovering a Hubble constant of **H₀ = 72.4 ± 1.0 km/s/Mpc**. We conclude by discussing how this null result provides a valuable constraint on geometric theories of redshift and motivates a more complex, dynamical framework for future work.

---

### **1. Introduction**

The Λ-Cold Dark Matter (ΛCDM) model has been remarkably successful in describing the evolution and large-scale structure of the universe. However, a significant and persistent discrepancy has emerged in the measurement of its current expansion rate, the Hubble constant (H₀). Measurements based on the early universe, specifically the Cosmic Microwave Background (CMB) anisotropies observed by the Planck satellite, yield a value of H₀ = 67.4 ± 0.5 km/s/Mpc [1]. In contrast, measurements from the local universe, based on a distance ladder calibrated with Type Ia supernovae (SNe Ia) from the SH0ES program, find a significantly higher value of H₀ = 73.0 ± 1.0 km/s/Mpc [2]. This ~5σ disagreement, known as the "Hubble Tension," suggests a potential breakdown in the standard cosmological model.

A number of theoretical solutions have been proposed to address this tension. These include models of dynamical dark energy, which suggest the energy density of the vacuum is not a constant but evolves over time [3], or the introduction of new particle physics. Other, more unconventional ideas, revisit the concept of "Tired Light," where photons are hypothesized to lose energy as they travel over cosmic distances [4]. While these models attempt to address the discrepancy, they often require the introduction of new physics beyond the Standard Model and General Relativity (GR), or face challenges in remaining consistent with the full suite of other cosmological observations.

In this work, we introduce and constrain a distinct alternative rooted in the geometric principles of General Relativity. We hypothesize that observed cosmological redshift, `z_obs`, is composed of not only the standard expansion component (`z_exp`) but also a "curvature-work" component (`z_cw`), arising from photons interacting with the spacetime geometry itself. Our foundational principle is that spacetime, as defined by the ability of light to follow a null geodesic (`ds²=0`), is an emergent phenomenon. We propose that in regions of extreme curvature, such as the environment surrounding a massive galaxy, a photon must expend energy to maintain its null path. This energy loss would manifest as an additional redshift dependent on the depth of the gravitational well from which the photon is escaping.

This paper presents the first rigorous observational test of this Curvature-Work hypothesis. We construct a specific, non-linear model where the effect is gated by the local spacetime curvature, quantified by the Kretschmann scalar. We then use premier cosmological datasets—the Pantheon+ SNe Ia compilation and results from the TDCOSMO collaboration—to constrain the strength of this proposed effect. Section 2 outlines the conceptual basis of the hypothesis. Section 3 details the construction of our non-linear observational model. Section 4 describes the data, and Section 5 details our Bayesian analysis and presents the results. Section 6 discusses the implications and the direction for future work.

### **2. The Curvature-Work Hypothesis**

The core of our hypothesis is that the energy of a photon is not solely affected by the global expansion of the metric, but also by local interactions with the curvature of spacetime. We propose a geometric field, Φ(w), embedded in a higher dimension `w`, which governs the permission for null propagation. When a photon traverses a gradient in this field (i.e., "climbs out" of a curvature well), it performs work, leading to an energy loss, or redshift, given by:

`1 + z_obs = (1 + z_exp)(1 + z_cw)`

A key prediction is that `z_cw` is not universal but depends on the environment of the light source. A photon originating from a low-mass galaxy in a shallow potential well would have a negligible `z_cw`, while a photon from a massive galaxy cluster would have a significant `z_cw`. This environment-dependent effect could systematically bias distance measurements, potentially explaining why local H₀ measurements, which are calibrated in various galactic environments, differ from the global value inferred from the CMB.

### **3. A Non-Linear Observational Model**

To make this hypothesis testable, we construct a specific, physically-motivated model. We posit that the curvature-work effect is not a continuous function of mass but is instead a threshold effect, activated only when the spacetime curvature `K` exceeds a critical value. This prevents the model from violating the well-tested success of GR in low-curvature environments.

#### **3.1 Mass Proxies and Curvature Calculation**
We use the Kretschmann scalar, `K = R_abcd R^abcd`, as our coordinate-invariant measure of spacetime curvature. For a static, spherically symmetric object of mass `M` in Schwarzschild coordinates, this is given by:

`K(r) = 48G²M² / c⁴r⁶` (Eq. 1)

The mass `M` is estimated from astrophysical observables: for supernova host galaxies, we use the stellar mass (`M_stellar`), converting it to a total halo mass via a scaling factor (`M_total = 15 × M_stellar`); for strong lensing galaxies, we use the measured stellar velocity dispersion (`σ_v`) and the Faber-Jackson relation (`M ∝ σ_v⁴`).

#### **3.2 Path-Integrated Curvature and the Sigmoid Gate**
The total effect on a photon depends on its path out of the potential well. We define a dimensionless path-integrated curvature, `J`, as:

`J = ∫ (K(l)/K₀)^p dl/l₀` (Eq. 2)

where the integral is taken along the photon's path `l` from a minimum radius `r_min` to a maximum `r_max`. `K₀` and `l₀` are reference scales. For this analysis, we use `p=1`. This path integral `J` serves as the input to a non-linear sigmoid gate function, `w(J)`, which models the smooth activation of the curvature-work effect:

`w(J) = 1 / (1 + exp[-(J - J₀) / σJ])` (Eq. 3)

Here, `J₀` represents the critical curvature threshold, and `σJ` determines the steepness of the transition. This function ensures `w(J)` is near zero for low curvature and approaches one for high curvature.

#### **3.3 The Cosmological Correction**
The final effect is parameterized by a single strength parameter, `α`. The full correction to the apparent distance modulus `μ_apparent` (the observed brightness) is given by:

`μ_apparent = μ_corrected + 5 log₁₀(1 - α × w(J))` (Eq. 4)

where `μ_corrected` is the standard distance modulus in a ΛCDM cosmology. A positive `α` implies that photons from high-curvature environments lose energy, appear fainter, and thus have a larger apparent distance modulus. Our goal is to use observational data to constrain the value of `α`.

### **4. Data and Analysis Methodology**

#### **4.1 Data**
Our primary dataset is the **Pantheon+** compilation of 1701 Type Ia supernovae light curves from 1550 distinct SNe. We utilize the full dataset, applying standard quality cuts to exclude calibrator SNe (`IS_CALIBRATOR == 0`) and ensure reliable host galaxy mass estimates and redshift measurements. For cosmological context and priors, we reference the H₀ measurements from the **TDCOSMO** collaboration, which uses strong gravitational lens time delays.

#### **4.2 Bayesian Analysis**
We employ a Bayesian framework to constrain the parameters of our model. The posterior probability distribution is sampled using `emcee`, an affine-invariant MCMC ensemble sampler. We fit for three free parameters: the Hubble constant (`H₀`), the matter density parameter (`Ωₘ`), and our new curvature-work strength parameter (`α`). We apply wide, uniform priors on all parameters. The likelihood function is a standard chi-squared, which compares the observed distance modulus of each supernova in the Pantheon+ sample to the theoretical distance modulus predicted by Eq. 4. The path integral `J` and gate function `w(J)` for each supernova are pre-computed before the MCMC run to ensure computational efficiency.

### **5. Results**

The MCMC analysis was run for 32 walkers over 1000 steps, with the first 200 steps discarded as burn-in. The fit converged on a well-behaved posterior distribution.

The primary result is the constraint on the curvature-work parameter:
**`α = 0.0002 ± 0.1516`**

This result is statistically consistent with zero at a high level of confidence, with the uncertainty being over 700 times larger than the best-fit value. The data, therefore, shows no evidence for the Kretschmann-gated curvature-work effect as formulated.

As a consequence of the null detection for `α`, the model does not resolve the Hubble Tension. The cosmological parameters recovered from the fit are:
**`H₀ = 72.4 ± 1.0 km/s/Mpc`**
**`Ωₘ = 0.356 ± 0.179`**

The recovered value for H₀ is in excellent agreement with the local measurement from the SH0ES team and remains in significant tension with the early-universe value from Planck. The matter density `Ωₘ` is weakly constrained, as is expected from SNe Ia data alone.

### **6. Discussion and Future Work**

The null result from our non-linear, physically-motivated model provides a powerful constraint on the Curvature-Work hypothesis. It demonstrates that a static, algebraic correction based on the curvature of the host environment is not sufficient to explain the Hubble Tension. The data strongly prefers a model with `α=0`, leaving the discrepancy between early and late-universe H₀ measurements unresolved.

This outcome does not entirely rule out the foundational principle of Curvature-Work, but it significantly narrows the possibilities. The result suggests that if such an effect exists, it is either too subtle to be detected with current data, or it operates through a mechanism not captured by our model.

One possibility is that the interaction is not static but fully dynamical, where the energy-momentum of the photon can feed back into the spacetime geometry itself. Such an interaction would not be described by an algebraic correction but would require a non-linear partial differential equation (PDE). Future theoretical work will focus on the development of such a PDE. The constraints derived in this paper serve as a crucial guidepost, demonstrating that any successful theory must produce a negligible effect under the conditions tested here, and highlighting the need for a more complex formulation.

---
### **References**

[1] Planck Collaboration, N. Aghanim, et al., "Planck 2018 results. VI. Cosmological parameters," *Astronomy & Astrophysics*, 641, A6 (2020). [https://doi.org/10.1051/0004-6361/201833910](https://doi.org/10.1051/0004-6361/201833910)

[2] Riess, A. G., et al., "A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc Uncertainty from the Hubble Space Telescope and the SH0ES Team," *The Astrophysical Journal Letters*, 934, L7 (2022). [https://doi.org/10.3847/2041-8213/ac5c5b](https://doi.org/10.3847/2041-8213/ac5c5b)

[3] Copeland, E. J., Sami, M., & Tsujikawa, S., "Dynamics of dark energy," *International Journal of Modern Physics D*, 15(11), 1753-1935 (2006). [arXiv:hep-th/0603057](https://arxiv.org/abs/hep-th/0603057)

[4] LaViolette, P. A., "Is the universe really expanding?," *The Astrophysical Journal*, 301, 544-553 (1986).