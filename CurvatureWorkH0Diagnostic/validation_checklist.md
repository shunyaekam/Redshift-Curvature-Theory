# Validation Protocol

**Subject:** Criteria for Detection of Geometric Redshift Effects

## 1. Introduction

This document outlines the standardized validation protocol required to claim a detection of geometric redshift effects. Given the potential for systematic errors in cosmological measurements, rigorous adherence to this protocol is mandatory before any results are publicized.

## 2. Pre-Detection Requirements

Any potential signal must satisfy the following conditions:

### 2.1 Independent Data Validation
*   **Split-Sample Test:** The dataset must be partitioned into independent training and testing sets (e.g., 50/50 split).
*   **Replicability:** Parameters derived from the training set must yield a statistically significant detection when applied to the testing set.

### 2.2 Parameter Robustness
*   **Threshold Stability:** The detected signal must persist across a reasonable range of critical curvature thresholds ($K_0$).
*   **Path Independence:** The result must be robust against variations in the integration bounds used to calculate the curvature depth.

### 2.3 Physical Correlation
*   **Scaling Relation:** The strength of the effect ($\alpha$) must exhibit a physically motivated correlation with host galaxy properties (e.g., Stellar Mass or Velocity Dispersion).
*   **Null Hypothesis Test:** Randomizing the host galaxy properties (shuffling the mass proxies) must result in a non-detection ($\alpha \approx 0$).

### 2.4 Systematic Control
*   **Literature Review:** The signal must be distinct from known astrophysical systematics such as Malmquist bias, dust extinction, or correlation with local star-formation rates.
*   **Statistical Significance:** Significance must be established via bootstrap resampling ($N \ge 1000$) to generate reliable confidence intervals.

## 3. Classification of Results

Results are to be classified into one of two categories:

**A. Constraint (Current Status)**
*   **Criteria:** $\alpha$ is consistent with zero within $3\sigma$, or fails any of the robustness tests.
*   **Action:** Report as an upper limit on the theory.

**B. Detection**
*   **Criteria:** $\alpha$ deviates from zero by $>3\sigma$ and passes all validation tests described in Section 2.
*   **Action:** Proceed to peer review preparation.