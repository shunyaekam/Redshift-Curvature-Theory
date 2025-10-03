# Critical Validation Checklist for Curvature-Work Detection

## 🚨 MUST DO BEFORE CLAIMING DETECTION

### 1. Independent Data Split Test
- [ ] Split Pantheon+ data into train/test (50/50)
- [ ] Tune parameters on training set only
- [ ] Test detection on held-out test set
- [ ] Result: detection persists? Y/N

### 2. Parameter Robustness Test  
- [ ] Vary K₀ by factors of 10: [1e-11, 1e-10, 1e-9]
- [ ] Vary integration bounds: [0.1-10km, 1-100km, 10-1000km]
- [ ] Check if α detection depends critically on these choices
- [ ] Result: robust across parameter space? Y/N

### 3. Host Mass Correlation Check
- [ ] Plot α correction vs host mass directly
- [ ] Check if effect is just "massive hosts = brighter SNe"
- [ ] Test with stellar mass vs total mass scaling
- [ ] Result: physically reasonable correlation? Y/N

### 4. Null Test on Randomized Data
- [ ] Randomize host masses while keeping other properties
- [ ] Run same analysis on scrambled data
- [ ] Should find α ≈ 0 if real physics
- [ ] Result: null detection on scrambled data? Y/N

### 5. Literature Cross-Check
- [ ] Search for known "host galaxy mass effects" in SN literature
- [ ] Check if this reproduces known systematics
- [ ] Verify this isn't rediscovering Malmquist bias
- [ ] Result: genuinely new effect? Y/N

### 6. Error Analysis
- [ ] Bootstrap resample data 1000 times
- [ ] Check distribution of α values
- [ ] Verify error bars are realistic
- [ ] Result: error estimate reliable? Y/N

## 🎯 SUCCESS CRITERIA

**CLAIM DETECTION ONLY IF:**
- ✅ Passes independent data test
- ✅ Robust across parameter choices  
- ✅ Shows physical correlation pattern
- ✅ Null test gives α ≈ 0
- ✅ Not explained by known systematics
- ✅ Error analysis confirms significance

**TIMELINE:** Complete in 2 weeks before any publication claims.

**IF ANY TEST FAILS:** Downgrade to "interesting systematic investigation" not "breakthrough detection"