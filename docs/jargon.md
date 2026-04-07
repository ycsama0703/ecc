# ECC Project Terminology Sheet

## 1. Core framing

### Noisy timing
**Preferred use:** noisy timing  
**Meaning:** timing information is imprecise or incomplete, so ECC-derived signals cannot be treated as perfectly time-aligned.  
**Do not replace with:** bad alignment, timestamp error, temporal noise  
**Notes:** use this as the broad empirical condition; do not use it as a synonym for reliability itself.

### Reliability
**Preferred use:** latent reliability  
**Symbol:** `s_i`  
**Meaning:** the unobserved reliability of the ECC-derived correction signal for event `i`.  
**Notes:** this is the main hidden quantity in the paper. Use “reliability” in formal sections; avoid loosely switching to “signal quality” unless in informal explanation.

### Observability / proxy variables
**Preferred use:** data-quality and observability indicators  
**Symbol:** `a_i`  
**Meaning:** observable indicators that provide noisy information about latent reliability, such as alignment quality, coverage, transcript completeness, and timing metadata quality.  
**Do not call them:** reliability itself, ground truth, exact alignment labels  
**Notes:** in methodology, refer to them as **noisy proxies** of latent reliability.

### Proxy uncertainty
**Preferred use:** proxy uncertainty  
**Meaning:** uncertainty induced by the fact that `a_i` does not uniquely identify `s_i`.  
**Notes:** use this term when motivating the minimax gate and abstention.

---

## 2. Prediction structure

### Target
**Preferred use:** `shock_minus_pre`  
**Symbol:** `y_i`  
**Definition:** `RV^{post}_{i,60m} - RV^{pre}_{i,60m}`  
**Meaning:** the incremental post-call volatility shock relative to the pre-call baseline.  
**Notes:** in natural language, call it a **residual-style target** or **incremental post-call shock**.

### Market prior
**Preferred use:** market prior  
**Symbol:** `\mu_i`  
**Definition:** `\mu_i = f_\phi(x_i)`  
**Meaning:** the component explained by pre-call market information and structured controls.  
**Allowed secondary phrase:** pre-call market model  
**Avoid overusing:** market baseline, baseline prediction, prior-only model in formal definitions  
**Notes:** in experiments, “baseline” is fine when comparing models; in formulation/method, prefer **market prior**.

### Residual component
**Preferred use:** residual component  
**Symbol:** `r_i`  
**Meaning:** the component of the target not explained by the market prior and potentially informed by the call.

### ECC correction signal
**Preferred use:** ECC correction signal  
**Symbol:** `z_i`  
**Definition:** `z_i = g_\theta(e_i)`  
**Meaning:** the call-derived correction term used on top of the market prior.  
**Do not call it:** ECC feature, ECC output, local signal, multimodal score  
**Notes:** in formal sections, keep this name fixed.

### Correction weight / gate
**Preferred use:** correction weight  
**Symbol:** `\alpha_i`  
**Meaning:** the amount of ECC correction admitted into the final prediction.  
**When robust version is needed:** proxy-robust gate, minimax gate  
**Symbol for robust version:** `\alpha_i^{\mathrm{mm}}`

### Final prediction
**Preferred use:** final prediction  
**Symbol:** `\hat y_i`  
**Definition:** `\hat y_i = \hat\mu_i + \alpha_i z_i`  
**Notes:** if abstention is active, use the indicator form with `A_i`.

---

## 3. Method name and method objects

### Method name
**Preferred use:** Proxy-Robust ECC Correction (PREC)  
**Short form after first mention:** PREC  
**Do not vary with:** PREC framework, proposed PREC approach, our PREC model  
**Notes:** after first definition, just use **PREC**.

### Oracle gate
**Preferred use:** reliability-adaptive gate  
**Symbol:** `\alpha_i^*(s_i)`  
**Meaning:** the conditional MSE-optimal gate when reliability is observed.

### Minimax gate
**Preferred use:** minimax gate  
**Symbol:** `\alpha_i^{\mathrm{mm}}`  
**Meaning:** the proxy-robust gate obtained under partial identification of reliability.

### Identification set
**Preferred use:** identification set  
**Symbol:** `S_i`  
**Meaning:** the set of reliability values consistent with observed proxies.

### Worst-case noise level
**Preferred use:** worst-case noise level  
**Symbol:** `\bar{\sigma}_i^2`  
**Meaning:** the supremum of `\sigma^2(s)` over the identification set `S_i`.

### Abstention
**Preferred use:** risk-controlled abstention  
**Symbol:** `A_i`  
**Meaning:** the decision to suppress the ECC correction when worst-case conditional risk exceeds a threshold.  
**Do not call it:** fallback trick, ignore rule, filter rule

---

## 4. Empirical scope

### Main empirical setting
**Preferred use:** clean after-hours setting  
**Notes:** in natural language, always use **after-hours** with a hyphen.  
**Only use** `after_hours` **inside code-style formatting**, table entries, or variable names.

### Main claim
**Preferred use:** ECC-derived information adds value beyond strong pre-call market predictors  
**Notes:** use this as the main empirical claim, not “audio helps” or “multimodal fusion helps”.

### A4
**Preferred use:** alignment metadata files (A4) in data descriptions; A4 alignment/observability signals in empirical discussion  
**Do not introduce without definition.**  
**Notes:** in intro and methodology, avoid saying “A4-like” unless already defined; prefer “data-quality and observability indicators”.

---

## 5. Style rules

1. In formal sections, prefer **reliability** over **signal quality**.
2. In formal sections, prefer **market prior** over **market baseline**.
3. Use **ECC correction signal** consistently for `z_i`.
4. Use **data-quality and observability indicators** consistently for `a_i`.
5. Use **after-hours** in prose, `after_hours` only in code-like text.
6. Use **PREC** consistently after first mention.
7. Do not call proxies “ground truth”.
8. Do not describe the method as generic multimodal fusion; describe it as a **prior-and-correction rule under noisy observability**.