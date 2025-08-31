# Upgrading Your **MoE Super-Weight** Detector

*A migration guide from your current two-phase method to the routing-aware approach*

> This document **starts from your current pipeline** (Phase 1 router analysis → Phase 2 per‑expert detection with iterative suppression) and shows **exactly what to change** and **what to add** to reach the improved method. All math is in LaTeX. Code snippets are drop‑in patterns, not full implementations.

---

## 0) Your current baseline (as you described)

**Phase 1 — Router Pattern Analysis**

1. Generate diverse input samples.
2. Register forward hooks on router/gating modules and log expert selections.
3. Count how often each expert is selected.
4. **Filter** to “active” experts: experts used in **> 50%** of samples.

**Phase 2 — Expert-Focused Detection**

1. On the filtered experts, run **dense-style super weight detection** per expert (co-spike / spike pairing in the expert’s MLP **down-proj**).
2. Create `MoESuperWeight(layer, expert_id, component, row, col)` objects.
3. **Iterative suppression:** zero detected scalars and repeat to find more.

The upgrade keeps your structure but replaces the hard filter and adds **routing-aware scoring**, **fast causal proxies**, and **interventional checks**.

---

## 1) High-level migration plan (what changes)

| Area                     | Keep                          | Replace / Add                                                                                                                              |
| ------------------------ | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Expert selection         | Two-phase flow                | Replace hard `> 50%` filter with **weighted** selection using $p_{\text{active}}$ and **position entropy**                                 |
| Detection inside experts | Your co-spike / spike pairing | Keep method, but **condition on routed tokens only** and add **micro‑ablation** proxies                                                    |
| Scoring & validation     | Perplexity only (expensive)   | Add **fast proxies** (super-activation energy, stopword skew) + **natural vs interventional** routing impact                               |
| Data model               | `MoESuperWeight` tuple        | Extend with $p_{\text{active}}$, entropy notes, capacity flags, and two causal scores $\mathcal{I}_{\text{nat}}, \mathcal{I}_{\text{int}}$ |

---

## 2) Routing statistics (drop‑in replacement for your Phase 1 filter)

You already log top‑$K$ expert indices per token and layer. Aggregate three statistics:

### 2.1 Expert usage probability

For layer $\ell$ and expert $e$,

$$
p_{\text{active}}^{(\ell)}(e)
=\mathbb{E}_t\!\left[\mathbf{1}\{e\in \text{TopK}(p^{(\ell)}_t)\}\right].
$$

### 2.2 Position‑wise routing entropy

At absolute token position `pos` inside a sequence,

$$
H^{(\ell)}(\text{pos})=-\sum_{e=1}^{E}\hat{p}^{(\ell)}_{\text{pos}}(e)\,\log \hat{p}^{(\ell)}_{\text{pos}}(e),
$$

where $\hat{p}^{(\ell)}_{\text{pos}}(e)$ is the empirical frequency that expert $e$ is selected at that position.

> Low $H^{(\ell)}(\text{pos})$ = **stable routing**, great for detection.

### 2.3 Capacity/overflow tagging

Track a per‑layer overflow rate $\rho_\text{overflow}^{(\ell)}$ (fraction of tokens that were rerouted/dropped due to capacity). You’ll use it to **exclude** noisy batches during validation.

### 2.4 New expert selection rule (replace your `>50%` filter)

```diff
- active = {e for e in experts if usage[e] > 0.5}
+ p_floor = 0.01  # 1%
+ low_entropy_pos = {pos for pos in positions if H[l][pos] <= H_threshold}
+ active = {
+   e for e in experts
+   if p_active[l][e] >= p_floor
+   or routes_through_any_low_entropy_pos(l, e, low_entropy_pos)
+}
```

Rationale: **rare-but-critical** experts should stay; we’ll **weight** their impact by $p_{\text{active}}$ instead of discarding them.

---

## 3) Per‑expert detection (builds on your Phase 2)

You already do dense-style spike detection per expert. Maintain that method, but constrain data to **routed tokens for that expert** and add a numeric co‑spike score.

### 3.1 Co‑spike score (explicit)

For routed batch $t=1..T$, expert input $X^{(\ell,e)}\in\mathbb{R}^{T\times d_{\text{ff}}}$, output $Y^{(\ell,e)}\in\mathbb{R}^{T\times d_{\text{model}}}$:

$$
\mathcal{S}^{(\ell,e)}(r,c)
=\frac{\sum_{t=1}^{T}\big|X^{(\ell,e)}_{t,r}\,Y^{(\ell,e)}_{t,c}\big|}
{\sqrt{\sum_{t=1}^{T}\big(X^{(\ell,e)}_{t,r}\big)^2}\;\sqrt{\sum_{t=1}^{T}\big(Y^{(\ell,e)}_{t,c}\big)^2}+\epsilon}.
$$

Take $(r^*,c^*)=\arg\max_{r,c}\mathcal{S}^{(\ell,e)}(r,c)$ and map to the single scalar

$$
w^* \;=\; W_{\text{down}}^{(\ell,e)}[c^*,\,r^*].
$$

> **Drop‑in:** wherever you currently select a (row, col) by heuristic spikes, replace with $\arg\max \mathcal{S}^{(\ell,e)}$ (or use it to re‑rank your candidates).

### 3.2 Iterative suppression stays the same, but add **micro‑ablation proxies**

After proposing $w^*$, temporarily set $W_{\text{down}}^{(\ell,e)}[c^*,r^*]\leftarrow 0$ and measure two cheap proxies **before** running full perplexity:

**Proxy A — Recurrent super‑activation energy**
Let $H^{(\ell+1)}$ be the post‑block hidden; track the energy in channel $c^*$:

$$
E_{c^*}=\frac{1}{T}\sum_{t=1}^{T}\big(H^{(\ell+1)}_{t,c^*}\big)^2.
$$

Zeroing $w^*$ should **decrease** $E_{c^*}$ on tokens that visited $e$.

**Proxy B — Stopword skew**
For a small stopword set $S$, compare the total probability mass:

$$
\Delta_{\text{stop}}=\mathbb{E}\!\Big[\sum_{s\in S}p_\theta(s\mid \text{context})\Big]_{\text{ablated}}
-\mathbb{E}\!\Big[\sum_{s\in S}p_\theta(s\mid \text{context})\Big]_{\text{baseline}}.
$$

In dense models, removal often **increases** stopword mass; check conditionally on routed tokens.

Use thresholds on these proxies to decide if a candidate merits full evaluation.

---

## 4) Routing‑aware importance (add to your scoring)

Your current scoring likely treats all batches equally. Make it routing‑aware:

$$
\boxed{
\mathcal{I}(w^*,e,\ell)
=\mathbb{E}_{\text{prompts}}
\!\left[
 p_{\text{active}}^{(\ell)}(e\mid \text{prompt}) \cdot \Delta \text{Metric}\!\left(\text{zero}(w^*)\right)
\right]}
$$

Two concrete estimators:

1. **Natural routing**

$$
\mathcal{I}_{\text{nat}}(w^*,e,\ell)=\mathbb{E}\!\left[p_{\text{active}}^{(\ell)}(e)\cdot \Delta\text{Metric}\right].
$$

2. **Interventional routing** (new; add this)
   Bias or force routing to expert $e$ on a small diagnostic slice:

$$
g^{(\ell)}_t(e)\leftarrow g^{(\ell)}_t(e)+\beta \quad(\text{pre-top-}K),
$$

or select $e$ directly (respect capacity). Then measure

$$
\mathcal{I}_{\text{int}}(w^*,e,\ell)=\mathbb{E}\!\left[\Delta\text{Metric}\right]_{\text{forced to }e}.
$$

**Call it a true MoE super weight if** $\mathcal{I}_{\text{nat}}$ and $\mathcal{I}_{\text{int}}$ agree on direction and exceed a small effect threshold.

---

## 5) Exact code‑level changes (patch patterns)

### 5.1 Replace expert filter

```diff
- active = {e for e in experts if expert_usage[e] > 0.5}
+ p_floor = cfg.p_active_floor  # default: 0.01
+ H_thr   = cfg.routing_entropy_thr  # e.g., 0.7 * median(H)
+ low_entropy_pos = positions_with_entropy_below(H, H_thr)
+ active = {
+   e for e in experts
+   if p_active[l][e] >= p_floor
+   or traverses_low_entropy_positions(routes[l], e, low_entropy_pos)
+}
```

### 5.2 Add routed‑only tensors in expert hooks

```diff
- X_all, Y_all = collect_expert_tensors(layer=l, expert=e, all_tokens=True)
+ X_routed, Y_routed = collect_expert_tensors(layer=l, expert=e, only_routed=True)
```

### 5.3 Swap spike heuristic for co‑spike score

```diff
- r_star, c_star = pick_spike_pair_heuristic(X_routed, Y_routed)
+ r_star, c_star, score = argmax_co_spike(X_routed, Y_routed, eps=1e-8)
+ if score < cfg.co_spike_tau: continue
```

### 5.4 Insert micro‑ablation proxies before perplexity

```diff
+ with single_scalar_zeroed(model, l, e, r_star, c_star):
+     delta_energy = measure_channel_energy_drop(model, l+1, c_star, routed_mask=(l,e))
+     delta_stop   = measure_stopword_skew(model, routed_mask=(l,e))
+ if not passes_proxy_thresholds(delta_energy, delta_stop): continue
```

### 5.5 Routing‑aware importance (natural + interventional)

```diff
- delta_ppl = eval_perplexity(model, dataset)
- record(w=(l,e,r_star,c_star), delta_ppl=delta_ppl)
+ I_nat = eval_weighted_metric(model, w=(l,e,r_star,c_star),
+                              mode="natural", weights=p_active[l])
+ I_int = eval_metric_with_routing_intervention(model, w=(l,e,r_star,c_star),
+                                               layer=l, expert=e, beta=cfg.router_bias)
+ record(w=(l,e,r_star,c_star), I_nat=I_nat, I_int=I_int,
+        p_active=p_active[l][e], proxies={"energy": delta_energy, "stop": delta_stop})
```

### 5.6 Iterative suppression stays — just keep the proxies in‑loop

```python
while True:
    r_star, c_star, score = argmax_co_spike(...)
    if score < tau: break
    if not micro_ablate_and_pass(model, l, e, r_star, c_star): break
    zero_single_scalar_(model, l, e, r_star, c_star)  # continue discovering
```

---

## 6) Data model (extend your `MoESuperWeight`)

Add routing stats and causal scores to your existing object or JSON schema:

```json
{
  "layer": L,
  "expert": E,
  "row_ff": r_star,
  "col_model": c_star,
  "score_co_spike": S_value,
  "p_active": p_active[L][E],
  "low_entropy_positions": [0, 1, 2],
  "capacity_overflow_rate": 0.03,
  "proxy": {"energy": -0.24, "stop": +0.07},
  "I_nat": -0.15,
  "I_int": -0.18
}
```

---

## 7) Thresholds & defaults (start here, then tune)

* $p_{\text{active}}$ floor: $0.5\%$–$2\%$, default $1\%$.
* Co‑spike detection $\tau$: 95th percentile of random $(r,c)$ scores or a permutation‑based cut.
* Router bias $\beta$ for intervention: $+1.5$ to $+3.0$ (logit units) on a **small** diagnostic slice.
* Proxy gates: require energy drop $E_{c^*}\downarrow$ by $\geq$ a small fraction (e.g., $5\%$) and a consistent stopword skew.

---

## 8) Validation checklist (migration‑aware)

* $\sum_e p_{\text{active}}^{(\ell)}(e)\approx K$ for top‑$K$ routing.
* Low‑entropy positions stable across prompt classes.
* Per‑expert: very few dominant $(r,c)$ pairs; iterative suppression reduces $E_{c^*}$ monotonically.
* $\mathcal{I}_{\text{nat}}$ and $\mathcal{I}_{\text{int}}$ agree on sign and are non‑trivial.
* Full perplexity confirms the proxies for top candidates.

---

## 9) Minimal CLI delta (so your scripts evolve, not restart)

```
moe_sw_detect \
  --p_active_floor 0.01 \
  --routing_entropy_thr auto \
  --co_spike_tau 0.12 \
  --router_bias 2.0 \
  --proxies energy,stopwords \
  --score_modes natural,interventional
```

---

## 10) Quick FAQ for the migration

**Q: What if an expert is extremely rare?**
Keep it if it consistently appears at low‑entropy positions. Otherwise, analyze with **lower sampling** and rely on $p_{\text{active}}$-weighted impact.

**Q: Can I skip the interventional route?**
You can, but $\mathcal{I}_{\text{int}}$ catches false positives that arise from routing noise or capacity artifacts.

**Q: Where should I look first?**
Earlier layers and earlier token positions (low entropy) tend to be the most stable for detection.

---

## 11) Summary

* **Replace** your `>50%` expert filter with $p_{\text{active}}$ + routing‑entropy informed selection.
* **Keep** your expert‑local co‑spike detection, but compute it on **routed tokens only** and require **proxy improvements** under micro‑ablation.
* **Add** routing‑aware importance $\mathcal{I}_{\text{nat}}$ and **interventional** $\mathcal{I}_{\text{int}}$ to validate true MoE super weights.
* Extend your catalog with routing stats so you can restore/ablate precisely where it matters.

