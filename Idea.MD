## Core idea (what it is, in one line)

A **lookback‑based, three‑state Markov chain** that detects the current regime (Bull / Bear / Neutral) in real time, **counts recent state transitions**, converts those counts into **transition probabilities** (with Laplace smoothing), and **visualizes**everything as an animated 3D triangle of spheres, arrows, and moving particles—while also outputting a clean **bias value (+1/0/−1)** you can wire into strategies. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)

---

## What the article emphasizes (interpretation & usage)

- **What you’re seeing:** Each sphere is a regime; the **loop** on a sphere is the “stay in this regime” probability; **arrows**show the most common next regime if a switch occurs. Importantly, **direction is not symmetric** (Bull→Neutral can differ from Neutral→Bull). [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)
- **How regimes are detected:** Either price‑based (adaptive **Z‑score of returns**) or **three custom indicators** you pick; values are normalized, classified into Bull/Bear/Neutral, and (if using three) averaged for robustness. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)
- **How probabilities are computed:** On **confirmed bars only**, it tracks the recent state sequence in a rolling window, **builds a transition matrix**, and extracts probabilities with **Laplace smoothing**; everything updates live. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)
- **Why Markov (not HMM):** States are **observable** and counted directly (simple and interpretable); HMMs require learning hidden states. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)
- **No repainting:** Updates happen **only when bars close** to avoid look‑ahead bias. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)
- **Programmatic output:** A bias plot **(+1 Bull, 0 Neutral, −1 Bear)** is exposed so you can `input.source()` it in other scripts/strategies. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)

---

## What the source code actually does (key mechanics)

### Regime engine

- **Price mode:** Uses **% return** `ta.change(close)/close[1]`, computes **Z‑score** over `lookback_period`, and classifies with **±1/3σ thresholds**: > +1/3 → Bull, < −1/3 → Bear, else Neutral.
- **Custom‑indicator mode:** Accepts **three sources**, each **Z‑scored and clamped to ±3σ**; each becomes +1/0/−1 via **±1σ** thresholds, then **equal‑weighted** and passed through **hysteresis** (±0.05) to reduce flip‑flops.
- **No repaint:** The variable `market_state` is **held on unconfirmed bars** (`barstate.isconfirmed ? current : previous`).

### Transition matrix (Markov chain)

- Tracks a **rolling window** of the last `lookback_period+1` confirmed states and **rebuilds** a 3×3 **count matrix** each bar.
- **Laplace smoothing** in probability extraction: `(count+1)/(row_sum+3)` equivalent per row.
- Also keeps **state occupancy** to show empirical Bull/Bear/Neutral frequencies over the window.

### 3D visualization

- **Wireframe spheres** (grid lines) for Bull/Bear/Neutral using spherical tessellation (`SEGMENTS = 12`) with animated **tilt & rotation**.
- **Active state pulse:** The current regime sphere gets a **pulsing inner sphere**; Neutral gets a subtle **dynamic lighting**effect when inactive.
- **Bidirectional arrows** between every pair of nodes: offset paths to prevent overlap, **arrowheads on the last segment**, and **midpoint probability labels** (both directions).
- **Particle flow:** Small moving, color‑gradient **dots** traverse each arrow in both directions for visual “flow” of transitions.
- **Warm‑up message** until at least two states are observed; then full rendering.
- **Performance guards:** `max_polylines_count = 100`, `max_labels_count = 100`.

### Inputs, controls & outputs

- **Model inputs:** `lookback_period` (reactivity vs stability), **toggle** for custom indicators, three `input.source()`slots.
- **Visual inputs:** `scale_factor`, `Horizontal Shift` (GUI offset), `node_size`, plus **Bull/Neutral/Bear colors**.
- **Output:** `plot(market_state, display = data_window)` gives the **bias** (+1/0/−1) for programmatic use; candles overlay is disabled by default for the visual panel.

---

## Key features (at a glance)

1. **Three‑state regime detection** (Bull/Bear/Neutral) from either **Z‑scored returns** or **Z‑scored custom indicators** (3 inputs, equal weight).
2. **Hysteresis & clamping** to suppress churn: indicators clamped to ±3σ; combined signal uses ±0.05 buffers; price Z‑score uses ±⅓σ thresholds.
3. **Rolling Markov chain** over a configurable window; **transition matrix** rebuilt on each confirmed bar.
4. **Laplace‑smoothed probabilities** so no direction is ever stuck at exactly 0% with sparse data.
5. **Strict bar‑close updates (no repaint)** for both regime classification and probability counts. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)
6. **Animated 3D UI**: spheres for states (with pulsing active state), **bidirectional** arrows, **moving particle flows**, and **probability labels** on each path. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)
7. **Readable interpretation**: self‑loops = persistence; arrow prominence = typical next switch; **directional asymmetry** shown explicitly. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)
8. **Tunable lookback** trades responsiveness vs stability; shorter windows adapt faster but can be noisier. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)
9. **Programmatic integration**: exposes a **bias plot** consumable via `input.source()` in other Pine scripts/strategies. [TradingView](https://www.tradingview.com/script/yA0KOPDO-Markov-Chain-3D-Fractalyst/)
10. **Customizability**: colors, scale, layout shift, and node size for aesthetic fit; optional custom indicator stack to match your market.

-

-—

## 1) Three‑state regime detection (Bull / Bear / Neutral)

### A) Price‑based mode (Z‑score of returns)

**Signal construction**

- **Return:** rt=Ct−Ct−1Ct−1rt=Ct−1Ct−Ct−1
- **Rolling stats (lookback = LL):**
    
    μt=SMA(rt,L),σt=stdev(rt,L)μt=SMA(rt,L),σt=stdev(rt,L)
    
- **Z‑score:**
    
    zt={rt−μtσt,σt>00,σt=0zt=⎩⎨⎧σtrt−μt,0,σt>0σt=0
    

**Classification (balanced, fast‑reacting thresholds)**

- Bull if zt>+13zt>+31
- Bear if zt<−13zt<−31
- Neutral otherwise

**Why this works**

- Using *returns* instead of absolute price changes makes the z‑score broadly comparable across assets/timeframes.
- A ±⅓σ band gives a reasonable “sensitivity” to departures from recent “normal,” leading to frequent but not frantic regime changes.

**What to tune**

- **Lookback LL:** shorter ⇒ adapts faster (more flips); longer ⇒ smoother (fewer flips).
- **Thresholds:** increasing to ±0.5σ or ±0.7σ can tame choppy intraday series; decreasing to ±0.25σ can catch early moves.

---

### B) Custom‑indicator mode (3 inputs, normalized + consensus)

**Normalization per input**

- For each input xt(k)xt(k):
    
    zt(k)=clamp ⁣(xt(k)−SMA(xt(k),L)stdev(xt(k),L),−3,+3)zt(k)=clamp(stdev(xt(k),L)xt(k)−SMA(xt(k),L),−3,+3)
    

**Per‑input hard decision (±1σ)**

- Bull if zt(k)>+1zt(k)>+1, Bear if zt(k)<−1zt(k)<−1, else Neutral (0).

**Equal‑weight consensus + hysteresis**

- Let st(k)∈{−1,0,+1}st(k)∈{−1,0,+1}. With equal weights wk=13wk=31:
    
    combined=∑kwk st(k)∈{−1,−23,−13,0,13,23,1}combined=∑kwkst(k)∈{−1,−32,−31,0,31,32,1}
    
- Final state: **Bull if combined > 0.05; Bear if < −0.05; else Neutral.**

**Important nuance**

- With **equal weights**, the smallest nonzero |combined| is **1/3**, so the ±0.05 hysteresis **has the same effect as requiring a strict non‑zero majority**—i.e., at least one net positive (or net negative) vote that isn’t canceled out.
    
    If you later **change weights** (e.g., 0.5/0.3/0.2), the ±0.05 buffer becomes a true “dead band” near 0.
    

**What to tune**

- Make the **indicator weights inputs** (they’re constants in the code) if your sources differ in reliability.
- Tighten/loosen the **±1σ** per‑input threshold depending on indicator volatility and how selective you want each indicator’s vote to be.

---

## 2) Hysteresis & clamping (churn control)

**Clamping (±3σ)**

- Protects against rare outliers that would otherwise dominate averages or trigger “whipsaw” classifications.
- In *this* script, clamping does **not** change the ±1σ decision itself (still a hard threshold) but **stabilizes** the normalized values feeding that decision—especially if you later reuse the zz-scores elsewhere.

**Hysteresis (±0.05 around 0)**

- Acts as a **dead zone** for the *combined* signal so tiny imbalances don’t cause flips.
- As noted, with equal 1/3 weights the effective dead zone is dominated by the discrete step size (1/3), so the buffer mostly matters once you **change weights**.

**If you need “temporal” stickiness**

- Current hysteresis is **spatial** (on the decision axis) rather than **temporal** (no memory of prior state).
- To make regimes “stickier,” you can add *state‑dependent* thresholds, e.g. require combined ≥ +⅔ to **enter** Bull but only fall below +⅓ to **exit** Bull (true temporal hysteresis).

---

## 3) Rolling Markov chain (on confirmed bars)

**What’s stored**

- A rolling array `recent_states` of the last **L+1L+1** confirmed states (encoded 0=Bull, 1=Bear, 2=Neutral).

**How the matrix is built each bar**

- **State counts:** frequency of each state in the window (empirical occupancy).
- **Transition counts:** for each consecutive pair (St−1,St)(St−1,St), increment the cell (from=St−1, to=St)(from=St−1,to=St).
- **Total transitions:** N=window size−1N=window size−1.

**Complexity & reliability**

- Rebuild is **O(L)O(L)** per confirmed bar—fine for L ≤ ~100.
- Because everything is computed on **confirmed bars**, you get a clean, reproducible history (no repaint).

**Conceptual note**

- This is a **time‑varying (non‑homogeneous) Markov chain** because the matrix is re‑estimated from a rolling window; it intentionally adapts as market behavior shifts.

---

## 4) Laplace‑smoothed probabilities (no zeros, stable early estimates)

**Formula per row (“from” state ii)**

p^i→j  =  ni→j+1∑k(ni→k+1)  =  ni→j+1Ni+3

p^i→j=∑k(ni→k+1)ni→j+1=Ni+3ni→j+1

- ni→jni→j: observed transitions from ii to jj within the window
- NiNi: total transitions out of state ii (row sum)
- “+1” with 3 outcomes is a **Dirichlet(1,1,1)** prior ⇒ **uniform** prior belief.

**What this buys you**

- **Never 0% / 100%** with sparse data (robust early in the window).
- **Shrinkage** toward 1/3 in proportion to small NiNi (less overfitting when you’ve seen few transitions from a given state).

**Tiny worked example**

Suppose from **Bull** over the window you observed:

- Bull→Bull: 15, Bull→Bear: 3, Bull→Neutral: 2 ⇒ Ni=20Ni=20

Then

- p(Bull→Bull)=15+120+3=1623≈0.6957p(Bull→Bull)=20+315+1=2316≈0.6957
- p(Bull→Bear)=3+123=423≈0.1739p(Bull→Bear)=233+1=234≈0.1739
- p(Bull→Neutral)=2+123=323≈0.1304p(Bull→Neutral)=232+1=233≈0.1304

**Early‑window behavior**

- If Ni=0Ni=0 (no transitions yet from state ii), each pi→j=1/3pi→j=1/3 by construction—sensible default.

---

## 5) Strict bar‑close updates (no repaint)

**Two safeguards in the code**

1. `market_state := barstate.isconfirmed ? market_state : market_state[1]`
    
    → On an unconfirmed (building) bar, the regime **holds** the prior value.
    
2. Transition updates and rolling window maintenance are gated by
    
    `if barstate.isconfirmed`
    
    → The Markov counts/probabilities **only** incorporate *closed* bars.
    

**Visualization note**

- The heavy 3D drawing runs only on the **last bar** (`barstate.islast`) for performance. Historical probabilities/regimes are still computed (on close), but the 3D scene you see is the current snapshot.

---

## Practical tuning & usage patterns

**Choosing the lookback LL**

- **Fast markets / intraday:** 20–50 captures recent behavior quickly.
- **Swing / daily:** 50–100 reduces churn and variance in transition estimates.

**Interpreting the matrix**

- On the **current state’s row**, the largest off‑diagonal cell is the **most likely next regime**; the diagonal cell is the **persistence** (expected to remain).
- Expected *dwell time* in a (locally) stationary row is ≈1/(1−pii)≈1/(1−pii); with rolling windows, treat this as a **local** heuristic.

**Strategy wiring (examples)**

- **Continuation bias:** If current state is Bull and pBull→BullpBull→Bull ≥ 0.6, allow trend‑following entries; else tighten stops or reduce size.
- **Switch anticipation:** If current state is Neutral and pNeutral→BullpNeutral→Bull ≫ pNeutral→BearpNeutral→Bear, favor long breakouts.
- **Regime filter:** Trade long setups only when `bias == +1`, short only when `bias == −1`, skip when `0`.

**Diagnostics you can add (quick wins)**

- Expose **row counts NiNi** in the data window so you can judge confidence (a row with Ni=3Ni=3 is far less reliable than Ni=60Ni=60).
- Plot a mini **confusion matrix** over time (or simply the three **row maxima**) to see how the chain’s structure is evolving.

---

## Subtle implementation notes (worth knowing)

- **Hysteresis semantics:** It’s *not temporal*—it’s a buffer at the aggregation step. If you want “sticky Bull until proof otherwise,” add state‑dependent entry/exit thresholds based on the **previous** decided regime.
- **“Scale Factor” vs what it does:** In the current code, `scale_factor` mainly shifts the drawing vertically (it’s not an object scale). The **size** of spheres is `node_size`. (Tooltips in the UI appear to be mismatched for a couple of inputs.)
- **Uniform prior strength:** Laplace here is “+1 per outcome.” If you want the prior to be weaker/stronger, generalize to “+α” per outcome (Dirichlet(α,…,α)). For example, α=0.25 gives lighter shrinkage toward 1/3; α=2 gives heavier.

---

## Quick checklist for adapting this to your market

- Pick **mode**: price Z‑score for simplicity; custom‑indicator mode if you have curated factors (e.g., trend, volatility, breadth).
- Set **lookback** to match your timeframe’s regime cadence.
- (Custom mode) Decide **weights** and consider making them `input.float` so you can optimize or at least A/B test.
- Consider **tighter thresholds** for hyper‑volatile instruments (crypto, small‑cap futures) and **looser** ones for staid FX crosses.
- Surface **row counts** and, optionally, a **confidence metric** (e.g., Ni/(L−1)Ni/(L−1)) alongside probabilities before you lean on them in risk decisions.

If you want, I can produce a small companion script that **exposes the row counts**

-