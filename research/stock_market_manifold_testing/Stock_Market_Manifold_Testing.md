# Stock Market Manifold Testing - Analysis

https://richardaragon.substack.com/p/the-hidden-geometry-of-markets-how?r=23t7gr&utm_campaign=post&utm_medium=web&triedRedirect=true&_src_ref=youtube.com

## What This Notebook Is Trying To Do (Plain English)

The core idea: **Can we predict where the stock market is going by treating price data as a "landscape" that changes shape over time?**

Think of it like weather forecasting. Imagine you have a topographic map that keeps shifting — mountains rising, valleys forming. If you can measure *how fast the landscape is changing*, maybe you can predict what comes next. This notebook tries to do exactly that, but with stock prices instead of terrain.

### The Building Blocks

**1. KNN (K-Nearest Neighbors) — "Find me 5 days that looked like today"**

Standard KNN is like asking: "In all of history, which 5 trading days had the most similar prices/volume to today?" Then you look at what happened *after* those days and vote: did the price go up or down?

Problem: a day from 2021 might look numerically similar to today, but the market context was completely different (COVID recovery vs. AI bubble, etc.).

**2. Temporal KNN — "Find me 5 *recent* days that looked like today"**

The fix: add a penalty for old data. The distance formula becomes:

```
Distance² = (how different the prices are)² + lambda² × (how far apart in time)²
```

- High lambda = "only look at very recent history"
- Low lambda = "old days are fine too"
- Also enforces **causality**: you can only look at the past, never the future

**3. Tension — "How stretched out is the landscape right now?"**

For each day, compute:
- Average distance to the 5 **nearest** neighbors (KNN) = "how clustered is the nearby data?"
- Average distance to the 5 **furthest** neighbors (KFN) = "how spread out is the overall data?"
- **Tension = Furthest / Nearest**

High tension = the nearby points are clustered tightly but the overall space is huge = something unusual is happening (volatility spike, regime change).

**4. GRU (Neural Network) — "Can we predict tomorrow's tension from the last 20 days of tension?"**

Take the tension values from the past 20 days, feed them into a small recurrent neural network, and try to predict what tomorrow's tension (or tension "regime") will be.

**5. Scrambled Test — "Is any of this real, or just noise?"**

Shuffle all the dates randomly. If the model still works just as well on scrambled data, then it was never actually learning temporal patterns — it was just memorizing statistical noise.

---

## Experiment-by-Experiment Results

### Experiment 1: Synthetic Proof-of-Concept (Cell 2)

| Metric | Spatial KNN | Temporal KNN |
|--------|-------------|--------------|
| Accuracy | 86.8% | **96.6%** |

**What happened:** They generated fake data where the "decision boundary" (the line separating up/down) slowly rotates 90 degrees over time. Temporal KNN crushed it because the time-awareness let it track the rotation. Standard KNN got confused by old data.

**Verdict:** This proves the algorithm *works on problems designed for it*. This tells us nothing about real markets.

### Experiment 2: AAPL Next-Day Direction (Cell 3)

| Metric | Spatial KNN | Temporal KNN |
|--------|-------------|--------------|
| Accuracy | **54.0%** | **54.0%** |

Class balance: 589 down / 665 up → random baseline ~53%

**What happened:** Both methods are coin flips. Adding time awareness provided **zero improvement** on real stock data for next-day direction prediction.

### Experiment 3: AAPL Volatility Regime — Real vs Scrambled (Cell 4)

| Series | Spatial KNN | Temporal KNN |
|--------|-------------|--------------|
| **REAL** | 57.2% | 57.2% |
| **SCRAMBLED** | 54.4% | 57.8% |

**What happened:** The scrambled temporal KNN (57.8%) actually *outperformed* the real spatial KNN (57.2%). The "real vs scrambled" gap is tiny and within noise. The temporal structure is providing no meaningful signal.

### Experiment 4: GRU Tension Forecasting (Cell 5)

| Model | MSE | R² |
|-------|-----|-----|
| REAL GRU | 4.50 | **-0.226** |
| REAL Persistence | 1.36 | **-0.317** |
| SCRAMBLED GRU | 3.08 | **-0.073** |
| SCRAMBLED Persistence | 2.02 | **-0.959** |

**What happened:** ALL R² values are **negative**. An R² of 0 means "predicting the average every time." Negative means your model is *worse than just guessing the average*. The GRU learned nothing useful about future tension dynamics.

### Experiment 5: Regime Classification — 1-Step (Cell 6)

| Series | GRU Accuracy | Persistence Baseline |
|--------|-------------|---------------------|
| **REAL** | **89.9%** | 83.4% |
| **SCRAMBLED** | 40.9% | 32.8% |

**THIS LOOKS AMAZING — but it's fake.** Look at the confusion matrix:

```
REAL GRU confusion matrix:
[[  0   0  21]    ← Class 0: all 21 predicted as class 2 (WRONG)
 [  0   0   4]    ← Class 1: all 4 predicted as class 2 (WRONG)
 [  0   0 222]]   ← Class 2: all 222 predicted as class 2 (correct by default)
```

The GRU learned to **always predict "class 2"** (the dominant regime). Since class 2 makes up 222/247 = 89.9% of the test set, the "accuracy" is just the class imbalance. The model learned absolutely nothing about regime transitions — it just bets on the majority class every time.

### Experiment 6: Multi-Step Regime H=5 (Cell 7)

Same story. REAL GRU gets 89.5% by predicting almost exclusively class 2.

### Experiment 7: BTC-USD (Cell 8)

| Series | GRU Accuracy | Persistence |
|--------|-------------|-------------|
| **REAL** | 58.4% | 54.8% |
| **SCRAMBLED** | 39.3% | 46.8% |

Slightly more balanced classes for BTC, so the majority-class trick doesn't inflate the number as much. The 58.4% is marginal and the confusion matrix still shows heavy majority-class bias.

---

## Critical Problems Found

### 1. Features Are Not Normalized (FATAL)

The raw features are: Open (~150), High (~155), Low (~145), Close (~152), Volume (~80,000,000).

When you compute Euclidean distance, Volume **completely dominates** because its values are millions of times larger than prices. The KNN is essentially doing "find me days with similar volume" and ignoring price entirely. This single mistake invalidates every KNN result in the notebook.

**Fix:** Standardize each feature to zero mean, unit variance before computing distances.

### 2. Tension Values Are Absurd

Mean tension: **1,844,313**. This number should be in the range of maybe 2-50 for a well-calibrated metric. The insane values come from the un-normalized features (the Volume dimension stretches the furthest-neighbor distance to astronomical values).

### 3. Lambda Is Unscaled

`lambda_time = 5.0` but the time values are Julian dates (around 2,460,000). The time differences between adjacent days are ~1.0 in Julian day units, so `lambda * dt` for a 100-day gap = 500. Meanwhile, a Volume difference might be 50,000,000. So the time penalty is **negligible** compared to the Volume dimension — the "temporal" KNN is basically identical to spatial KNN. This explains why temporal KNN = spatial KNN in every real-data experiment.

### 4. The "90% Accuracy" Is Majority-Class Collapse

As shown above, the GRU simply predicts the dominant class. This is a classic pitfall with imbalanced classification. The regime labels are ~46% class 2, but because the test set happens to be late-period data (where one regime dominates), the accuracy is inflated.

### 5. No Train/Validation Tuning

Lambda, K, the GRU architecture, learning rate, number of regimes — all hardcoded. No hyperparameter search, no cross-validation, no sensitivity analysis. A single set of arbitrary parameters proves nothing.

### 6. O(N²) Brute-Force Scaling

The tension computation loops over every day and computes distance to all previous days. For 1,256 days this takes minutes. For a year of minute-level data (~100,000 points), this would take **days**. A real-time bot needs millisecond latency.

### 7. No Transaction Cost / Slippage Modeling

Even if the predictions worked, there's no simulation of what happens when you actually trade: commissions, bid-ask spread, slippage, market impact. A 54% accuracy signal gets eaten alive by transaction costs.

### 8. The Scrambled Test Doesn't Prove What It Claims

The notebook claims that Real >> Scrambled proves the temporal structure is meaningful. But Real ≈ Scrambled in most experiments. Where Real > Scrambled (regime forecasting), it's because the real series has a dominant regime that's easy to "predict" by always guessing it, while scrambling breaks that pattern.

---

## What Actually Has Value Here

Despite the problems, the *ideas* are not worthless:

1. **Temporal KNN with causal masking** is a legitimate technique in time-series analysis. The implementation just needs feature normalization and lambda tuning.

2. **Tension as a volatility proxy** has some theoretical merit. The ratio of far-to-near distances could detect regime shifts. But it needs to be computed on properly normalized features, and compared against well-known alternatives (realized volatility, VIX, Garman-Klass volatility, etc.).

3. **The scrambled-test methodology** is a good idea. It's a form of permutation testing — a real statistical tool. The execution here just doesn't produce the result they hoped for.

4. **Regime prediction via GRU** is a well-established approach. The problem isn't the architecture; it's the garbage-in (un-normalized tension) and the failure to handle class imbalance.

---

## Can We Use Any of This for a Market-Ready Bot?

### Short answer: No — not in its current state.

### Longer answer: The *framework* could be a small component of a larger system, but only after substantial rework.

**What would need to change:**

1. **Feature engineering overhaul:** Raw OHLCV is not predictive. You'd need returns, log-returns, normalized volume, moving averages, RSI, MACD, order book features, sentiment signals, etc.

2. **Proper normalization:** Z-score or min-max scaling on ALL features before any distance computation.

3. **Lambda calibration:** Either learn it from data (cross-validation) or scale it relative to feature magnitudes. The current value is essentially turning the temporal dimension off.

4. **Class-imbalance handling:** Use stratified splits, class weights, F1 optimization, or over/undersampling.

5. **Realistic evaluation:** Walk-forward validation, out-of-sample testing on unseen time periods, and inclusion of transaction costs.

6. **Speed:** Replace brute-force KNN with approximate nearest neighbors (Annoy, FAISS, etc.) or use a sliding window to limit the history.

7. **The fundamental question:** Next-day stock direction prediction is widely considered one of the hardest problems in quantitative finance. Armies of PhDs at Renaissance Technologies and Citadel with petabytes of data and proprietary signals barely beat the market. A temporal KNN on OHLCV data is not going to crack this.

### Where the ideas *might* help:

- **As a regime detector (not predictor):** Tension could serve as a *concurrent* indicator of market stress — telling you "right now the market geometry is stretched, so be cautious" — rather than trying to predict the future.
- **As one feature among many:** Feed the tension value into a larger feature set for a more sophisticated model. It's novel enough that it might add marginal value alongside standard technical indicators.
- **For betting/prediction markets (Polymarket, Kalshi):** These markets are often less efficient than stock markets. A regime-detection tool might help you identify when a market is in an unstable state where prices haven't caught up to reality yet. But you'd still need a separate signal for *direction*.
