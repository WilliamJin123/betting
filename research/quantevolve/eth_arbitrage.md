# GA-Evolved Ethereum Arbitrage Bot - Notebook Walkthrough & Analysis

Source notebook: `ga_eth_arbitrage_demo.ipynb`

## What This Notebook Does

This notebook demonstrates a **genetic algorithm (GA) for tuning the parameters of a simulated Ethereum arbitrage bot**. The core idea: instead of hand-picking strategy parameters (when to trade, how to model gas costs, slippage tolerance, etc.), let an evolutionary process discover better values offline, then deploy those parameters in a real-time executor.

The architecture is:
1. **Offline GA** evolves a parameter vector across generations using simulated market episodes
2. **Online heuristic executor** takes the evolved parameters and acts on price divergences between two AMMs

This is purely a simulation -- no real trades, no live network connections.

## AMM Market Simulator

The simulator models two **constant-product AMMs** (Uniswap v2 style) trading ETH/USDC on two venues (A and B).

### How the AMM works
- Each AMM holds reserves of base (ETH) and quote (USDC) tokens
- Spot price = `quote_reserve / base_reserve`
- Swaps follow the `x * y = k` invariant with a 30 bps (0.30%) fee
- `swap_base_for_quote()` and `swap_quote_for_base()` update reserves after each trade

### Market dynamics per step
- Random order flow hits each AMM independently (normal distribution around ~1000 USDC or ~0.3 ETH)
- This causes the two AMMs to **drift apart in price**, creating arbitrage opportunities
- Network gas load is drawn from `N(100, 60)`, affecting gas costs

### Realism terms
| Factor | Implementation |
|--------|---------------|
| **Gas cost** | Base $1.50 + $0.50 per 100k gwei of network load, multiplied by a tunable `gas_k` |
| **Slippage** | Compared against a configurable guard (in bps); trade rejected if slippage exceeds guard |
| **Latency** | Penalizes the edge by 0.01 bps per ms of assumed latency |
| **MEV loss** | 3% chance of losing 8 bps to MEV extraction |

## Arbitrage Execution Logic

The `execute_arbitrage()` function implements a two-leg arb: buy ETH on the cheaper AMM, sell on the expensive one.

Steps:
1. Simulate the two swaps on deep copies to calculate gross edge (in bps)
2. Check slippage on both legs against `slip_guard_bps` -- reject if exceeded
3. Subtract a latency discount from the edge
4. Compare effective edge against `threshold_bps` -- reject if below
5. Apply probabilistic MEV loss
6. Subtract gas cost (`gas_k * 2 * $2.00`)
7. If PnL > 0, execute the real swaps (mutating AMM state)

The parameter vector being evolved is: `{threshold_bps, gas_k, slip_guard_bps, latency_ms}`

## Genetic Algorithm

### Individual representation
Each individual is a dict of four floats:
- `threshold_bps`: minimum edge (in bps) required before acting (higher = more selective)
- `gas_k`: multiplier on gas cost model (lower = more aggressive gas assumption)
- `slip_guard_bps`: max slippage tolerance per leg
- `latency_ms`: assumed execution latency

### Fitness function
```
fitness = total_pnl - 0.5 * pnl_std - 0.1 * max(0, trades - 20)
```
This rewards high PnL, penalizes variance, and penalizes excessive trading (>20 trades per episode).

### Evolution mechanics
- **Population**: 24 individuals
- **Generations**: 10
- **Selection**: Top 6 elites survive directly
- **Crossover**: Uniform (50/50 per gene from two random elite parents)
- **Mutation**: 90% chance, Gaussian perturbation with configurable sigma
- **Evaluation**: Each individual runs 3 episodes, fitness averaged

### Baseline comparison
The baseline uses hand-picked defaults: `threshold_bps=12, gas_k=1.0, slip_guard_bps=60, latency_ms=250`

## Results

### GA training progression
The GA fitness fluctuates significantly across generations (from 13.58 to 78.66), reflecting the stochastic nature of the simulation and the small number of evaluation episodes (3) per individual.

Best parameters found: `threshold_bps=38.6, gas_k=0.22, slip_guard_bps=154, latency_ms=309`

Key shifts from baseline:
- **Threshold nearly 3x higher** (38.6 vs 12 bps): the GA learned to wait for larger opportunities
- **Gas multiplier cut to 0.22** (vs 1.0): the GA aggressively assumes low gas, taking trades the baseline would skip
- **Slippage guard 2.5x wider** (154 vs 60 bps): more tolerant of price impact
- **Latency slightly higher** (309 vs 250 ms): not a major shift

### Held-out evaluation (8 unseen episodes)

| Metric | Baseline | GA |
|--------|----------|-----|
| Mean total PnL | $14.66 | $32.82 |
| PnL std | $11.04 | $31.98 |
| Mean trades | 2.5 | 1.375 |

The GA strategy roughly **doubles the mean PnL** but with **3x the variance**. It trades less frequently but takes much larger individual positions when it does act.

### PnL distribution characteristics
- Baseline: small, frequent wins ($3-10 per trade), 2-5 trades per episode
- GA: rare but large wins ($20-33 per trade), 0-4 trades per episode, with some episodes producing $0 (no trades taken)
- Both strategies have episodes with zero PnL (no opportunity met the threshold)

---

## Insights

### The GA found a real behavioral shift
The evolved strategy isn't just a minor tweak -- it represents a fundamentally different philosophy: **be extremely patient, then bet big**. The baseline takes many small edges; the GA waits for large dislocations and captures most of the move. This maps to a known pattern in real arbitrage: high-frequency small-edge strategies require execution speed and volume, while selective large-edge strategies require patience and conviction.

### Low `gas_k` is doing the heavy lifting
The single most impactful parameter change is `gas_k` dropping from 1.0 to 0.22. This effectively tells the executor "assume gas will be 5x cheaper than the model predicts." In the simulation, this is valid because the gas model is conservative. In reality, this would be dangerous -- gas spikes are common and a 0.22 multiplier would massively underestimate costs during congestion.

### The fitness function shapes strategy style
The `-0.5 * pnl_std` penalty is meant to encourage consistency, but the GA still evolved toward high-variance strategies. This suggests the PnL reward overwhelms the variance penalty at this weighting. A stronger variance penalty (or Sharpe-ratio-based fitness) would push toward the more consistent baseline-like behavior.

---

## Limitations

### Simulation fidelity
1. **Constant-product AMM is simplified**: Real Uniswap v2 has much deeper liquidity and tighter spreads. The simulator's reserve sizes (1000 ETH / $3M USDC) create unrealistic price impact for $5k trades. Real arb on pools this size would have much smaller edges.

2. **Random walk order flow**: Real order flow is clustered, autocorrelated, and driven by news/events. The i.i.d. Gaussian flow in the simulator doesn't produce the kind of persistent dislocations or flash crashes that create real arb opportunities.

3. **No competition**: In reality, dozens of searchers compete for the same arb. The simulator assumes you're the only player, so every opportunity you see is available. Real arb is a race; most opportunities are captured within 1-2 blocks.

4. **Gas model is too simple**: Real gas is a function of network congestion, block space demand, and priority fees. The `N(100, 60)` model doesn't capture gas spikes (which can 10-100x base costs) or the EIP-1559 base fee mechanism.

5. **MEV modeling is naive**: A flat 3% chance of 8 bps loss doesn't reflect how MEV actually works. In practice, MEV bots front-run or sandwich your transactions based on the specific opportunity size and mempool state. Larger arbs attract more MEV attention.

### Optimization concerns
6. **Evaluation noise**: Each individual is evaluated on only 3 episodes (300 steps each). The GA's fitness signal is extremely noisy -- the best individual in generation 9 (fitness 78.66) drops to the best in generation 10 at 13.58. This noise means the "best" parameters may be lucky rather than genuinely superior.

7. **No out-of-sample regime testing**: All training and evaluation use the same simulator distribution. If market conditions change (different volatility, different gas regime), the evolved parameters may break down entirely. There's no robustness testing.

8. **Small parameter space**: Only 4 parameters are evolved. Real arb bots have dozens of tunable knobs (position sizing curves, multi-hop routing weights, priority fee bidding strategies, timeout policies). The GA is optimizing a tiny slice of the full decision space.

9. **Seed dependency**: Results are seeded for reproducibility, but the GA's performance is likely sensitive to seed choice. The held-out evaluation uses seeds 700-707 -- a different seed range could tell a very different story given the high variance.

### Architectural gaps
10. **No multi-hop or cross-DEX routing**: Real arb bots find paths across 3+ pools (e.g., ETH->USDC on pool A, USDC->DAI on pool B, DAI->ETH on pool C). The two-pool model misses the dominant source of on-chain arb profit.

11. **No Uniswap v3 concentrated liquidity**: V3's tick-based liquidity is fundamentally different from v2's constant-product. Most DeFi liquidity is on v3 now; v2-only modeling is outdated.

12. **Static trade size**: The bot always trades $5k. Real bots size dynamically based on available liquidity, edge size, and gas cost. Optimal sizing is often the difference between profit and loss.

## Connection to Broader Research

This notebook complements the QuantEvolve notebook by applying evolutionary optimization at a different level:
- **QuantEvolve** uses an LLM to evolve *strategy logic* (which indicators, which rules)
- **This notebook** uses a classical GA to evolve *strategy parameters* (thresholds, multipliers)

The parameter-tuning approach is simpler and more tractable but limited to the strategy structure you define upfront. The LLM approach can discover novel structures but is harder to control and validate. A hybrid (LLM proposes structure, GA tunes parameters) would combine both strengths -- and is suggested in the notebook's "Next Steps" section via RL integration.

For DeFi arbitrage specifically, the key takeaway is that **the simulation-to-reality gap is the binding constraint**. Getting the GA to find good parameters in simulation is the easy part. The hard part is building a simulator that faithfully represents on-chain execution (MEV, gas auctions, block timing, liquidity dynamics) so that optimized parameters transfer to live trading.
