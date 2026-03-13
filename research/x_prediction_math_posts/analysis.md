# RohOnChain Prediction Market Math Posts - Analysis

Sources: @RohOnChain (Roan) X posts + linked articles by navnoorbawa on Substack + Phemex coverage

The three linked posts are part of a broader series by RohOnChain about the math behind Polymarket arbitrage bots (particularly referencing "gabagool22," a well-known profitable bot). The content covers three major topics:

1. **Arbitrage detection** via Frank-Wolfe + Bregman Projection
2. **Position sizing** via Kelly Criterion
3. **Probability estimation** and market microstructure

---

## Topic 1: Arbitrage Detection (Frank-Wolfe + Bregman Projection)

### What is arbitrage in prediction markets? (Plain English)

Prediction markets like Polymarket let you buy "YES" or "NO" shares on events (e.g., "Will BTC hit $100k by July?"). A YES share pays $1.00 if the event happens, $0.00 if not. The price of a YES share = the market's implied probability.

**Key rule:** YES price + NO price should always = $1.00. If they don't, free money exists.

**Simple example:**
- Polymarket YES: $0.42
- Kalshi NO: $0.56
- You buy both for $0.98 total
- One of them MUST pay $1.00
- Guaranteed profit: $0.02 per pair (2.04% return, risk-free)

This also works within a single platform when there are multiple outcomes:
- Candidate A: $0.38, Candidate B: $0.33, Candidate C: $0.27
- Total: $0.98 (should be $1.00)
- Buy all three for $0.98, guaranteed $1.00 payout

### What does the Frank-Wolfe algorithm actually do?

The problem gets complicated when you have **many** markets with **many** outcomes that are **correlated**. You can't just eyeball it — you need to find the optimal combination of bets across dozens of markets simultaneously.

Think of it like this: you have a giant landscape of possible bet combinations. Most combinations lose money or are invalid (probabilities don't add up right). The "valid" region is a complex shape. You need to find the point inside that shape that maximizes your profit.

- **Frank-Wolfe** = an optimization algorithm that walks toward the best solution step by step, but unlike gradient descent, it stays inside the "valid region" at every step (no need to fix invalid bets after the fact)
- **Fully-Corrective** = at each step, it doesn't just take the next step — it re-optimizes across ALL previous steps. Faster convergence.
- **Adaptive** = automatically adjusts how big each step is based on how much progress it's making
- **Bregman Projection** = when you land outside the valid region (probabilities don't sum to 1), this snaps you back to the nearest valid point using a distance metric that makes sense for probabilities (not just Euclidean distance)

In practice: runs 50-150 iterations, finishes in a few minutes per market scan.

### Limitations

- **The $0.02 is gross profit.** Polymarket/Kalshi charge fees. Kalshi's fee formula: `ceil(0.07 * contracts * price * (1-price))`, which ranges from ~0.6% to 1.75%. A 2% gross arb can easily become 0.3% net or even negative after fees.
- **Execution risk.** By the time you detect the arb and submit both orders, the prices may have moved. You need sub-second execution.
- **Capital efficiency.** You lock up $0.98 to make $0.02. That capital is frozen until the event resolves (could be months). Annualized, this might be worse than a savings account.
- **Cross-platform arb requires capital on both platforms** + dealing with different settlement mechanisms, withdrawal delays, and counterparty risk.
- **The $40M figure is misleading.** That's aggregate across many sophisticated actors over years. Individual retail bots are fighting over scraps after the big players take the obvious arbs.
- **The algorithm itself is well-known.** Frank-Wolfe is from 1956. The edge isn't the algorithm — it's the infrastructure (speed, capital, API access).

---

## Topic 2: Position Sizing (Kelly Criterion)

### What is it? (Plain English)

If you think an event has a 75% chance of happening but the market prices it at 60%, you have an "edge." But how much of your money should you bet?

- Bet too little = you grow slowly and waste your edge
- Bet too much = one bad streak wipes you out

Kelly Criterion gives the mathematically optimal answer for maximum long-term growth:

```
f* = (p - m) / (1 - m)

where:
  f* = fraction of your capital to bet
  p  = YOUR estimate of true probability
  m  = current market price (the market's implied probability)
```

**Worked example:**
- Market price: $0.60 (market thinks 60% likely)
- Your estimate: 75% likely
- f* = (0.75 - 0.60) / (1 - 0.60) = 0.15 / 0.40 = **37.5% of your bankroll**

### Why practitioners use fractional Kelly

Full Kelly is mathematically optimal but **terrifyingly volatile**. If your probability estimate is even slightly wrong, you can blow up.

- **Full Kelly:** ~33% chance of losing half your bankroll at some point
- **Half Kelly (0.5x):** ~75% of full Kelly's returns, much smoother ride
- **Quarter Kelly (0.25x):** <3% chance of ruin, still meaningful growth

**Rule of thumb from the articles:** Fractional Kelly at 0.25x-0.50x is standard practice.

Real position sizing in practice:
- High-confidence opportunities: 12-15% of capital
- Moderate confidence: 2-5% of capital
- No clear edge: don't trade

### Limitations

- **Garbage in, garbage out.** Kelly's output is only as good as your probability estimate `p`. If you think something is 75% likely but it's actually 60%, Kelly tells you to bet big on a losing proposition. The entire system depends on having BETTER probability estimates than the market — which is the hard part nobody's formula solves.
- **Assumes independent bets.** If you have 10 positions and they're all correlated (e.g., all crypto markets crash together), Kelly's per-bet sizing is dangerously optimistic.
- **Assumes you can bet any fraction.** Real markets have minimum order sizes and liquidity constraints.
- **Ignores time value of money.** A 37.5% Kelly bet that locks up capital for 6 months might be worse than a 5% Kelly bet that resolves in 2 days.
- **Doesn't tell you WHAT to bet on.** Kelly is a position-sizing tool, not a signal generator. You still need an edge.

---

## Topic 3: Market Microstructure & Probability Estimation

### Order Book Signals

The articles describe how order flow predicts short-term price movement:

```
Order Book Imbalance (OBI) = (Bid_volume - Ask_volume) / (Bid_volume + Ask_volume)
```

- OBI > 0 = more buyers than sellers = price likely to rise short-term
- Empirical finding: OBI explains ~65% of short-interval price variance (R² = 0.65)
- Imbalance Ratio > 0.65 predicts price increase within 15-30 minutes (58% accuracy)

```
Volume-Adjusted Mid Price (VAMP) = (P_bid * Q_ask + P_ask * Q_bid) / (Q_bid + Q_ask)
```

VAMP gives a "fairer" mid-price that accounts for order sizes, not just prices.

### Dynamic Hedging Near Resolution

As an event gets closer to resolving, small probability shifts cause huge price swings:

```
Position(t) = Initial_Position * sqrt(T_remaining / T_initial)
```

Example: $10,000 position placed 30 days before resolution:
- 7 days left: reduce to $4,830
- 1 day left: reduce to $1,826

This is because gamma (rate of price change) explodes near resolution.

### Monte Carlo for Strategy Validation

The Monte Carlo post describes simulating thousands of possible market scenarios to stress-test a strategy before deploying it. Run 10,000 simulations with random price paths, check how often you go bust, and only deploy if the strategy survives the worst cases.

### Limitations

- **58% accuracy from OBI is marginal.** After fees and slippage, a 58% edge on short-term price moves is thin. You need high volume to make it work.
- **OBI signals are ephemeral.** By the time you detect and act on order book imbalance, market makers have already adjusted. This is a speed game.
- **Prediction markets are thin.** Unlike stock markets with billions in daily volume, Polymarket markets often have $10k-$100k in the order book. Large orders move prices against you.
- **The 90-94% accuracy claim** (markets being "right" 30 days / hours before events) means the market is already efficient. There's not much room to be smarter.
- **Wash trading is rampant.** 14% of wallets, 20-60% of volume. Order book signals are polluted by fake orders.

---

## Target Performance from the Articles

The articles claim a well-built systematic prediction market strategy targets:

| Metric | Target |
|--------|--------|
| Annual Return | 15-25% |
| Sharpe Ratio | 2.0-2.8 |
| Win Rate | 52-58% |
| Avg Edge per Trade | 2-4% |
| Max Drawdown | 12-18% |

**Edge breakdown:** 60% probability estimation, 25% arbitrage, 15% microstructure

---

## Overall Assessment: What Can We Actually Use?

### Directly usable (with work)

1. **Kelly Criterion for position sizing** - Simple formula, proven math, immediate value for any bot. Must use fractional Kelly (0.25x). This is a must-have for any bot we build, regardless of the market.

2. **Cross-market arbitrage detection** - The simple version (check if YES + NO < $1.00 across platforms) is trivial to implement. Worth building as a baseline scanner even if the edges are small.

3. **Order Book Imbalance** - If we're building a Polymarket bot, monitoring OBI is cheap to implement and gives a modest short-term directional signal.

### Useful as concepts, not directly implementable

4. **Frank-Wolfe + Bregman Projection** - Only needed for complex multi-market combinatorial arbitrage. Overkill for starting out. The simple arb scanner covers 90% of opportunities.

5. **Dynamic position reduction near resolution** - Good risk management principle. Easy to hardcode as a rule rather than building the full gamma model.

6. **Monte Carlo backtesting** - Good practice for validating any strategy. Standard tooling exists (no need to build from scratch).

### Not usable / oversold

7. **The "guaranteed profit" framing** - The articles heavily market the idea of "risk-free" money. In practice, the risk-free arbs are tiny (sub-1% after fees), capital-intensive, and competed away quickly. The real money comes from directional bets with edge — which is the hard problem none of these posts solve.

8. **gabagool22 as a template** - Presented as proof that bots print money. But gabagool22 likely has: (a) very fast infrastructure, (b) large capital base, (c) direct API access with lower fees, (d) years of iteration. Copying the math doesn't copy the edge.

---

## Key Takeaway

These posts provide a solid **math toolkit** (Kelly, arbitrage constraints, OBI) but **zero alpha generation**. They tell you how to *size* bets and *detect* mispricings, but not how to *estimate probabilities better than the market* — which is where 60% of the edge supposedly comes from. That's the hard part, and it's conspicuously absent from all of these posts.

For our bot: Kelly + simple arb scanning + OBI monitoring = a reasonable starting foundation. But we still need to solve the probability estimation problem, which is the actual core of any profitable prediction market bot.
