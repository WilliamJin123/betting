# QuantEvolve - Notebook Walkthrough & Analysis

https://arxiv.org/pdf/2510.18569

## What QuantEvolve Is

QuantEvolve is an **LLM-driven evolutionary algorithm for trading strategy generation**. The core idea: start with a simple seed strategy, then use an LLM (here, Gemini) to iteratively *mutate* it across generations -- generating new hypotheses, writing new code, backtesting it, and feeding results back into the next cycle. It draws from a paper of the same name, with citations throughout the notebook.

The analogy is biological evolution: strategies are "organisms," backtesting is "natural selection," and the LLM acts as the "mutation operator" that proposes intelligent (rather than random) modifications.

## Architecture: Three-Agent System

The system is built around three LLM-powered agents that run sequentially each generation:

### 1. ResearchAgent (Hypothesis Generation)
- **Input**: The current best strategy's hypothesis, code, backtest results, and insight.
- **Output**: A structured hypothesis for a *child* strategy, including rationale, objectives, risks, and ideas for future generations.
- **Role**: This is the "creative mutation" step. It reads what went wrong with the parent and proposes a fix (e.g., "add a volatility filter to avoid choppy markets").

### 2. CodingTeam (Implementation)
- **Input**: The new hypothesis + the parent's code.
- **Output**: A Python class called `EvolvedStrategy` using `backtesting.py`.
- **Role**: Translates the hypothesis into executable code. The prompt is heavily scaffolded -- it provides the class skeleton, helper functions, and even ATR calculation boilerplate.

### 3. EvaluationTeam (Insight Generation)
- **Input**: The hypothesis, code, and backtest results of the new strategy.
- **Output**: A one-sentence actionable insight (e.g., "the filter was too aggressive").
- **Role**: Closes the loop. This insight gets fed to the ResearchAgent in the *next* generation, guiding the direction of evolution.

## The Evolutionary Loop

```
Gen 0 (manual seed: SMA crossover)
  |
  v
For each generation 1..N:
  1. Select parent = current best strategy
  2. ResearchAgent proposes new hypothesis from parent's weakness
  3. CodingTeam implements hypothesis as code
  4. exec() the code, run backtest on AAPL 2018-2023
  5. EvaluationTeam generates insight from results
  6. Store in evolutionary_database
  7. If score > best, update best_strategy_archive
```

**Scoring**: `Combined Score = Sharpe Ratio + |Max Drawdown|` (where MDD is flipped to positive, so lower drawdown improves the score). This comes from the paper [cite: 371].

## Data & Execution Details

- **Data**: AAPL stock, 2018-01-01 to 2023-12-31, downloaded via `yfinance`.
- **Backtesting**: Uses `backtesting.py` with $100k starting cash, 0.2% commission per trade.
- **LLM**: Gemini 1.5 Flash via `google-generativeai` SDK, API key stored in Colab Secrets.
- **Generations**: Only 3 generations are run (configurable via `NUM_GENERATIONS`).
- **Code execution**: LLM-generated code is run via `exec()` -- the notebook explicitly flags this as a security risk.

## Generation 0: The Seed Strategy

A textbook **SMA crossover** strategy:
- Fast SMA (10-period) vs. Slow SMA (30-period)
- Buy when fast crosses above slow, sell when fast crosses below
- Known weakness: whipsaws in sideways/choppy markets
- This establishes a baseline for the evolutionary process to improve upon

## State Management

Two data structures track evolution:

1. **`evolutionary_database`** (list): Stores every strategy ever generated -- hypothesis, code, results, insight. This is the paper's "Evolutionary Database."
2. **`best_strategy_archive`** (dict): Tracks the single best strategy by combined score. This is a simplified version of the paper's "Feature Map" -- in the real paper this would be a multi-dimensional quality-diversity archive; here it's just a greedy best-so-far.

## Final Output

After the loop, the notebook:
- Prints the best strategy's hypothesis, insight, and code
- Creates a comparison table (baseline vs. evolved) across score, Sharpe, MDD, return, trades, win rate
- Plots the equity curve of the best evolved strategy

---

## Limitations

### Fundamental Issues

1. **Single asset, single timeframe**: Everything is tested on AAPL 2018-2023. This is a massive overfitting risk -- the evolved strategy may be learning AAPL-specific patterns (the big tech run-up, COVID crash/recovery) rather than generalizable alpha. No out-of-sample testing, no cross-asset validation.

2. **Greedy parent selection**: The loop always evolves from the current best. The paper describes a richer "Feature Map" for quality-diversity search -- exploring diverse strategies across multiple behavioral niches. This implementation collapses that into a single best, which means evolution can get stuck in local optima. One lucky strategy dominates and all children are minor variations of it.

3. **No error recovery / retry logic**: If any agent fails (bad hypothesis, broken code, backtest crash), the generation is simply skipped (`continue`). With only 3 generations, losing one is a 33% reduction in exploration. There's no mechanism to retry with a different prompt or fallback parent.

4. **`exec()` on LLM output**: The notebook acknowledges this is dangerous. The LLM could generate arbitrary code. In this Colab demo context it's acceptable, but this architecture fundamentally requires sandboxed execution (containers, restricted interpreters) for any real use.

5. **Scoring function is simplistic**: `Sharpe + |MDD|` treats both components equally with no weighting. A strategy with Sharpe=0.5 and MDD=-10% scores the same as Sharpe=5.5 and MDD=-15%. The paper likely has more nuanced fitness metrics. Also, MDD is calculated as `stats['Max. Drawdown [%]'] * -1`, which flips it positive -- meaning *worse* drawdowns (more negative) actually produce *larger* positive values, so MDD is being treated as a penalty that *increases* score. This appears to be a bug: the intended formula is likely `Sharpe - |MDD|` (penalizing drawdown) rather than `Sharpe + |MDD|`.

6. **No population / parallel search**: Real evolutionary algorithms maintain a *population* of candidates per generation. Here, each generation produces exactly one child. This severely limits exploration and makes the process fragile -- one bad generation wastes the entire slot.

### Practical Issues

7. **LLM code reliability**: The CodingTeam prompt assumes the LLM will produce clean, executable `backtesting.py` code. In practice, LLMs frequently produce code with subtle bugs (wrong indicator indexing, look-ahead bias, off-by-one errors in rolling windows). There's no static analysis or validation step between code generation and execution.

8. **No look-ahead bias protection**: The LLM could generate code that accidentally peeks at future data (e.g., using the full series rather than the rolling window available at time `t`). The `backtesting.py` framework mitigates some of this, but custom indicator functions defined via `exec()` could bypass those protections.

9. **Prompt coupling**: The prompts are heavily engineered with specific examples (ATR, SMA, "add a regime filter"). This steers the LLM toward a narrow class of strategies (indicator-based trend following with filters). More creative strategies (mean reversion, pairs trading, options-like payoffs, ML-based signals) are unlikely to emerge because the prompts don't leave room for them.

10. **No transaction cost realism**: 0.2% flat commission is a rough approximation. Real trading has slippage, bid-ask spread, market impact, and varying commission structures. Strategies optimized against flat commission may not survive real execution costs.

## Insights & Takeaways

### What's Interesting

- **LLM as mutation operator**: This is a genuinely clever idea. Traditional evolutionary strategy optimization uses random parameter perturbation or crossover. Using an LLM means mutations are *semantically meaningful* -- "add a volatility filter" is a much more targeted mutation than "randomly change n1 from 10 to 14." The insight-to-hypothesis feedback loop is the key innovation.

- **Structured hypothesis format**: Forcing the LLM to output `<hypothesis>`, `<rationale>`, `<risks_limitations>`, `<next_step_ideas>` is smart prompt engineering. It makes the LLM reason about *why* a change should work before writing code, and the `next_step_ideas` field seeds future generations.

- **The insight loop is the real value**: The three-agent cycle (hypothesize -> implement -> evaluate -> insight -> hypothesize) mirrors how a human quant researcher works. The EvaluationTeam's insight is the critical piece -- it converts raw backtest numbers into actionable direction for the next iteration.

### What Would Make This Actually Useful

- **Out-of-sample splits**: Train on 2018-2021, validate on 2022, test on 2023. Reject strategies that don't generalize.
- **Population-based search**: Run N candidates per generation in parallel, keep the top K, cross-pollinate ideas between them.
- **Multi-asset testing**: Evolve strategies that work across SPY, QQQ, individual stocks, etc.
- **Code validation layer**: Run static analysis, check for look-ahead bias, validate indicator calculations before backtesting.
- **Richer scoring**: Include Calmar ratio, Sortino ratio, maximum consecutive losses, profit factor -- not just Sharpe + MDD.
- **Sandboxed execution**: Replace `exec()` with a container or restricted Python interpreter.

### Connection to Broader Betting/Market Research

This notebook demonstrates a **meta-strategy**: using AI to *search the space of strategies* rather than to trade directly. This is relevant to your broader research because:

- The same evolutionary framework could generate strategies for prediction markets (Polymarket/Kalshi) or sports betting -- swap the backtesting engine for a different evaluation function.
- The "insight loop" pattern (hypothesis -> test -> learn -> iterate) is universal across all betting domains. The question is always whether the LLM's mutations are better than random search or human intuition.
- The biggest risk in any automated strategy generation system is **overfitting** -- strategies that look great in backtest but fail live. This is especially dangerous when the LLM has access to the same historical data it's optimizing against.
