# Football Betting ML System

Machine learning and statistical strategies for football betting — tested on the
Beat The Bookie dataset (479K matches, 2005–2015, 32 bookmakers, 72 time
points) and live via The Odds API.

## Data

- **Historical:** [Beat The Bookie Worldwide Football Dataset](https://www.kaggle.com/datasets/austro/beat-the-bookie-worldwide-football-dataset)
  — 479K matches, up to 32 bookmakers, 72 timestamps per match
- **Live:** [The Odds API](https://the-odds-api.com) — free tier (500 req/mo)

## Strategies

### 1. Consensus Pricing (`consensus.py`, `consensus_monitor.py`, `opening_consensus.py`)

**Idea:** Consensus of all bookmakers is a better estimator of true probability
than any single bookmaker. Back outcomes where one bookmaker's odds are 5%+
above consensus.

| Variant | Data | Key Result |
|---|---|---|
| `consensus.py` | Closing odds, 479K matches, 32 bookies | +0.25% ROI — thin but positive |
| `opening_consensus.py` | Opening odds (timestamp 0), 72 timestamps | Uses richer odds_series data |
| `ts_consensus.py` | Time-series consensus (72 time points) | Edge concentrated at 15%+ outliers |
| `consensus_monitor.py` | Live (The Odds API) / historical scan | Production-ready monitor with cron |

**Best edge:** 15%+ edge trades: 71 bets/month, +12% ROI, £6.10 EV/bet.

### 2. Scalping (`scalping.py`, `betfair_scalper.py`)

**Idea:** Exploit intra-market price movements across 72 time points.

| Script | Approach |
|---|---|
| `scalping.py` | Simulate back/lay across consecutive time points |
| `betfair_scalper.py` | Full Betfair scalping framework — requires real tick data |

**Status:** 92% of markets have scalpable volatility in proxy data. Requires
real Betfair tick data (Historical API) to validate.

### 3. Back-to-Lay (`betfair_bot.py`)

Back at opening odds, lay at closing odds. Sim-only — original backtest
invalidated by a data bug (bookmaker back prices used as proxy for exchange lay
prices).

### 4. Asian vs European Markets (`run.py`)

**Idea:** Test whether Asian leagues have exploitable pricing inefficiencies
compared to efficient European top-5 markets. XGBoost & logistic regression
on 479K matches.

**Result:** All strategies lost money (−84% to −97%).

### 5. Arbitrage (`monitor.py`, `time_series.py`)

Inter-bookmaker arbitrage detection across 72 time points and 32 bookmakers.
Simulation + live modes.

## Scripts Reference

| Script | Strategy | Mode |
|---|---|---|
| `consensus.py` | Consensus pricing (closing odds) | Historical sim |
| `consensus_monitor.py` | Consensus value scanning | Historical + live |
| `opening_consensus.py` | Consensus pricing (opening odds) | Historical sim |
| `ts_consensus.py` | Pre-match odds consensus (t=71), edge ≥8%, odds ≤2.5, £10 flat, pause5L | Historical sim (opt.) |
| `paper_trader.py` | Live paper trader (Consensus) | Live (requires ODDS_API_KEY) |
| `run.py` | Asian vs European ML | Historical sim |
| `scalping.py` | Time-scalping sim | Historical sim |
| `betfair_scalper.py` | Betfair scalping framework | Live + sim |
| `betfair_bot.py` | Back-to-lay arb | Sim only |
| `monitor.py` | Arbitrage monitor | Historical + live |
| `time_series.py` | Multiple time-series strategies | Historical sim |

## Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
```

## Quick Start

```bash
# Historical consensus backtest
python3 consensus.py

# Opening odds consensus
python3 opening_consensus.py

# Time-series consensus
python3 ts_consensus.py

# Live consensus monitor (requires API key)
python3 consensus_monitor.py --mode live --api-key YOUR_KEY

# Paper trader (requires API key)
export ODDS_API_KEY=your_key
python3 paper_trader.py
```

## Key Findings

- **Consensus pricing works** — +0.25% ROI over 10 years, +12% on high-edge trades
- **Outcome prediction ML fails** — no model outperforms the market consistently
- **Scalping is unvalidated** — proxy data suggests potential, needs real tick data
- **Arbitrage opportunities exist** but thin and short-lived
- **Profitable edge is real but small** — ~£1,700/year realistic from consensus alone
