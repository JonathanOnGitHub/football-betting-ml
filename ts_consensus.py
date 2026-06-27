#!/usr/bin/env python3
"""Consensus pricing on time-series data — optimised, using FIRST published odds.
   Edge >= 8%, odds <= 2.5, £10 flat stake, pause after 5 losses."""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings, time
warnings.filterwarnings('ignore')
t0 = time.time()

SEP = "=" * 60
DATA = Path.home() / '.cache/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2'
OUT = Path('/home/burley/football-ml')

# ── Optimised config ────────────────────────────────────────────────
BANKROLL = 1000
FLAT_STAKE = 10           # £10 fixed per bet (1% of initial)
MIN_EDGE = 8.0            # Only bet when bookmaker 8%+ above consensus
MIN_ODDS = 1.3
MAX_ODDS = 2.5            # Cuts out 2.5-3.0 range that lost -£4,858
CONSEC_LOSS_PAUSE = 5     # Stop after 5 consecutive losses, resume on next win
TS = 71                   # Pre-match odds (most bookmakers; t=0 = first published)

print(SEP)
print("CONSENSUS PRICING — PRE-MATCH ODDS (t=71)")
print(f"  Edge >= {MIN_EDGE}%, odds {MIN_ODDS}-{MAX_ODDS}, £{FLAT_STAKE} flat, pause after {CONSEC_LOSS_PAUSE}L")
print(SEP)

# Load pre-match odds (t=71)
basic = ['match_id','match_date','score_home','score_away']
cols = basic[:]
for outcome in ['home','draw','away']:
    for b in range(1, 9):
        cols.append(f'{outcome}_b{b}_{TS}')

print("\nLoading...")
df = pd.read_csv(DATA / 'odds_series.csv.gz', compression='gzip',
                 encoding='latin1', usecols=cols)
df = df.dropna(subset=['score_home','score_away'])
print(f"  {len(df):,} completed matches")

# ── Simulation ──
bankroll = BANKROLL
peak = BANKROLL
trades = []
consec_loss = 0
paused = False
max_dd = 0
skipped_sparse = 0  # matches with <2 bookmakers

for idx, row in df.iterrows():
    home_s = float(row['score_home'])
    away_s = float(row['score_away'])

    # Collect all bookmaker first-published odds
    bk_data = {'home': [], 'draw': [], 'away': []}
    for outcome in ['home','draw','away']:
        for b in range(1, 9):
            val = row.get(f'{outcome}_b{b}_{TS}')
            if pd.notna(val) and val > 0:
                bk_data[outcome].append((val, f'b{b}'))

    # Need at least 2 bookmakers for a meaningful consensus
    min_bks = min(len(v) for v in bk_data.values())
    if min_bks < 2:
        skipped_sparse += 1
        continue

    # Consensus = median (robust)
    consensus = {}
    for outcome in ['home','draw','away']:
        odds_vals = [v[0] for v in bk_data[outcome]]
        consensus[outcome] = np.median(odds_vals)

    # Scan for value bets
    bet_placed = False
    for outcome in ['home','draw','away']:
        if bet_placed:
            break
        for bookie_odds, bookie_name in bk_data[outcome]:
            if not (MIN_ODDS <= bookie_odds <= MAX_ODDS):
                continue
            if bookie_odds <= consensus[outcome]:
                continue

            edge = (bookie_odds / consensus[outcome] - 1) * 100
            if edge < MIN_EDGE:
                continue

            # Determine outcome
            if outcome == 'home':
                would_win = home_s > away_s
            elif outcome == 'draw':
                would_win = home_s == away_s
            else:
                would_win = away_s > home_s

            # Consecutive loss pause
            if paused:
                if would_win:
                    paused = False
                    consec_loss = 0
                else:
                    continue
            if consec_loss >= CONSEC_LOSS_PAUSE:
                paused = True
                continue

            # Place bet
            stake = min(FLAT_STAKE, bankroll)
            if stake < 1 or bankroll <= 0:
                continue

            profit = stake * (bookie_odds - 1) if would_win else -stake
            bankroll += profit
            if bankroll > peak:
                peak = bankroll

            consec_loss = 0 if would_win else consec_loss + 1
            dd = (peak - bankroll) / peak * 100
            if dd > max_dd:
                max_dd = dd

            trades.append({
                'date': str(row['match_date'])[:10],
                'outcome': outcome,
                'bookie': bookie_name,
                'back_odds': bookie_odds,
                'consensus': consensus[outcome],
                'edge': round(edge, 1),
                'stake': round(stake, 2),
                'won': would_win,
                'profit': round(profit, 2),
                'bankroll': round(bankroll, 2),
            })
            bet_placed = True
            break

    if (idx + 1) % 5000 == 0:
        n = len(trades)
        elapsed = time.time() - t0
        print(f"  ... {idx+1:,}/{len(df):,} matches, {n} bets, "
              f"equity=£{bankroll:,.0f}, {elapsed:.0f}s")

# ── Report ──
trades_df = pd.DataFrame(trades)
n = len(trades_df)
wins = trades_df['won'].sum() if n > 0 else 0
total_staked = trades_df['stake'].sum() if n > 0 else 0
pnl = bankroll - BANKROLL
roi = pnl / total_staked * 100 if total_staked > 0 else 0

print(f"\n{SEP}")
print("RESULTS — PRE-MATCH ODDS (t=71)")
print(SEP)
print(f"  Matches scanned: {len(df):,}")
print(f"  Skipped (<2 bks): {skipped_sparse:,}")
print(f"  Bets placed:      {n:,}")
if n > 0:
    print(f"  Win rate:         {wins/n*100:.1f}%")
    print(f"  Avg odds:         {trades_df['back_odds'].mean():.2f}")
    print(f"  Avg edge:         {trades_df['edge'].mean():.1f}%")
    print(f"  Total staked:     £{total_staked:,.0f}")
    print(f"  Avg stake:        £{total_staked/n:.2f}")
print(f"  Starting:         £{BANKROLL:,.0f}")
print(f"  Final:            £{bankroll:,.0f}")
print(f"  P&L:              £{pnl:+,.0f}")
print(f"  Return:           {(bankroll/BANKROLL-1)*100:+.1f}%")
if total_staked > 0:
    print(f"  ROI on staked:    {roi:.2f}%")
print(f"  Max DD:           {max_dd:.1f}%")
print()
print("  Filters:")
print(f"    Pre-match odds (t=71)")
print(f"    Edge >= {MIN_EDGE}%")
print(f"    Odds {MIN_ODDS}-{MAX_ODDS}")
print(f"    Flat stake: £{FLAT_STAKE}")
print(f"    Pause after {CONSEC_LOSS_PAUSE} losses: resume on next win")

trades_df.to_csv(OUT / 'ts_consensus.csv', index=False)
print(f"\n  Trade log: {OUT}/ts_consensus.csv")
print(f"  Runtime:   {time.time()-t0:.0f}s")
print(SEP)
