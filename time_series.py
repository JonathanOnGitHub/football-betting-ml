#!/usr/bin/env python3
"""
Time-series odds analysis — Beat The Bookie minute-to-minute data.
31K matches, 32 bookmakers, 72 time points, Sep 2015 - Feb 2016.

Strategies tested:
  1. Opening → Closing: Back at best opening odds, lay at best closing
  2. Inter-bookmaker arb: At one time point, back all 3 outcomes at best odds
  3. Best price betting: Always take the best available odds
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings, time
warnings.filterwarnings('ignore')
t0 = time.time()

SEP = "=" * 60
DATA = Path.home() / '.cache/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2'
OUT = Path('/home/burley/football-ml')
OUT.mkdir(exist_ok=True)

print(SEP)
print("TIME-SERIES ODDS ANALYSIS — 32 BOOKMAKERS × 72 TIME POINTS")
print(SEP)

# ── 1. Load only what we need ──
print("\n[1] Loading time-series odds...")

# Define columns to load: basic cols + opening (t=71) and closing (t=0) odds for all bookies
basic_cols = ['match_id','match_date','match_time','score_home','score_away']
time_cols = []
for outcome in ['home','draw','away']:
    for t in [0, 71]:  # closing = 0, opening = 71
        for b in range(1, 33):
            time_cols.append(f'{outcome}_b{b}_{t}')

all_cols = basic_cols + time_cols
print(f"  Loading {len(all_cols)} columns from {len(time_cols)} time-series odds...")

df = pd.read_csv(DATA / 'odds_series.csv.gz', compression='gzip',
                 encoding='latin1', usecols=all_cols)
print(f"  Rows: {len(df):,} matches")
print(f"  Date: {df['match_date'].min()} to {df['match_date'].max()}")

# Parse scores
df[['score_home','score_away']] = df[['score_home','score_away']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['score_home','score_away'])
df['home_win'] = (df['score_home'] > df['score_away']).astype(int)
df['draw'] = (df['score_home'] == df['score_away']).astype(int)
df['away_win'] = (df['score_home'] < df['score_away']).astype(int)

print(f"  Usable: {len(df):,} matches, home win rate: {df['home_win'].mean()*100:.1f}%")

# ── 2. Compute best opening and closing odds across all bookmakers ──
print("\n[2] Computing best prices across 32 bookmakers...")

# Opening (t=71) and closing (t=0) best odds
for label, t in [('open', 71), ('close', 0)]:
    for outcome in ['home','draw','away']:
        cols = [f'{outcome}_b{b}_{t}' for b in range(1, 33)]
        df[f'best_{outcome}_{label}'] = df[cols].max(axis=1, skipna=True)

    # Overround analysis
    for impl, name in [('best', 'best')]:
        over = (1/df[f'{name}_home_{label}'] + 1/df[f'{name}_draw_{label}'] + 1/df[f'{name}_away_{label}'])
        df[f'overround_{label}'] = over - 1
    
    print(f"  {label}: avg best overround = {df[f'overround_{label}'].mean()*100:.2f}%")

# ── 3. STRATEGY 1: Opening-to-Closing on same outcome ──
print(f"\n{SEP}")
print("STRATEGY 1: Back at opening → Lay at closing (same outcome)")
print("Back if opening odds > closing odds = guaranteed profit")
print(SEP)

for outcome, label in [('home', 'Home'), ('draw', 'Draw'), ('away', 'Away')]:
    col_open = f'best_{outcome}_open'
    col_close = f'best_{outcome}_close'
    
    trades = df[(df[col_open].notna()) & (df[col_close].notna()) & 
                (df[col_open] > df[col_close])].copy()
    n = len(trades)
    if n == 0:
        print(f"  {label}: No qualifying trades")
        continue
    
    trades['ratio'] = trades[col_open] / trades[col_close]
    trades['return_pct'] = (trades['ratio'] - 1) * 100
    
    # Green-book profit: back £10 at opening odds, lay at closing
    trades['profit_10'] = 10 * (trades['ratio'] - 1)
    total_profit = trades['profit_10'].sum()
    
    print(f"  {label:6s}: {n:>5,} trades ({n/len(df)*100:.1f}% of matches)")
    print(f"          Median swing: {trades['ratio'].median():.2f}x ({trades['return_pct'].median():.1f}%)")
    print(f"          Back £10 each → £{total_profit:,.0f} gross profit")
    print(f"          Less 5% comm:    £{total_profit*0.95:,.0f}")
    print(f"          ROI:             {total_profit/(n*10)*100:.1f}%")

# ── 4. STRATEGY 2: Cross-outcome arb at closing ──
print(f"\n{SEP}")
print("STRATEGY 2: Match arb at closing (back all 3 outcomes)")
print("Check if sum(1/best_odds) < 1.0 = guaranteed profit")
print(SEP)

for label, t in [('Open (t=71)', 71), ('Close (t=0)', 0)]:
    df[f'implied_close'] = 1/df[f'best_home_close'] + 1/df[f'best_draw_close'] + 1/df[f'best_away_close']
    
    arb = df[df[f'implied_close'] < 1.0].copy()
    n = len(arb)
    print(f"  {label}: {n:,} arbs ({n/len(df)*100:.2f}% of matches)")
    
    if n > 0:
        # If implied < 1.0, we can back all three outcomes proportionally
        arb['arb_return'] = (1.0 / arb['implied_close'] - 1) * 100
        arb['stake_home'] = 100 / arb['best_home_close'] / arb['implied_close']
        arb['stake_draw'] = 100 / arb['best_draw_close'] / arb['implied_close']
        arb['stake_away'] = 100 / arb['best_away_close'] / arb['implied_close']
        
        print(f"     Avg arb return: {arb['arb_return'].mean():.2f}%")
        print(f"     Max arb return: {arb['arb_return'].max():.2f}%")
        print(f"     Total staked (per £100): £{n*100:,.0f}")
        print(f"     Guaranteed profit: £{arb['arb_return'].sum():,.0f}")

# ── 5. STRATEGY 3: Opening arb (back at opening best odds) ──
print(f"\n{SEP}")
print("STRATEGY 3: Opening arb (back all 3 at opening best odds)")
print(SEP)

df['implied_open'] = 1/df['best_home_open'] + 1/df['best_draw_open'] + 1/df['best_away_open']
open_arb = df[df['implied_open'] < 1.0]
n = len(open_arb)
print(f"  {n:,} opening arbs ({n/len(df)*100:.2f}%)")
if n > 0:
    print(f"  Avg return: {(1.0/open_arb['implied_open'] - 1).mean()*100:.2f}%")

# ── 6. STRATEGY 4: Multi-time arb chain ──
print(f"\n{SEP}")
print("STRATEGY 4: Opening arb → unlock at closing")
print("Back arb at opening odds, lay at closing odds for each outcome")
print(SEP)

trades = df[df['implied_open'] < 1.0].copy()
n = len(trades)
if n > 0:
    # For each match where opening odds have an arb:
    # Back all 3 outcomes at opening best odds for guaranteed profit
    # Then at closing, we can lock in by laying each outcome
    
    # Simple: back the arb at opening = guaranteed £X
    trades['arb_pct'] = (1.0 / trades['implied_open'] - 1) * 100
    
    print(f"  {n:,} opening arb opportunities")
    print(f"  Avg arb: {trades['arb_pct'].mean():.2f}%")
    print(f"  Total: back £100 each → £{trades['arb_pct'].sum():.0f} guaranteed")
    print(f"  Less 5% comm: £{trades['arb_pct'].sum()*0.95:.0f}")

# ── 7. STRATEGY 5: Best-price-only betting (no arb) ──
print(f"\n{SEP}")
print("STRATEGY 5: Best-price home win betting")
print("Bet on home wins only when using best closing odds > avg odds")
print(SEP)

# Compare best odds to the average odds used in closing_odds.csv
# We don't have avg_odds here, so let's compare best vs worst bookie

# Best = max odds, worst = min odds
for outcome in ['home','draw','away']:
    best_close = [f'{outcome}_b{b}_0' for b in range(1,33)]
    df[f'best_{outcome}_c'] = df[best_close].max(axis=1)
    df[f'worst_{outcome}_c'] = df[best_close].min(axis=1)
    df[f'median_{outcome}_c'] = df[best_close].median(axis=1)
    df[f'spread_{outcome}'] = (df[f'best_{outcome}_c'] / df[f'worst_{outcome}_c'] - 1) * 100

print("  Closing odds spread (best/worst ratio):")
for outcome in ['home','draw','away']:
    print(f"    {outcome:6s}: median spread {df[f'spread_{outcome}'].median():.1f}%  "
          f"mean {df[f'spread_{outcome}'].mean():.1f}%")

# Backtest: bet home win at best closing odds, only where value
trades = df[df['best_home_close'].notna() & df['best_draw_close'].notna() & 
            df['best_away_close'].notna()].copy()

# Market implied from best odds
trades['mkt_home'] = (1/trades['best_home_close']) / (
    1/trades['best_home_close'] + 1/trades['best_draw_close'] + 1/trades['best_away_close'])
trades['edge'] = trades['mkt_home'] - (
    1/trades['best_home_close'])  # market-relative edge

# Simple: bet on all home wins at best price
BANKROLL = 10000
STAKE = 0.02
bankroll = BANKROLL
wins = 0
bets = 0
for _, row in trades.iterrows():
    # Only bet if best_home odds are decent
    if row['best_home_close'] < 1.3 or row['best_home_close'] > 10:
        continue
    stake = bankroll * STAKE
    bets += 1
    if row['home_win']:
        bankroll += stake * (row['best_home_close'] - 1) * 0.95
        wins += 1
    else:
        bankroll -= stake

ret = (bankroll / BANKROLL - 1) * 100
print(f"\n  Best-price home win betting (all matches, odds 1.3-10):")
print(f"    Bets: {bets:,} | Win: {wins/bets*100:.1f}%")
print(f"    £{BANKROLL:,.0f} → £{bankroll:,.0f} ({ret:+.0f}%)")

# Same but using worst odds (what a casual bettor gets)
bankroll2 = BANKROLL
wins2 = 0
bets2 = 0
for _, row in trades.iterrows():
    wc = row['worst_home_c']
    if pd.isna(wc) or wc < 1.3 or wc > 10:
        continue
    stake = bankroll2 * STAKE
    bets2 += 1
    if row['home_win']:
        bankroll2 += stake * (wc - 1) * 0.95
        wins2 += 1
    else:
        bankroll2 -= stake

ret2 = (bankroll2 / BANKROLL - 1) * 100
print(f"\n  Worst-price home win betting (same matches, worst bookie):")
print(f"    Bets: {bets2:,} | Win: {wins2/bets2*100:.1f}%")
print(f"    £{BANKROLL:,.0f} → £{bankroll2:,.0f} ({ret2:+.0f}%)")

# ── SUMMARY ──
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"\n  Data: {len(df):,} matches, 32 bookmakers, 72 time points")
print(f"  Period: Sep 2015 - Feb 2016")
print(f"  Runtime: {time.time()-t0:.1f}s")
print(f"\n  Best-pricing advantage: avg {(df['best_home_close']/df['worst_home_c']).mean():.2f}x")
print(f"  between best and worst bookmaker")
print(SEP)
