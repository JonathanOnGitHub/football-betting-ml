#!/usr/bin/env python3
"""
Betfair Exchange Scalping — Simulation
========================================
Simulates scalping the back/lay spread across 72 time points.
 
Strategy:
  1. At time T, BACK at the current price
  2. At time T+1 (next snapshot), LAY at the current price  
  3. If price moved in our favour (lay > back), profit on the green book
  4. If price moved against us, we take the loss

Key assumption: Betfair spread ≈ 2% (back price × 1.02 ≈ lay price)
Liquid markets have tighter spreads (0.5-1%), illiquid wider (3-5%).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
import warnings, time
warnings.filterwarnings('ignore')
t0 = time.time()

SEP = "=" * 60
DATA = Path.home() / '.cache/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2'
OUT = Path('/home/burley/football-ml')
OUT.mkdir(exist_ok=True)

BANKROLL = 10000
STAKE_PER_TRADE = 100   # £100 scalp size
BF_SPREAD = 0.02        # 2% back/lay spread (conservative for liquid markets)

print(SEP)
print("BETFAIR EXCHANGE SCALPING — SIMULATION")
print(SEP)
print(f"  Bankroll: £{BANKROLL:,.0f}")
print(f"  Stake per scalp: £{STAKE_PER_TRADE}")
print(f"  Betfair spread:  {BF_SPREAD*100:.0f}% (back price × {1+BF_SPREAD:.2f} ≈ lay price)")
print(f"  Data: 72 time points across 8 bookmaker prices per time point")

# ═══════════════════════════════════════════════════════════════════
# Load time-series data (all 72 time points)
# ═══════════════════════════════════════════════════════════════════
print("\n[1] Loading time-series data...")

# We need basic cols + all 72 time points for all 8 bookies
basic = ['match_id','match_date','match_time','score_home','score_away']
cols = basic[:]
for outcome in ['home','draw','away']:
    for t in range(72):
        for b in range(1, 5):  # First 4 bookmakers (liquidity proxy)
            cols.append(f'{outcome}_b{b}_{t}')

df = pd.read_csv(DATA / 'odds_series.csv.gz', compression='gzip',
                 encoding='latin1', usecols=cols)
df = df.dropna(subset=['score_home','score_away'])
print(f"  {len(df):,} completed matches, {len(cols)} columns")

# ═══════════════════════════════════════════════════════════════════
# SCALPING ENGINE
# ═══════════════════════════════════════════════════════════════════

def simulate_scalping(row, spread=BF_SPREAD, stake=STAKE_PER_TRADE):
    """
    Simulate scalping on a single match across 72 time points.
    
    For each consecutive pair of time points (t, t+1):
      1. Get consensus price at time t (average of available bookmakers)
      2. Get consensus price at time t+1
      3. Simulated lay price at t+1 = back_price_{t+1} × (1 + spread)
      4. Back at price_t. If lay at price_{t+1} × (1+spread) > price_t, scalp profit.
      5. Otherwise, we have a losing scalp.
    
    Returns list of scalp results.
    """
    results = []
    
    for t in range(71):  # 71 consecutive pairs
        # Get available prices at time t and t+1
        prices_t = []
        prices_t1 = []
        for b in range(1, 5):
            for outcome in ['home','draw','away']:
                v_t = row.get(f'{outcome}_b{b}_{t}')
                v_t1 = row.get(f'{outcome}_b{b}_{t+1}')
                if pd.notna(v_t) and v_t > 0:
                    prices_t.append(v_t)
                if pd.notna(v_t1) and v_t1 > 0:
                    prices_t1.append(v_t1)
        
        if len(prices_t) < 3 or len(prices_t1) < 3:
            continue
        
        # Consensus price at each time point (median)
        price_t = np.median(prices_t)
        price_t1 = np.median(prices_t1)
        
        if not (1.3 <= price_t <= 50) or not (1.3 <= price_t1 <= 50):
            continue
        
        # Simulate: BACK at price_t, LAY at price_t1 × (1+spread)
        back_price = price_t
        lay_price = price_t1 * (1 + spread)  # Simulated lay price
        
        # Green-book calculation
        # Back £X at back_price, lay £Y at lay_price
        # For equal profit: Y = X * back_price / lay_price
        if lay_price <= back_price:
            # Price moved in our favour
            lay_stake = stake * back_price / lay_price
            profit_if_wins = stake * (back_price - 1) - lay_stake * (lay_price - 1)
            profit_if_loses = -stake + lay_stake
            profit = min(profit_if_wins, profit_if_loses) * 0.95  # comm
            won = True
        else:
            # Price moved against — we take the loss
            lay_stake = stake * back_price / lay_price
            profit_if_wins = stake * (back_price - 1) - lay_stake * (lay_price - 1)
            profit_if_loses = -stake + lay_stake
            profit = min(profit_if_wins, profit_if_loses) * 0.95
            won = False
        
        results.append({
            't': t,
            'back_price': round(back_price, 3),
            'lay_price': round(lay_price, 3),
            'move_pct': round((price_t1 / price_t - 1) * 100, 2),
            'profit': round(profit, 2),
            'won': won,
        })
    
    return results


# ═══════════════════════════════════════════════════════════════════
# RUN SIMULATION
# ═══════════════════════════════════════════════════════════════════

print("\n[2] Running scalping simulation...")
t1 = time.time()

all_scalps = []
match_scalps = 0

for idx, row in df.iterrows():
    scalps = simulate_scalping(row)
    if scalps:
        for s in scalps:
            s['match_date'] = str(row['match_date'])[:10]
            s['match_id'] = int(row['match_id'])
        all_scalps.extend(scalps)
        match_scalps += 1
    
    if (idx + 1) % 5000 == 0:
        n = len(all_scalps)
        wins = sum(1 for s in all_scalps if s['won'])
        profit = sum(s['profit'] for s in all_scalps)
        print(f"  ... {idx+1:,}/{len(df):,} matches, "
              f"{n} scalps, WR {wins/n*100:.1f}%, P&L £{profit:+,.0f} "
              f"[{time.time()-t1:.0f}s]")

# ═══════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(all_scalps)
n = len(results_df)
wins = results_df['won'].sum() if n > 0 else 0
total_pnl = results_df['profit'].sum() if n > 0 else 0
win_rate = wins / n * 100 if n > 0 else 0

print(f"\n{SEP}")
print("RESULTS")
print(SEP)
print(f"  Matches traded:    {match_scalps:,}")
print(f"  Total scalp trades:{n:,}")
print(f"  Win rate:          {win_rate:.1f}%")
print(f"  Avg profit/trade:  £{total_pnl/n:.2f}" if n > 0 else "")
print(f"  Total P&L:         £{total_pnl:+,.0f}")
print(f"  ROI on stake:      {total_pnl/(n*STAKE_PER_TRADE)*100:.2f}%" if n > 0 else "")

if n > 0:
    # Analysis by time point
    print(f"\n  Scalps by time interval:")
    t_stats = results_df.groupby('t').agg(
        count=('profit', 'count'),
        win_rate=('won', 'mean'),
        pnl=('profit', 'sum'),
    )
    t_stats['win_rate'] *= 100
    for t_val, row in t_stats.iterrows():
        if row['count'] >= 500:
            print(f"    t={t_val:2d}→{t_val+1:2d}: {int(row['count']):>5,} scalps, "
                  f"WR {row['win_rate']:.1f}%, P&L £{row['pnl']:+,.0f}")
    
    # Analysis by price movement
    print(f"\n  Scalps by price movement direction:")
    up = results_df[results_df['move_pct'] > 0]
    down = results_df[results_df['move_pct'] < 0]
    flat = results_df[results_df['move_pct'] == 0]
    for label, sub in [('Price ↑', up), ('Price ↓', down), ('Price →', flat)]:
        if len(sub) > 0:
            print(f"    {label:10s}: {len(sub):>6,} scalps, "
                  f"WR {sub['won'].mean()*100:.1f}%, "
                  f"avg move {sub['move_pct'].mean():+.2f}%")
    
    # Analysis by odds band
    print(f"\n  Scalps by odds band:")
    for lo, hi in [(1.3,2),(2,3),(3,5),(5,10),(10,50)]:
        sub = results_df[(results_df['back_price']>=lo)&(results_df['back_price']<hi)]
        if len(sub) > 0:
            print(f"    {lo:.0f}-{hi:.0f}: {len(sub):>6,} scalps, "
                  f"WR {sub['won'].mean()*100:.1f}%, "
                  f"EV £{sub['profit'].mean():+.2f}")

# Save
results_df.to_csv(OUT / 'scalping_results.csv', index=False)
print(f"\n  Saved to {OUT}/scalping_results.csv")
print(f"  Runtime: {time.time()-t0:.0f}s")
print(SEP)
