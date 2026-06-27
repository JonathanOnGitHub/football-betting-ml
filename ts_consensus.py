#!/usr/bin/env python3
"""Consensus pricing on time-series data — clean implementation."""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings, time
warnings.filterwarnings('ignore')
t0 = time.time()

SEP = "=" * 60
DATA = Path.home() / '.cache/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2'
OUT = Path('/home/burley/football-ml')

BANKROLL = 1000
STAKE_FRAC = 0.02
MIN_EDGE = 5.0
MIN_ODDS = 1.3
MAX_ODDS = 3.0

print(SEP)
print("CONSENSUS PRICING — TIME-SERIES (full scan)")
print(SEP)

# Load only opening (t=71) and closing (t=0) odds for all 8 bookmakers
basic = ['match_id','match_date','score_home','score_away']
cols = basic[:]
for outcome in ['home','draw','away']:
    for t in [71, 0]:
        for b in range(1, 9):
            cols.append(f'{outcome}_b{b}_{t}')

print("\nLoading...")
df = pd.read_csv(DATA / 'odds_series.csv.gz', compression='gzip',
                 encoding='latin1', usecols=cols)
df = df.dropna(subset=['score_home','score_away'])
print(f"  {len(df):,} completed matches")

# Process each match
bankroll = BANKROLL
peak = BANKROLL
trades = []
dd_info = {'max': 0}

for idx, row in df.iterrows():
    home_s = float(row['score_home'])
    away_s = float(row['score_away'])
    
    # For each time point (71 = opening, 0 = closing)
    for label, t in [('open', 71)]:  # OPENING ONLY — closing has no edge
        # Collect all bookmaker odds at this time point
        bk_data = {'home': [], 'draw': [], 'away': []}
        for outcome in ['home','draw','away']:
            for b in range(1, 9):
                val = row.get(f'{outcome}_b{b}_{t}')
                if pd.notna(val) and val > 0:
                    bk_data[outcome].append((val, f'b{b}'))
        
        # Need enough bookmakers for a meaningful consensus
        if any(len(v) < 2 for v in bk_data.values()):
            continue
        
        # Consensus = median (robust)
        consensus = {}
        for outcome in ['home','draw','away']:
            odds_vals = [v[0] for v in bk_data[outcome]]
            consensus[outcome] = np.median(odds_vals)
        
        # Check each outcome for value
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
                
                # Place bet
                stake = bankroll * STAKE_FRAC
                if stake < 1:
                    continue
                
                if outcome == 'home':
                    won = home_s > away_s
                elif outcome == 'draw':
                    won = home_s == away_s
                else:
                    won = away_s > home_s
                
                profit = stake * (bookie_odds - 1) if won else -stake
                bankroll += profit
                if bankroll > peak:
                    peak = bankroll
                
                dd = (peak - bankroll) / peak * 100
                dd_info['max'] = max(dd_info['max'], dd)
                
                trades.append({
                    'date': str(row['match_date'])[:10],
                    'time_point': label,
                    'outcome': outcome,
                    'bookie': bookie_name,
                    'back_odds': bookie_odds,
                    'consensus': consensus[outcome],
                    'edge': round(edge, 1),
                    'stake': round(stake, 2),
                    'won': won,
                    'profit': round(profit, 2),
                    'bankroll': round(bankroll, 2),
                })
                bet_placed = True
                break  # one bet per match per time point
    
    # Progress
    if (idx + 1) % 5000 == 0:
        n = len(trades)
        elapsed = time.time() - t0
        print(f"  ... {idx+1:,}/{len(df):,} matches, {n} bets, "
              f"equity=£{bankroll:,.0f}, {elapsed:.0f}s")

# Report
trades_df = pd.DataFrame(trades)
n = len(trades_df)
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"  Matches scanned: {len(df):,}")
print(f"  Bets placed:     {n:,}")
if n > 0:
    wins = trades_df['won'].sum()
    total_staked = trades_df['stake'].sum()
    print(f"  Win rate:        {wins/n*100:.1f}%")
    print(f"  Avg odds:        {trades_df['back_odds'].mean():.2f}")
    print(f"  Avg edge:        {trades_df['edge'].mean():.1f}%")
    print(f"  Total staked:    £{total_staked:,.0f}")
    print(f"  Starting:        £{BANKROLL:,.0f}")
    print(f"  Final:           £{bankroll:,.0f}")
    print(f"  P&L:             £{bankroll-BANKROLL:+,.0f}")
    print(f"  Return:          {(bankroll/BANKROLL-1)*100:+.1f}%")
    print(f"  ROI on staked:   {(bankroll-BANKROLL)/total_staked*100:.2f}%" if total_staked > 0 else "")
    print(f"  Max DD:          {dd_info['max']:.1f}%")

# By time point
if n > 0:
    print(f"\n  By time point:")
    for tp in ['open', 'close']:
        sub = trades_df[trades_df['time_point'] == tp]
        if len(sub) > 0:
            wr = sub['won'].mean() * 100
            pnl = sub['profit'].sum()
            print(f"    {tp:6s}: {len(sub):>4,} bets, WR {wr:.1f}%, P&L £{pnl:+,.0f}")

trades_df.to_csv(OUT / 'ts_consensus.csv', index=False)
print(f"\n  Runtime: {time.time()-t0:.0f}s")
print(SEP)
