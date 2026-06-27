#!/usr/bin/env python3
"""
Opening Odds Consensus Strategy — Beat The Bookie (corrected)
==============================================================
The published strategy says: at OPENING time, consensus is a better
estimator of true probability than any single bookmaker's price.
Bookmakers make errors when lines first open. By closing time, the
market has corrected those errors.

This version uses odds_series.csv.gz which has:
  - 72 timestamps per match (0 = opening, 71 = closing)
  - Up to 7 bookmakers (b1-b7)
  - home_bX_N, draw_bX_N, away_bX_N columns for each timestamp N

Strategy:
  1. For each match, collect opening odds (timestamp 0) from all bookmakers
  2. Compute consensus = average opening odds
  3. If any bookmaker's opening odds are 5%+ above consensus, back it
  4. Track result using actual score
  5. £1k start, 2% per bet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import warnings, time, json, gzip
warnings.filterwarnings('ignore')

t0 = time.time()
SEP = "=" * 60
DATA = Path.home() / '.cache/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2'
OUT = Path('/home/burley/football-ml')
OUT.mkdir(exist_ok=True)

# ── Config ──
BANKROLL = 1000
STAKE_FRAC = 0.02
MIN_EDGE_PCT = 5.0
MIN_ODDS = 1.3
MAX_ODDS = 2.0
TIMESTAMP_OPEN = 0    # Opening odds
TIMESTAMP_CLOSE = 71  # Closing odds (for reference)

@dataclass
class Trade:
    match_date: str
    league: str
    home: str
    away: str
    outcome: str
    back_odds: float
    back_bookie: str
    consensus_odds: float
    edge_pct: float
    n_bookmakers: int
    stake: float
    won: bool
    profit: float

@dataclass
class Portfolio:
    bankroll: float
    trades: list = field(default_factory=list)
    peak: float = 0.0
    
    def __post_init__(self):
        self.peak = self.bankroll
    
    @property
    def equity(self):
        return self.bankroll + sum(t.profit for t in self.trades)
    
    @property
    def total_pnl(self):
        return sum(t.profit for t in self.trades)
    
    @property
    def drawdown_pct(self):
        eq = self.equity
        if eq > self.peak:
            self.peak = eq
        if self.peak == 0:
            return 0
        return (self.peak - eq) / self.peak * 100
    
    def bet(self, trade: Trade):
        if trade.won:
            trade.profit = trade.stake * (trade.back_odds - 1)
        else:
            trade.profit = -trade.stake
        self.trades.append(trade)
        if self.equity > self.peak:
            self.peak = self.equity
    
    def summary(self):
        wins = sum(1 for t in self.trades if t.won)
        total_staked = sum(t.stake for t in self.trades)
        
        months = {}
        for t in self.trades:
            m = t.match_date[:7]
            if m not in months:
                months[m] = {'bets': 0, 'wins': 0, 'pnl': 0}
            months[m]['bets'] += 1
            months[m]['wins'] += 1 if t.won else 0
            months[m]['pnl'] += t.profit
        
        print(f"\n{SEP}")
        print("OPENING ODDS CONSENSUS — PORTFOLIO SUMMARY")
        print(SEP)
        print(f"  Starting bankroll:  £{self.bankroll:,.2f}")
        print(f"  Current equity:     £{self.equity:,.2f}")
        print(f"  Total P&L:          £{self.total_pnl:+,.2f}")
        print(f"  Return:             {(self.equity/self.bankroll-1)*100:+.2f}%")
        print(f"  Total bets:         {len(self.trades):,}")
        print(f"  Win rate:           {wins/len(self.trades)*100:.1f}%" if self.trades else "  Win rate:           N/A")
        print(f"  Total staked:       £{total_staked:,.2f}")
        print(f"  ROI on staked:      {self.total_pnl/total_staked*100:.2f}%" if total_staked > 0 else "")
        print(f"  Max drawdown:       {self.drawdown_pct:.1f}%")
        print(f"  Months active:      {len(months)}")
        
        if months:
            avg_month = sum(m['pnl'] for m in months.values()) / len(months)
            print(f"  Avg monthly P&L:    £{avg_month:+.2f}")
            best_m = max(months.items(), key=lambda x: x[1]['pnl'])
            worst_m = min(months.items(), key=lambda x: x[1]['pnl'])
            print(f"  Best month:         {best_m[0]} (£{best_m[1]['pnl']:+,.0f})")
            print(f"  Worst month:        {worst_m[0]} (£{worst_m[1]['pnl']:+,.0f})")
        
        # Edge analysis
        if self.trades:
            print(f"\n  Edge analysis:")
            for threshold in [5, 10, 15, 20]:
                sub = [t for t in self.trades if t.edge_pct >= threshold]
                if sub:
                    wr = sum(1 for t in sub if t.won) / len(sub) * 100
                    print(f"    Edge >= {threshold:2d}%: {len(sub):>4,} bets, WR {wr:.1f}%")
        
        # Odds bands
        if self.trades:
            print(f"\n  Odds band analysis:")
            for lo, hi in [(1.3, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0)]:
                sub = [t for t in self.trades if lo <= t.back_odds < hi]
                if sub:
                    wr = sum(1 for t in sub if t.won) / len(sub) * 100
                    pnl = sum(t.profit for t in sub)
                    print(f"    {lo:.0f}-{hi:.0f}: {len(sub):>4,} bets, WR {wr:.1f}%, P&L £{pnl:+,.0f}")
        
        # Top leagues
        if self.trades:
            leagues = {}
            for t in self.trades:
                if t.league not in leagues:
                    leagues[t.league] = {'bets': 0, 'wins': 0, 'pnl': 0}
                leagues[t.league]['bets'] += 1
                leagues[t.league]['wins'] += 1 if t.won else 0
                leagues[t.league]['pnl'] += t.profit
            print(f"\n  Top 10 leagues by P&L:")
            for league, data in sorted(leagues.items(), key=lambda x: x[1]['pnl'], reverse=True)[:10]:
                wr = data['wins']/data['bets']*100
                print(f"    {league[:35]:35s} {data['bets']:>4,} bets, WR {wr:.1f}%, £{data['pnl']:+,.0f}")
            
            # Bottom 5 leagues by P&L
            print(f"\n  Bottom 5 leagues by P&L:")
            for league, data in sorted(leagues.items(), key=lambda x: x[1]['pnl'])[:5]:
                wr = data['wins']/data['bets']*100
                print(f"    {league[:35]:35s} {data['bets']:>4,} bets, WR {wr:.1f}%, £{data['pnl']:+,.0f}")


def extract_opening_odds(row, ts=0):
    """
    Extract opening odds from all bookmakers for a match row.
    Returns dict of {outcome: {bookie: odds}} only for non-NaN values.
    Also returns score_home, score_away.
    """
    odds = {'home': {}, 'draw': {}, 'away': {}}
    t = ts
    
    for bk in range(1, 8):  # bookmakers b1-b7
        for outcome_key, outcome in [('home', 'home'), ('draw', 'draw'), ('away', 'away')]:
            col = f'{outcome_key}_b{bk}_{t}'
            val = row.get(col)
            if pd.notna(val) and val > 0:
                odds[outcome][bk] = float(val)
    
    score_h = row.get('score_home', None)
    score_a = row.get('score_away', None)
    
    return odds, score_h, score_a


def run_opening_odds_simulation():
    """Run simulation using opening odds from odds_series data."""
    print(SEP)
    print("OPENING ODDS CONSENSUS STRATEGY — Simulation")
    print(SEP)
    print(f"  Data: odds_series.csv.gz + odds_series_matches.csv.gz")
    print(f"  Bankroll: £{BANKROLL:,.0f}")
    print(f"  Stake: {STAKE_FRAC*100:.0f}% per bet")
    print(f"  Min edge: {MIN_EDGE_PCT}% over consensus")
    print(f"  Odds range: {MIN_ODDS}-{MAX_ODDS}")
    print(f"  Using opening odds (timestamp {TIMESTAMP_OPEN})")
    
    # ── Load matches metadata ──
    print("\n[1] Loading match metadata...")
    matches = pd.read_csv(DATA / 'odds_series_matches.csv.gz', compression='gzip',
                          encoding='latin1')
    matches.columns = [c.strip() for c in matches.columns]  # Some cols have leading spaces
    print(f"  {len(matches):,} matches loaded")
    
    # Parse score column: format "3:1"
    def parse_score(s):
        try:
            parts = str(s).split(':')
            return int(parts[0]), int(parts[1])
        except:
            return None, None
    
    scores = matches['score'].apply(parse_score)
    matches['home_score'] = scores.apply(lambda x: x[0])
    matches['away_score'] = scores.apply(lambda x: x[1])
    matches = matches.dropna(subset=['home_score', 'away_score'])
    print(f"  {len(matches):,} matches with valid scores")
    
    # Build match_id lookup
    match_lookup = {}
    for _, row in matches.iterrows():
        match_lookup[row['match_id']] = row
    
    # ── Load odds series ──
    print("\n[2] Loading odds series (opening odds, timestamp 0)...")
    # We only need opening odds columns
    bk_cols = []
    for bk in range(1, 8):
        for outcome in ['home', 'draw', 'away']:
            bk_cols.append(f'{outcome}_b{bk}_{TIMESTAMP_OPEN}')
    
    use_cols = ['match_id', 'match_date', 'match_time', 'score_home', 'score_away'] + bk_cols
    series = pd.read_csv(DATA / 'odds_series.csv.gz', compression='gzip',
                         encoding='latin1', usecols=use_cols)
    print(f"  {len(series):,} matches with opening odds")
    
    # Merge with metadata
    print("\n[3] Merging with match metadata...")
    matches_dict = matches.set_index('match_id').to_dict('index')
    
    # ── Scan for value bets ──
    print(f"\n[4] Scanning for opening-odds value bets...")
    strategy_name = "opening_odds_consensus"
    portfolio = Portfolio(bankroll=BANKROLL)
    skipped_no_edge = 0
    skipped_odds_range = 0
    skipped_not_in_meta = 0
    skipped_no_bookies = 0
    
    t1 = time.time()
    bets_placed = 0
    
    for idx, row in series.iterrows():
        mid = row['match_id']
        
        # Get match metadata
        meta = matches_dict.get(mid)
        if meta is None:
            skipped_not_in_meta += 1
            continue
        
        league = str(meta.get(' league', meta.get('league', ''))).strip()
        home_team = str(meta.get(' home_team', meta.get('home_team', ''))).strip()
        away_team = str(meta.get(' away_team', meta.get('away_team', ''))).strip()
        home_score = meta.get('home_score', meta.get('score_home'))
        away_score = meta.get('away_score', meta.get('score_away'))
        match_date = str(row.get('match_date', ''))
        
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        
        home_score = int(home_score)
        away_score = int(away_score)
        
        # Extract opening odds from all bookmakers
        bk_odds_row = {}
        for bk in range(1, 8):
            for outcome_key, outcome in [('home', 'home'), ('draw', 'draw'), ('away', 'away')]:
                col = f'{outcome_key}_b{bk}_{TIMESTAMP_OPEN}'
                val = row.get(col)
                if pd.notna(val) and float(val) > 0:
                    if outcome not in bk_odds_row:
                        bk_odds_row[outcome] = {}
                    bk_odds_row[outcome][bk] = float(val)
        
        # Check each outcome for value
        for outcome, outcome_key in [('home', 'home'), ('draw', 'draw'), ('away', 'away')]:
            oodds = bk_odds_row.get(outcome, {})
            if len(oodds) < 3:  # Need at least 3 bookmakers for meaningful consensus
                skipped_no_bookies += 1
                continue
            
            # Compute consensus = mean of opening odds
            consensus_odds = np.mean(list(oodds.values()))
            
            # Best odds among bookmakers
            max_odds = max(oodds.values())
            best_bookie_idx = max(oodds, key=oodds.get)
            
            # Must beat consensus
            if max_odds <= consensus_odds:
                skipped_no_edge += 1
                continue
            
            # Odds range filter
            if not (MIN_ODDS <= max_odds <= MAX_ODDS):
                skipped_odds_range += 1
                continue
            
            # Edge calculation
            edge_pct = (max_odds / consensus_odds - 1) * 100
            if edge_pct < MIN_EDGE_PCT:
                skipped_no_edge += 1
                continue
            
            # Determine result
            if outcome == 'home':
                won = home_score > away_score
            elif outcome == 'draw':
                won = home_score == away_score
            else:
                won = away_score > home_score
            
            # Place bet
            stake = portfolio.equity * STAKE_FRAC
            if stake < 1:
                continue
            
            trade = Trade(
                match_date=str(match_date)[:10],
                league=league,
                home=home_team,
                away=away_team,
                outcome=outcome,
                back_odds=max_odds,
                back_bookie=f'b{best_bookie_idx}',
                consensus_odds=consensus_odds,
                edge_pct=edge_pct,
                n_bookmakers=len(oodds),
                stake=stake,
                won=won,
                profit=0.0,
            )
            portfolio.bet(trade)
            bets_placed += 1
        
        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - t1
            print(f"  ... {idx+1:,}/{len(series):,} matches, "
                  f"{bets_placed} bets, "
                  f"[{elapsed:.0f}s]")
    
    elapsed = time.time() - t1
    print(f"\n  Scan complete: {len(series):,} matches in {elapsed:.0f}s")
    print(f"  Bets placed: {bets_placed}")
    print(f"  Skipped: {skipped_not_in_meta} no metadata, "
          f"{skipped_no_bookies} <3 bookmakers, "
          f"{skipped_odds_range} out of range, "
          f"{skipped_no_edge} no edge")
    
    portfolio.summary()
    
    # Save trades
    trades_df = pd.DataFrame([{
        'date': t.match_date,
        'league': t.league,
        'home': t.home,
        'away': t.away,
        'outcome': t.outcome,
        'back_odds': t.back_odds,
        'bookie': t.back_bookie,
        'consensus': round(t.consensus_odds, 2),
        'edge_pct': round(t.edge_pct, 1),
        'n_bookies': t.n_bookmakers,
        'stake': round(t.stake, 2),
        'won': t.won,
        'profit': round(t.profit, 2),
    } for t in portfolio.trades])
    
    trades_df.to_csv(OUT / 'opening_odds_trades.csv', index=False)
    print(f"\n  Trade log saved to {OUT}/opening_odds_trades.csv")
    print(f"  Runtime: {time.time()-t0:.0f}s")
    print(SEP)
    return portfolio


if __name__ == '__main__':
    run_opening_odds_simulation()
