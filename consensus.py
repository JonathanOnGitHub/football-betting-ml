#!/usr/bin/env python3
"""
Consensus Pricing Strategy — Beat The Bookie (replication)
=========================================================
The published, peer-reviewed approach:

  1. For each match, compute CONCENSUS odds = average across 32 bookmakers
  2. Normalise consensus to remove overround (true probabilities)
  3. If a bookmaker offers odds significantly ABOVE consensus = VALUE
  4. Back that outcome at the best available price
  5. The market clears at the true outcome (consensus is more accurate)
  6. Profit comes from systematic bookmaker pricing errors

Data: closing_odds.csv.gz — 479K matches, 2005-2015
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import warnings, time, json
warnings.filterwarnings('ignore')
t0 = time.time()

SEP = "=" * 60
DATA = Path.home() / '.cache/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2'
OUT = Path('/home/burley/football-ml')
OUT.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────
BANKROLL = 10000
STAKE_FRAC = 0.02        # 2% of bankroll per bet
MIN_EDGE_PCT = 5.0       # Minimum edge over consensus
MIN_ODDS = 1.3           
MAX_ODDS = 2.5            # Tight range — consensus edge only exists for short prices
COMMISSION = 0.00         # No commission — backing with bookmakers directly
TRAIN_END = '2013-12-31'  # Train consensus model on earlier data

# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    match_date: str
    league: str
    home: str
    away: str
    outcome: str          # 'home', 'draw', 'away'
    back_odds: float
    back_bookie: str      # which bookmaker offered the best price
    consensus_odds: float # average across all bookmakers
    edge_pct: float       # how much better this price is than consensus
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
        if self.peak == 0: return 0
        return (self.peak - eq) / self.peak * 100
    
    def bet(self, trade: Trade):
        """Place a bet and record the outcome."""
        if trade.won:
            trade.profit = trade.stake * (trade.back_odds - 1)
        else:
            trade.profit = -trade.stake
        
        self.trades.append(trade)
        
        if self.equity > self.peak:
            self.peak = self.equity
    
    def summary(self):
        """Full portfolio report."""
        wins = sum(1 for t in self.trades if t.won)
        losses = len(self.trades) - wins
        total_staked = sum(t.stake for t in self.trades)
        
        # Monthly breakdown
        months = {}
        for t in self.trades:
            m = t.match_date[:7]
            if m not in months:
                months[m] = {'bets':0, 'wins':0, 'pnl':0}
            months[m]['bets'] += 1
            months[m]['wins'] += 1 if t.won else 0
            months[m]['pnl'] += t.profit
        
        print(f"\n{SEP}")
        print("PORTFOLIO SUMMARY")
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
            edges = [t.edge_pct for t in self.trades]
            print(f"\n  Edge analysis:")
            for threshold in [5, 10, 15, 20]:
                sub = [t for t in self.trades if t.edge_pct >= threshold]
                if sub:
                    wr = sum(1 for t in sub if t.won) / len(sub) * 100
                    print(f"    Edge >= {threshold:2d}%: {len(sub):>4,} bets, WR {wr:.1f}%")
        
        # Odds band analysis
        if self.trades:
            print(f"\n  Odds band analysis:")
            for lo, hi in [(1.3,2.0), (2.0,3.0), (3.0,5.0), (5.0,10.0), (10.0,15.0)]:
                sub = [t for t in self.trades if lo <= t.back_odds < hi]
                if sub:
                    wr = sum(1 for t in sub if t.won) / len(sub) * 100
                    pnl = sum(t.profit for t in sub)
                    print(f"    {lo:.0f}-{hi:.0f}: {len(sub):>4,} bets, WR {wr:.1f}%, P&L £{pnl:+,.0f}")
        
        # Best leagues
        if self.trades:
            leagues = {}
            for t in self.trades:
                if t.league not in leagues:
                    leagues[t.league] = {'bets':0, 'wins':0, 'pnl':0}
                leagues[t.league]['bets'] += 1
                leagues[t.league]['wins'] += 1 if t.won else 0
                leagues[t.league]['pnl'] += t.profit
            
            print(f"\n  Top 10 leagues by P&L:")
            for league, data in sorted(leagues.items(), key=lambda x: x[1]['pnl'], reverse=True)[:10]:
                wr = data['wins']/data['bets']*100
                print(f"    {league[:30]:30s} {data['bets']:>4,} bets, WR {wr:.1f}%, £{data['pnl']:+,.0f}")


# ═══════════════════════════════════════════════════════════════════
# CONSENSUS STRATEGY
# ═══════════════════════════════════════════════════════════════════

def normalise_probs(home_p, draw_p, away_p):
    """Normalise probabilities to sum to 1.0 (remove overround)."""
    total = home_p + draw_p + away_p
    if total <= 0: return (0, 0, 0)
    return (home_p/total, draw_p/total, away_p/total)


class ConsensusStrategy:
    """
    Back outcomes where a bookmaker's odds significantly exceed
    the consensus (average across all bookmakers).
    
    The consensus is our estimate of the 'true' probability.
    When a bookmaker offers better odds, it's a pricing error.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {
            'min_edge_pct': MIN_EDGE_PCT,
            'min_odds': MIN_ODDS,
            'max_odds': MAX_ODDS,
        }
    
    def evaluate_match(self, row, bankroll: float, stake_frac: float = STAKE_FRAC) -> list[Trade]:
        """
        Evaluate a single match for value bets.
        
        Uses:
          - avg_odds_* = consensus (average across all bookmakers)
          - max_odds_* = best available price
          - top_bookie_* = which bookmaker offers the best price
        
        Returns list of Trade objects (usually 1, maybe 0, rarely 2+).
        """
        trades = []
        
        for outcome, avg_col, max_col, bookie_col, score_col in [
            ('home', 'avg_odds_home_win', 'max_odds_home_win', 'top_bookie_home_win', 'home_score'),
            ('draw', 'avg_odds_draw', 'max_odds_draw', 'top_bookie_draw', None),
            ('away', 'avg_odds_away_win', 'max_odds_away_win', 'top_bookie_away_win', 'away_score'),
        ]:
            avg_odds = row[avg_col]
            max_odds = row[max_col]
            bookie = str(row.get(bookie_col, ''))
            
            if pd.isna(avg_odds) or pd.isna(max_odds) or avg_odds <= 0 or max_odds <= 0:
                continue
            
            # Must beat consensus
            if max_odds <= avg_odds:
                continue
            
            # Odds range filter
            if not (self.config['min_odds'] <= max_odds <= self.config['max_odds']):
                continue
            
            # Edge = how much better is this price than consensus?
            # e.g., consensus = 2.0, max = 2.4, edge = 20%
            edge_pct = (max_odds / avg_odds - 1) * 100
            
            if edge_pct < self.config['min_edge_pct']:
                continue
            
            # Calculate stake
            stake = bankroll * stake_frac
            if stake < 1:
                continue
            
            # Determine outcome
            if score_col:
                home_s = row.get('home_score', -1)
                away_s = row.get('away_score', -1)
                try:
                    home_s = float(home_s) if pd.notna(home_s) else -1
                    away_s = float(away_s) if pd.notna(away_s) else -1
                except:
                    home_s, away_s = -1, -1
                
                if outcome == 'home':
                    won = home_s > away_s
                elif outcome == 'draw':
                    won = home_s == away_s
                else:
                    won = away_s > home_s
            else:
                won = False  # shouldn't happen
            
            trades.append(Trade(
                match_date=str(row.get('match_date', '')),
                league=str(row.get('league', '')),
                home=str(row.get('home_team', '')),
                away=str(row.get('away_team', '')),
                outcome=outcome,
                back_odds=max_odds,
                back_bookie=bookie,
                consensus_odds=avg_odds,
                edge_pct=edge_pct,
                stake=stake,
                won=won,
                profit=0.0,
            ))
        
        return trades


# ═══════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════

def run_simulation():
    """Load data, run consensus strategy, report results."""
    print(SEP)
    print("CONSENSUS PRICING STRATEGY — Simulation")
    print(SEP)
    print(f"\n  Data: closing_odds.csv.gz")
    print(f"  Bankroll: £{BANKROLL:,.0f}")
    print(f"  Stake: {STAKE_FRAC*100:.0f}% per bet")
    print(f"  Min edge: {MIN_EDGE_PCT}% over consensus")
    print(f"  Odds range: {MIN_ODDS}-{MAX_ODDS}")
    
    # ── Load ──
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA / 'closing_odds.csv.gz', compression='gzip',
                     encoding='latin1', low_memory=False)
    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
    df = df.dropna(subset=['match_date'])
    
    # Clean numeric columns
    for c in ['home_score','away_score','avg_odds_home_win','avg_odds_draw',
              'avg_odds_away_win','max_odds_home_win','max_odds_draw','max_odds_away_win']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Only keep rows with valid scores and odds
    df = df.dropna(subset=['home_score','away_score','avg_odds_home_win','avg_odds_draw','avg_odds_away_win'])
    print(f"  {len(df):,} matches ({df['match_date'].min().date()} to {df['match_date'].max().date()})")
    
    # Remove overround from consensus odds to get true probabilities
    for outcome in ['home','draw','away']:
        o_col = f'avg_odds_{outcome}_win'
        if o_col not in df.columns:
            o_col = f'avg_odds_{outcome}'
        p_col = f'consensus_{outcome}_prob'
        df[p_col] = 1.0 / df[o_col]
    
    # Normalise consensus probabilities
    total_p = df['consensus_home_prob'] + df['consensus_draw_prob'] + df['consensus_away_prob']
    for outcome in ['home','draw','away']:
        df[f'consensus_{outcome}_norm'] = df[f'consensus_{outcome}_prob'] / total_p
    
    # Strategy
    strategy = ConsensusStrategy()
    portfolio = Portfolio(bankroll=BANKROLL)
    skipped_no_edge = 0
    skipped_odds_range = 0
    
    # ── Scan all matches ──
    print(f"\n[2] Scanning for value bets...")
    t1 = time.time()
    
    for idx, row in df.iterrows():
        trades = strategy.evaluate_match(row, portfolio.equity, STAKE_FRAC)
        
        for trade in trades:
            portfolio.bet(trade)
        
        # Progress
        if (idx + 1) % 50000 == 0:
            print(f"  ... {idx+1:,}/{len(df):,} matches scanned "
                  f"({len(portfolio.trades)} bets placed) "
                  f"[{time.time()-t1:.0f}s]")
    
    elapsed = time.time() - t1
    
    # ── Report ──
    print(f"\n  Scan complete: {len(df):,} matches in {elapsed:.0f}s")
    portfolio.summary()
    
    # Save
    trades_df = pd.DataFrame([{
        'date': t.match_date,
        'league': t.league,
        'home': t.home,
        'away': t.away,
        'outcome': t.outcome,
        'back_odds': t.back_odds,
        'bookie': t.back_bookie,
        'consensus': t.consensus_odds,
        'edge_pct': t.edge_pct,
        'stake': t.stake,
        'won': t.won,
        'profit': t.profit,
    } for t in portfolio.trades])
    
    trades_df.to_csv(OUT / 'consensus_trades.csv', index=False)
    print(f"\n  Full trade log saved to {OUT}/consensus_trades.csv")
    print(f"  Runtime: {time.time()-t0:.0f}s")
    print(SEP)
    
    return portfolio


if __name__ == '__main__':
    run_simulation()
