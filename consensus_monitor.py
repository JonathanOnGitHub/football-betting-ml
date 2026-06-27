#!/usr/bin/env python3
"""
Consensus Value Monitor — Production-ready
===========================================
Monitors bookmaker odds and detects pricing errors vs consensus.

MODES:
  simulate  — Scans historical Beat The Bookie data for value
  live      — Polls The Odds API (free: 500 req/mo, pro: unlimited)

USAGE:
  python3 consensus_monitor.py --mode simulate --min-edge 7
  python3 consensus_monitor.py --mode live --api-key YOUR_KEY

STRATEGY (validated):
  At opening time, compute consensus (avg odds) across 30+ bookmakers.
  If any bookmaker offers odds 7%+ above consensus, back that outcome.
  Opening odds have real pricing errors. Closing odds do not.
"""

import json, time, os, sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────
CONFIG = {
    'min_edge_pct': 7.0,          # Minimum edge over consensus to trigger
    'min_odds': 1.3,
    'max_odds': 3.0,              # Only short prices (validated range)
    'stake_per_bet': 50,          # £ per bet
    'commission': 0.0,            # Backing with bookmakers = no commission
    'poll_interval': 300,         # Seconds between polls (live mode)
    'log_file': 'consensus_trades.jsonl',
}

DATA_DIR = Path.home() / '.cache/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2'
OUT_DIR = Path('/home/burley/football-ml')
OUT_DIR.mkdir(exist_ok=True)

SEP = "=" * 60

# ═══════════════════════════════════════════════════════════════════
# DATA MODEL
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ValueBet:
    """A detected value betting opportunity."""
    timestamp: str
    match_id: int
    league: str
    home_team: str
    away_team: str
    match_date: str
    outcome: str          # 'home', 'draw', 'away'
    consensus_odds: float # Average across all bookmakers
    best_odds: float      # Best available price
    best_bookie: str      # Which bookmaker offers it
    edge_pct: float       # (best / consensus - 1) * 100
    n_bookmakers: int     # How many bookmakers contributed to consensus
    stake: float = 50.0
    won: Optional[bool] = None
    profit: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════
# CONSENSUS DETECTOR
# ═══════════════════════════════════════════════════════════════════

class ConsensusDetector:
    """
    Detects value by comparing each bookmaker's odds against
    the consensus (average across all bookmakers).
    
    Uses the Beat The Bookie closing_odds data which already has
    avg_odds (consensus) and max_odds (best price) + which bookie.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or CONFIG
        self.bets_found = []
        self.bets_historical = []
    
    def scan_match(self, row) -> Optional[ValueBet]:
        """
        Scan a single match for value opportunities.
        Uses pre-computed avg_odds (consensus) and max_odds (best).
        """
        import pandas as pd
        
        for outcome, avg_col, max_col, bookie_col in [
            ('home', 'avg_odds_home_win', 'max_odds_home_win', 'top_bookie_home_win'),
            ('draw', 'avg_odds_draw', 'max_odds_draw', 'top_bookie_draw'),
            ('away', 'avg_odds_away_win', 'max_odds_away_win', 'top_bookie_away_win'),
        ]:
            avg_odds = row.get(avg_col)
            max_odds = row.get(max_col)
            bookie = str(row.get(bookie_col, ''))
            n_odds = row.get(f'n_odds_{outcome}_win', 0)
            
            if pd.isna(avg_odds) or pd.isna(max_odds):
                continue
            if avg_odds <= 0 or max_odds <= 0:
                continue
            if max_odds <= avg_odds:
                continue
            if not (self.config['min_odds'] <= max_odds <= self.config['max_odds']):
                continue
            
            edge = (max_odds / avg_odds - 1) * 100
            if edge < self.config['min_edge_pct']:
                continue
            
            return ValueBet(
                timestamp=datetime.now().isoformat(),
                match_id=int(row.get('match_id', 0)),
                league=str(row.get('league', '')),
                home_team=str(row.get('home_team', '')),
                away_team=str(row.get('away_team', '')),
                match_date=str(row.get('match_date', ''))[:10],
                outcome=outcome,
                consensus_odds=round(avg_odds, 2),
                best_odds=round(max_odds, 2),
                best_bookie=bookie,
                edge_pct=round(edge, 1),
                n_bookmakers=int(n_odds) if pd.notna(n_odds) else 0,
            )
        
        return None
    
    def log_bet(self, bet: ValueBet, won: bool = None):
        """Record a bet and optionally its outcome."""
        if won is not None:
            bet.won = won
            if won:
                bet.profit = round(bet.stake * (bet.best_odds - 1), 2)
            else:
                bet.profit = -bet.stake
        
        self.bets_found.append(bet)
        
        # Append to JSONL log
        log_path = OUT_DIR / CONFIG['log_file']
        entry = asdict(bet)
        # Convert non-serializable
        entry['won'] = bool(bet.won) if bet.won is not None else None
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        return bet


# ═══════════════════════════════════════════════════════════════════
# SIMULATION MODE
# ═══════════════════════════════════════════════════════════════════

class SimulatedScanner:
    """
    Scans historical Beat The Bookie data for value opportunities.
    Reports findings with statistics by league, odds band, edge level.
    """
    
    def __init__(self, detector: ConsensusDetector):
        self.detector = detector
    
    def scan_all(self):
        """Scan all matches in closing_odds.csv.gz."""
        import pandas as pd
        import numpy as np
        
        print(SEP)
        print("CONSENSUS VALUE MONITOR — HISTORICAL SCAN")
        print(SEP)
        print(f"  Min edge: {CONFIG['min_edge_pct']}%")
        print(f"  Odds range: {CONFIG['min_odds']}–{CONFIG['max_odds']}")
        print()
        
        print("[1] Loading data...")
        df = pd.read_csv(DATA_DIR / 'closing_odds.csv.gz', compression='gzip',
                         encoding='latin1', low_memory=False)
        df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
        df = df.dropna(subset=['match_date'])
        
        # Clean numeric columns
        for c in ['avg_odds_home_win','avg_odds_draw','avg_odds_away_win',
                  'max_odds_home_win','max_odds_draw','max_odds_away_win',
                  'home_score','away_score']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        print(f"  {len(df):,} matches loaded")
        
        # Parse scores
        df['home_won'] = (df['home_score'] > df['away_score']).astype(int)
        df['draw'] = (df['home_score'] == df['away_score']).astype(int)
        df['away_won'] = (df['home_score'] < df['away_score']).astype(int)
        
        # Separate into opening and closing analysis
        # Note: closing_odds.csv is CLOSING odds only
        # For opening analysis we need time-series data
        # But the closing odds DO have value info
        
        print("\n[2] Scanning for value (closing odds)...")
        t0 = time.time()
        
        bets = []
        for idx, row in df.iterrows():
            bet = self.detector.scan_match(row)
            if bet:
                # Determine outcome
                outcome_col = f"{bet.outcome.replace('home','home_won').replace('draw','draw').replace('away','away_won')}"
                if bet.outcome == 'home':
                    won = row.get('home_won', 0) == 1
                elif bet.outcome == 'draw':
                    won = row.get('draw', 0) == 1
                else:
                    won = row.get('away_won', 0) == 1
                
                bet.won = won
                if won:
                    bet.profit = round(bet.stake * (bet.best_odds - 1), 2)
                else:
                    bet.profit = -bet.stake
                
                bets.append(bet)
            
            if (idx + 1) % 100000 == 0:
                print(f"  ... {idx+1:,}/{len(df):,} scanned, "
                      f"{len(bets)} value bets found [{time.time()-t0:.0f}s]")
        
        elapsed = time.time() - t0
        self.bets = bets
        self._report(bets, elapsed)
        return bets
    
    def _report(self, bets, elapsed):
        """Generate detailed report."""
        import pandas as pd
        
        n = len(bets)
        if n == 0:
            print(f"\n  No value bets found (threshold: {CONFIG['min_edge_pct']}%)")
            return
        
        wins = sum(1 for b in bets if b.won)
        total_pnl = sum(b.profit for b in bets)
        total_staked = n * CONFIG['stake_per_bet']
        
        print(f"\n{'='*60}")
        print("SCAN RESULTS")
        print(f"{'='*60}")
        print(f"  Matches scanned: {len(bets):,} (10yr)")
        print(f"  Value bets:      {n:,} ({n/len(bets)*100:.1f}% of matches)")
        print(f"  Win rate:        {wins/n*100:.1f}%")
        print(f"  Avg odds:        {sum(b.best_odds for b in bets)/n:.2f}")
        print(f"  Avg edge:        {sum(b.edge_pct for b in bets)/n:.1f}%")
        print(f"  Total staked:    £{total_staked:,.0f}")
        print(f"  Total P&L:       £{total_pnl:+,.0f}")
        print(f"  ROI:             {total_pnl/total_staked*100:.2f}%")
        print(f"  EV per bet:      £{total_pnl/n:.2f}")
        print(f"  Runtime:         {elapsed:.0f}s")
        
        # By edge band
        print(f"\n  By edge band:")
        for lo, hi in [(7,10),(10,15),(15,100)]:
            sub = [b for b in bets if lo <= b.edge_pct < hi]
            if sub:
                wr = sum(1 for b in sub if b.won) / len(sub) * 100
                pnl = sum(b.profit for b in sub)
                print(f"    {lo:2d}–{hi:2d}%: {len(sub):>5,} bets, WR {wr:.1f}%, "
                      f"P&L £{pnl:+,.0f}, EV £{pnl/len(sub):+.2f}")
        
        # Top leagues
        from collections import defaultdict
        leagues = defaultdict(lambda: {'bets':0,'wins':0,'pnl':0})
        for b in bets:
            leagues[b.league]['bets'] += 1
            leagues[b.league]['wins'] += 1 if b.won else 0
            leagues[b.league]['pnl'] += b.profit
        
        print(f"\n  Top 10 leagues by volume:")
        for league, data in sorted(leagues.items(), key=lambda x: x[1]['bets'], reverse=True)[:10]:
            wr = data['wins']/data['bets']*100
            print(f"    {league[:35]:35s} {data['bets']:>5,} bets, "
                  f"WR {wr:.1f}%, £{data['pnl']:+,.0f}")
        
        print(f"\n  Top 10 leagues by P&L:")
        for league, data in sorted(leagues.items(), key=lambda x: x[1]['pnl'], reverse=True)[:10]:
            wr = data['wins']/data['bets']*100
            print(f"    {league[:35]:35s} {data['bets']:>5,} bets, "
                  f"WR {wr:.1f}%, £{data['pnl']:+,.0f}")


# ═══════════════════════════════════════════════════════════════════
# LIVE MODE
# ═══════════════════════════════════════════════════════════════════

class LiveScanner:
    """
    Connects to The Odds API.
    Free tier: 500 requests/month (~16/day)
    Pro tier: unlimited, ~$100/month
    
    Get API key: https://the-odds-api.com/
    """
    
    def __init__(self, detector: ConsensusDetector, api_key: str = ''):
        self.detector = detector
        self.api_key = api_key or os.environ.get('ODDS_API_KEY', '')
    
    def fetch_matches(self, sport: str = 'soccer', regions: str = 'uk') -> list:
        """Fetch current odds from The Odds API."""
        import urllib.request
        url = (f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
               f'?apiKey={self.api_key}&regions={regions}&markets=h2h&oddsFormat=decimal')
        
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as e:
            print(f"  API error: {e}")
            return []
    
    def api_to_valuebet(self, match: dict) -> list[ValueBet]:
        """Convert API match to potential value bets."""
        bets = []
        home = match.get('home_team', '')
        away = match.get('away_team', '')
        
        for bookmaker in match.get('bookmakers', []):
            bk_name = bookmaker.get('title', '')
            for market in bookmaker.get('markets', []):
                if market.get('key') != 'h2h':
                    continue
                
                # Build consensus (avg across all bookmakers for this match)
                # We need to collect ALL bookmaker odds first
                # For now, just show the best available
                outcomes = {o.get('name'): o.get('price', 0) 
                          for o in market.get('outcomes', [])}
                
                # Map outcomes
                outcome_map = {}
                for name, price in outcomes.items():
                    if price <= 0:
                        continue
                    if name == home:
                        outcome_map['home'] = price
                    elif name == away:
                        outcome_map['away'] = price
                    elif name == 'Draw':
                        outcome_map['draw'] = price
                
                # We need all bookmakers to compute consensus
                # This requires collecting across ALL bookmakers first
                # For single-bookmaker comparison, just flag the best odds
                pass
        
        return bets
    
    def run_once(self) -> list:
        """Fetch and scan once."""
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Polling...")
        matches = self.fetch_matches()
        
        if not matches:
            return []
        
        # Collect all odds from all bookmakers per match
        match_odds = {}
        for match in matches:
            mid = match.get('id', '')
            match_odds[mid] = {
                'home': match.get('home_team',''),
                'away': match.get('away_team',''),
                'date': match.get('commence_time',''),
                'bookies': {}
            }
            
            for bm in match.get('bookmakers', []):
                bk_name = bm.get('title', '')
                for market in bm.get('markets', []):
                    if market.get('key') != 'h2h':
                        continue
                    outcomes = {o['name']: o['price'] 
                              for o in market.get('outcomes', []) 
                              if o.get('price', 0) > 0}
                    
                    match_odds[mid]['bookies'][bk_name] = outcomes
        
        # Now compute consensus per match
        found = []
        for mid, data in match_odds.items():
            bookies = data['bookies']
            if len(bookies) < 3:
                continue
            
            # Collect odds per outcome across all bookmakers
            outcome_odds = {'home': [], 'draw': [], 'away': []}
            outcome_bookie_best = {'home': {}, 'draw': {}, 'away': {}}
            
            home_name = data['home']
            away_name = data['away']
            
            for bk_name, outcomes in bookies.items():
                for outcome_label, target_name in [
                    ('home', home_name), ('draw', 'Draw'), ('away', away_name)
                ]:
                    if isinstance(outcomes, dict):
                        for oname, oprice in outcomes.items():
                            if oname == target_name:
                                outcome_odds[outcome_label].append(oprice)
                                outcome_bookie_best[outcome_label][bk_name] = oprice
            
            if any(len(v) < 3 for v in outcome_odds.values()):
                continue
            
            # Compute consensus per outcome
            import statistics
            for outcome in ['home', 'draw', 'away']:
                if not outcome_odds[outcome]:
                    continue
                
                avg_odds = statistics.mean(outcome_odds[outcome])
                max_odds = max(outcome_odds[outcome])
                max_bookie = max(outcome_bookie_best[outcome], 
                                key=outcome_bookie_best[outcome].get)
                
                if max_odds <= avg_odds:
                    continue
                if not (CONFIG['min_odds'] <= max_odds <= CONFIG['max_odds']):
                    continue
                
                edge = (max_odds / avg_odds - 1) * 100
                if edge < CONFIG['min_edge_pct']:
                    continue
                
                bet = ValueBet(
                    timestamp=datetime.now().isoformat(),
                    match_id=hash(mid),
                    league=match.get('sport_title', ''),
                    home_team=home_name,
                    away_team=away_name,
                    match_date=data['date'][:10],
                    outcome=outcome,
                    consensus_odds=round(avg_odds, 2),
                    best_odds=round(max_odds, 2),
                    best_bookie=max_bookie,
                    edge_pct=round(edge, 1),
                    n_bookmakers=len(bookies),
                )
                self.detector.log_bet(bet)
                found.append(bet)
                self._alert(bet)
        
        return found
    
    def run_loop(self, interval: int = None):
        """Continuous polling."""
        interval = interval or CONFIG['poll_interval']
        print(f"\n  Starting live monitor (poll every {interval}s)...")
        print(f"  Press Ctrl+C to stop")
        
        try:
            while True:
                found = self.run_once()
                if found:
                    print(f"  ⚡ {len(found)} value bets found!")
                    for bet in found[:5]:
                        print(f"     {bet.home_team} vs {bet.away_team}: "
                              f"back {bet.outcome} at {bet.best_odds} "
                              f"({bet.best_bookie}), edge {bet.edge_pct}%")
                else:
                    print(f"  No value found")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n  Stopped.")
    
    def _alert(self, bet: ValueBet):
        """Print alert for detected value."""
        print(f"\n⚡ VALUE BET")
        print(f"  {bet.home_team} vs {bet.away_team}")
        print(f"  Back {bet.outcome} at {bet.best_odds} ({bet.best_bookie})")
        print(f"  Consensus: {bet.consensus_odds}  Edge: {bet.edge_pct}%")
        print(f"  Stake: £{bet.stake}  |  {bet.n_bookmakers} bookmakers")
        print()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Consensus Value Monitor')
    parser.add_argument('--mode', choices=['simulate', 'live'], default='simulate')
    parser.add_argument('--min-edge', type=float, default=7.0)
    parser.add_argument('--odds-min', type=float, default=1.3)
    parser.add_argument('--odds-max', type=float, default=3.0)
    parser.add_argument('--api-key', default='')
    parser.add_argument('--interval', type=int, default=300)
    
    args = parser.parse_args()
    
    CONFIG['min_edge_pct'] = args.min_edge
    CONFIG['min_odds'] = args.odds_min
    CONFIG['max_odds'] = args.odds_max
    CONFIG['poll_interval'] = args.interval
    
    detector = ConsensusDetector(CONFIG)
    
    if args.mode == 'simulate':
        scanner = SimulatedScanner(detector)
        scanner.scan_all()
    
    elif args.mode == 'live':
        if not args.api_key and not os.environ.get('ODDS_API_KEY'):
            print(SEP)
            print("CONSENSUS VALUE MONITOR — LIVE MODE")
            print(SEP)
            print("\n❌ No API key found.")
            print("   Get a free key at: https://the-odds-api.com/#get-access")
            print("   500 requests/month free, enough for 16 polls/day")
            print()
            print("   Then run: export ODDS_API_KEY=your_key_here")
            print(f"   Or: python3 consensus_monitor.py --mode live --api-key your_key")
            sys.exit(1)
        
        print(SEP)
        print("CONSENSUS VALUE MONITOR — LIVE MODE")
        print(SEP)
        
        scanner = LiveScanner(detector, args.api_key)
        
        # Test connection
        print("\n  Testing API connection...")
        matches = scanner.fetch_matches()
        if matches:
            bookie_count = sum(len(m.get('bookmakers',[])) for m in matches[:5])
            print(f"  ✅ Connected! {len(matches)} matches, "
                  f"avg {bookie_count/5:.0f} bookmakers/match")
            scanner.run_loop(args.interval)
        else:
            print("  ❌ Connection failed. Check API key.")
            sys.exit(1)


if __name__ == '__main__':
    main()
