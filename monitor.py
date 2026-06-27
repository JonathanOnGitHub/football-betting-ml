#!/usr/bin/env python3
"""
Football Arbitrage Monitor — Real-time odds monitoring & trade execution.
======================================================================

TWO MODES:
  1. SIMULATION: Replays historical time-series odds data to detect 
     missed arb/trading opportunities.
  2. LIVE: Connects to The Odds API (free tier) for real monitoring.

USAGE:
  # Simulation mode (default):
  python3 monitor.py --mode simulate
  
  # Live mode (requires API key):
  python3 monitor.py --mode live --api-key YOUR_KEY
"""

import json, time, os, sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────
CONFIG = {
    'min_arb_return': 0.5,     # Minimum arb % to alert (e.g., 0.5%)
    'min_swing': 3.0,          # Minimum odds swing % for back-to-lay
    'stake_per_trade': 100,    # £ per arb trade
    'commission': 0.05,        # Betfair commission
    'poll_interval': 60,       # Seconds between polls (live mode)
    'max_spread': 0.15,        # Max arb as fraction (15% = extreme)
    'log_file': 'trades.log',
    'alert_telegram': False,   # Set True to send alerts via Hermes
}

DATA_DIR = Path.home() / '.cache/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2'
OUT_DIR = Path('/home/burley/football-ml')
OUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OddsSnapshot:
    """Odds for one match at one time point across all bookmakers."""
    match_id: int
    league: str
    home_team: str
    away_team: str
    match_date: str
    match_time: str
    timestamp: str
    
    # Best odds across all bookmakers for each outcome
    best_home: float
    best_draw: float  
    best_away: float
    
    # Full bookmaker matrix (bookie_id -> odds)
    all_home: dict  # {bookie_id: odds}
    all_draw: dict
    all_away: dict
    
    @property
    def implied_sum(self):
        """Sum of implied probabilities from best odds."""
        return 1/self.best_home + 1/self.best_draw + 1/self.best_away
    
    @property
    def arb_return_pct(self):
        """Guaranteed return % if arb exists (negative = no arb)."""
        if self.implied_sum >= 1.0:
            return 0.0
        return (1.0 / self.implied_sum - 1.0) * 100


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity."""
    match_id: int
    league: str
    home_team: str
    away_team: str
    match_date: str
    detected_at: str
    
    best_home_odds: float
    best_draw_odds: float
    best_away_odds: float
    
    best_home_bookie: str
    best_draw_bookie: str
    best_away_bookie: str
    
    return_pct: float
    
    # Stakes for £100 total return
    stake_home: float = 0
    stake_draw: float = 0
    stake_away: float = 0
    total_stake: float = 0
    guaranteed_profit: float = 0
    
    def __post_init__(self):
        """Calculate optimal stakes for £100 guaranteed return."""
        implied = 1/self.best_home_odds + 1/self.best_draw_odds + 1/self.best_away_odds
        if implied < 1.0:
            # Stake to return £100 on any outcome
            self.stake_home = 100 / self.best_home_odds
            self.stake_draw = 100 / self.best_draw_odds
            self.stake_away = 100 / self.best_away_odds
            self.total_stake = self.stake_home + self.stake_draw + self.stake_away
            self.guaranteed_profit = 100 - self.total_stake


@dataclass
class SwingTrade:
    """Opening→Closing back-to-lay opportunity."""
    match_id: int
    league: str
    home_team: str
    away_team: str
    match_date: str
    
    outcome: str  # 'home', 'draw', 'away'
    open_odds: float
    close_odds: float
    swing_pct: float
    
    # Green-book calculation
    back_stake: float = 100  # £100 back at open odds
    lay_stake: float = 0     # Calculated
    guaranteed_profit: float = 0
    
    def __post_init__(self):
        """Calculate green-book stakes."""
        if self.open_odds > self.close_odds:
            # Back £X at open_odds, lay £Y at close_odds
            # Y = X * open_odds / close_odds (for equal profit regardless)
            self.lay_stake = self.back_stake * self.open_odds / self.close_odds
            profit_if_wins = self.back_stake * (self.open_odds - 1) - self.lay_stake * (self.close_odds - 1)
            profit_if_loses = -self.back_stake + self.lay_stake
            self.guaranteed_profit = min(profit_if_wins, profit_if_loses) * (1 - CONFIG['commission'])


# ═══════════════════════════════════════════════════════════════════
# ARB DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════

class ArbDetector:
    """Detects arbitrage and trading opportunities from odds data."""
    
    def __init__(self, config: dict = None):
        self.config = config or CONFIG
        self.trades_log = []
        self.opportunities_found = 0
    
    def check_match_odds(self, snapshot: OddsSnapshot) -> Optional[ArbOpportunity]:
        """Check if a single match snapshot has an arbitrage opportunity."""
        implied = snapshot.implied_sum
        
        if implied >= 1.0:
            return None
        
        ret_pct = snapshot.arb_return_pct
        if ret_pct < self.config['min_arb_return']:
            return None
        if ret_pct > self.config['max_spread'] * 100:
            return None  # Too good to be true — likely data error
        
        # Find which bookmaker offers each best price
        def best_bookie(odds_dict, target_odds):
            for bk, od in odds_dict.items():
                if abs(od - target_odds) < 0.001:
                    return bk
            return 'unknown'
        
        return ArbOpportunity(
            match_id=snapshot.match_id,
            league=snapshot.league,
            home_team=snapshot.home_team,
            away_team=snapshot.away_team,
            match_date=snapshot.match_date,
            detected_at=datetime.now().isoformat(),
            best_home_odds=snapshot.best_home,
            best_draw_odds=snapshot.best_draw,
            best_away_odds=snapshot.best_away,
            best_home_bookie=best_bookie(snapshot.all_home, snapshot.best_home),
            best_draw_bookie=best_bookie(snapshot.all_draw, snapshot.best_draw),
            best_away_bookie=best_bookie(snapshot.all_away, snapshot.best_away),
            return_pct=ret_pct,
        )
    
    def check_swing(self, open_snap: OddsSnapshot, close_snap: OddsSnapshot) -> list[SwingTrade]:
        """Check for back-to-lay opportunities between two time points."""
        trades = []
        
        for outcome, open_val, close_val in [
            ('home', open_snap.best_home, close_snap.best_home),
            ('draw', open_snap.best_draw, close_snap.best_draw),
            ('away', open_snap.best_away, close_snap.best_away),
        ]:
            if open_val is None or close_val is None or open_val <= close_val:
                continue
            
            swing = (open_val / close_val - 1) * 100
            if swing < self.config['min_swing']:
                continue
            
            trade = SwingTrade(
                match_id=open_snap.match_id,
                league=open_snap.league,
                home_team=open_snap.home_team,
                away_team=open_snap.away_team,
                match_date=open_snap.match_date,
                outcome=outcome,
                open_odds=open_val,
                close_odds=close_val,
                swing_pct=swing,
            )
            trades.append(trade)
        
        return trades
    
    def log_trade(self, trade):
        """Log a discovered opportunity."""
        self.opportunities_found += 1
        self.trades_log.append(trade)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': type(trade).__name__,
            'match_id': trade.match_id,
            'league': trade.league,
            'home': trade.home_team,
            'away': trade.away_team,
            'return_pct': getattr(trade, 'return_pct', None) or getattr(trade, 'swing_pct', None),
            'profit': getattr(trade, 'guaranteed_profit', 0),
        }
        
        # Append to log file
        log_path = OUT_DIR / CONFIG['log_file']
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return log_entry


# ═══════════════════════════════════════════════════════════════════
# SIMULATION MODE: Replay historical time-series data
# ═══════════════════════════════════════════════════════════════════

class SimulatedMonitor:
    """Replays Beat The Bookie time-series data as if it were live."""
    
    def __init__(self, detector: ArbDetector):
        self.detector = detector
        self.df = None
        self.bookmaker_cols = {}
    
    def load_data(self):
        """Load the time-series odds data."""
        print("Loading time-series data for simulation...")
        
        import pandas as pd
        
        # Load only the columns we need
        basic = ['match_id','match_date','match_time','score_home','score_away']
        
        # We'll load all time points for bookmakers 1-8 (subset for speed)
        # In production, you'd load all 32
        time_cols = []
        for outcome in ['home','draw','away']:
            for t in [71, 0]:  # Opening (71) and closing (0)
                for b in range(1, 9):  # First 8 bookmakers
                    col = f'{outcome}_b{b}_{t}'
                    time_cols.append(col)
                    self.bookmaker_cols[col] = (outcome, b, t)
        
        df = pd.read_csv(
            DATA_DIR / 'odds_series.csv.gz',
            compression='gzip', encoding='latin1',
            usecols=basic + time_cols
        )
        
        # Filter to matches with scores (completed matches in simulation)
        df = df.dropna(subset=['score_home', 'score_away'])
        print(f"  Loaded {len(df):,} completed matches")
        
        # Melt to time-point format and sort
        self.df = df.sort_values('match_date').reset_index(drop=True)
        return self
    
    def scan_all(self):
        """Scan all historical data for opportunities."""
        if self.df is None:
            self.load_data()
        
        import pandas as pd
        df = self.df
        results = {'arbs': 0, 'swings': 0, 'total_profit': 0.0}
        
        print(f"\nScanning {len(df):,} matches for opportunities...")
        t0 = time.time()
        
        for idx, row in df.iterrows():
            match_id = row['match_id']
            
            # Build opening snapshot (t=71)
            open_home = {b: row.get(f'home_b{b}_71') for b in range(1, 9)}
            open_draw = {b: row.get(f'draw_b{b}_71') for b in range(1, 9)}
            open_away = {b: row.get(f'away_b{b}_71') for b in range(1, 9)}
            open_home_v = max(v for v in open_home.values() if pd.notna(v)) if any(pd.notna(v) for v in open_home.values()) else None
            open_draw_v = max(v for v in open_draw.values() if pd.notna(v)) if any(pd.notna(v) for v in open_draw.values()) else None
            open_away_v = max(v for v in open_away.values() if pd.notna(v)) if any(pd.notna(v) for v in open_away.values()) else None
            
            if None in (open_home_v, open_draw_v, open_away_v):
                continue
            
            open_snap = OddsSnapshot(
                match_id=match_id, league='', home_team='', away_team='',
                match_date=str(row['match_date']), match_time=str(row['match_time']),
                timestamp=str(row['match_date']),
                best_home=open_home_v, best_draw=open_draw_v, best_away=open_away_v,
                all_home=open_home, all_draw=open_draw, all_away=open_away,
            )
            
            # Build closing snapshot (t=0)
            close_home = {b: row.get(f'home_b{b}_0') for b in range(1, 9)}
            close_draw = {b: row.get(f'draw_b{b}_0') for b in range(1, 9)}
            close_away = {b: row.get(f'away_b{b}_0') for b in range(1, 9)}
            close_home_v = max(v for v in close_home.values() if pd.notna(v)) if any(pd.notna(v) for v in close_home.values()) else None
            close_draw_v = max(v for v in close_draw.values() if pd.notna(v)) if any(pd.notna(v) for v in close_draw.values()) else None
            close_away_v = max(v for v in close_away.values() if pd.notna(v)) if any(pd.notna(v) for v in close_away.values()) else None
            
            if None in (close_home_v, close_draw_v, close_away_v):
                continue
            
            close_snap = OddsSnapshot(
                match_id=match_id, league='', home_team='', away_team='',
                match_date=str(row['match_date']), match_time=str(row['match_time']),
                timestamp=str(row['match_date']),
                best_home=close_home_v, best_draw=close_draw_v, best_away=close_away_v,
                all_home=close_home, all_draw=close_draw, all_away=close_away,
            )
            
            # Check for arb at opening
            arb = self.detector.check_match_odds(open_snap)
            if arb:
                results['arbs'] += 1
                results['total_profit'] += arb.guaranteed_profit
                entry = self.detector.log_trade(arb)
                if results['arbs'] <= 5:
                    print(f"  ⚡ ARB: {row['match_date']} | {open_home_v:.1f}/{open_draw_v:.1f}/{open_away_v:.1f} "
                          f"| return {arb.return_pct:.1f}% | profit £{arb.guaranteed_profit:.2f}")
            
            # Check for swing trades
            swings = self.detector.check_swing(open_snap, close_snap)
            for swing in swings:
                results['swings'] += 1
                results['total_profit'] += swing.guaranteed_profit
                entry = self.detector.log_trade(swing)
                if results['swings'] <= 5:
                    print(f"  📈 SWING: {row['match_date']} | {swing.outcome} "
                          f"{swing.open_odds:.1f}→{swing.close_odds:.1f} ({swing.swing_pct:.1f}%) "
                          f"| profit £{swing.guaranteed_profit:.2f}")
            
            # Progress
            if (idx + 1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (len(df) - idx - 1) / rate
                print(f"  ... {idx+1}/{len(df)} matches scanned "
                      f"({results['arbs']} arbs, {results['swings']} swings) "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")
        
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Matches scanned:     {len(df):,}")
        print(f"  Arbs found:          {results['arbs']:,}")
        print(f"  Swing trades found:  {results['swings']:,}")
        print(f"  Total opportunities: {results['arbs'] + results['swings']:,}")
        print(f"  Total profit:        £{results['total_profit']:,.2f}")
        print(f"  Avg per trade:       £{results['total_profit']/(results['arbs']+results['swings']+0.001):.2f}")
        print(f"  Runtime:             {elapsed:.0f}s")
        print(f"  Log saved to:        {OUT_DIR / CONFIG['log_file']}")
        print()
        
        return results


# ═══════════════════════════════════════════════════════════════════
# LIVE MODE: Poll The Odds API
# ═══════════════════════════════════════════════════════════════════

class LiveMonitor:
    """
    Connects to The Odds API for real-time monitoring.
    
    GET A FREE API KEY:
      https://the-odds-api.com/#get-access
      Free tier: 500 requests/month, covers major bookmakers.
    
    REQUIRED ENV: ODDS_API_KEY
    """
    
    def __init__(self, detector: ArbDetector, api_key: str = None):
        self.detector = detector
        self.api_key = api_key or os.environ.get('ODDS_API_KEY', '')
        self.base_url = 'https://api.the-odds-api.com/v4'
    
    def fetch_odds(self, sport: str = 'soccer', regions: str = 'uk,eu,us') -> list:
        """
        Fetch current odds from The Odds API.
        
        Args:
            sport: 'soccer', 'basketball', etc. Use 'upcoming' for all.
            regions: Bookmaker regions. 'uk' for UK bookies, 'eu' for European.
            
        Returns:
            List of match odds objects.
        """
        import urllib.request
        url = (f'{self.base_url}/sports/{sport}/odds'
               f'?apiKey={self.api_key}'
               f'&regions={regions}'
               f'&markets=h2h'
               f'&oddsFormat=decimal')
        
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            return data
        except Exception as e:
            print(f"  API error: {e}")
            return []
    
    def api_to_snapshot(self, match: dict) -> OddsSnapshot:
        """Convert The Odds API match object to our OddsSnapshot format."""
        home_team = match.get('home_team', '')
        away_team = match.get('away_team', '')
        
        all_home = {}
        all_draw = {}
        all_away = {}
        
        for bookmaker in match.get('bookmakers', []):
            bk_name = bookmaker.get('title', 'unknown')
            for market in bookmaker.get('markets', []):
                if market.get('key') != 'h2h':
                    continue
                for outcome in market.get('outcomes', []):
                    name = outcome.get('name', '')
                    price = outcome.get('price', 0)
                    if name == home_team:
                        all_home[bk_name] = price
                    elif name == away_team:
                        all_away[bk_name] = price
                    elif name == 'Draw':
                        all_draw[bk_name] = price
        
        best_home = max(all_home.values()) if all_home else 0
        best_draw = max(all_draw.values()) if all_draw else 0
        best_away = max(all_away.values()) if all_away else 0
        
        return OddsSnapshot(
            match_id=hash(match.get('id', '')),
            league=match.get('sport_title', ''),
            home_team=home_team,
            away_team=away_team,
            match_date=match.get('commence_time', ''),
            match_time='',
            timestamp=datetime.now().isoformat(),
            best_home=best_home,
            best_draw=best_draw,
            best_away=best_away,
            all_home=all_home,
            all_draw=all_draw,
            all_away=all_away,
        )
    
    def run_once(self) -> list:
        """Fetch odds once and check for opportunities."""
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Polling odds...")
        
        matches = self.fetch_odds()
        
        if not matches:
            print("  No matches returned")
            return []
        
        found = []
        for match in matches:
            snapshot = self.api_to_snapshot(match)
            if snapshot.best_home == 0:
                continue
            
            # Check for arb
            arb = self.detector.check_match_odds(snapshot)
            if arb:
                entry = self.detector.log_trade(arb)
                found.append(entry)
                self._alert(arb)
        
        return found
    
    def run_loop(self, interval: int = None):
        """Run continuous monitoring loop."""
        interval = interval or CONFIG['poll_interval']
        print(f"  Starting live monitor (poll every {interval}s)...")
        print(f"  Press Ctrl+C to stop")
        
        try:
            while True:
                found = self.run_once()
                if found:
                    print(f"  ⚡ {len(found)} opportunities found!")
                else:
                    print(f"  No opportunities this poll")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n  Monitor stopped.")
    
    def _alert(self, arb: ArbOpportunity):
        """Send alert about a detected opportunity."""
        msg = (
            f"⚡ ARB DETECTED\n"
            f"{arb.home_team} vs {arb.away_team}\n"
            f"Return: {arb.return_pct:.1f}%\n"
            f"Profit: £{arb.guaranteed_profit:.2f}\n"
            f"Odds: {arb.best_home_odds:.2f} ({arb.best_home_bookie}) / "
            f"{arb.best_draw_odds:.2f} ({arb.best_draw_bookie}) / "
            f"{arb.best_away_odds:.2f} ({arb.best_away_bookie})"
        )
        print(f"\n{'='*50}")
        print(msg)
        print(f"{'='*50}\n")
        
        if CONFIG.get('alert_telegram'):
            # This would use Hermes send_message tool in a cron context
            print("  (Telegram alert configured but needs cron integration)")


# ═══════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Football Arbitrage Monitor')
    parser.add_argument('--mode', choices=['simulate', 'live'], default='simulate',
                       help='Run mode: simulate (replay history) or live (API)')
    parser.add_argument('--api-key', help='The Odds API key (live mode)')
    parser.add_argument('--min-return', type=float, default=0.5,
                       help='Minimum arb return % to alert')
    parser.add_argument('--min-swing', type=float, default=3.0,
                       help='Minimum odds swing % for back-to-lay trade')
    parser.add_argument('--interval', type=int, default=60,
                       help='Poll interval in seconds (live mode)')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['min_arb_return'] = args.min_return
    CONFIG['min_swing'] = args.min_swing
    CONFIG['poll_interval'] = args.interval
    
    detector = ArbDetector(CONFIG)
    
    if args.mode == 'simulate':
        print("=" * 60)
        print("FOOTBALL ARBITRAGE MONITOR — SIMULATION MODE")
        print("=" * 60)
        print(f"\nConfig: min_arb={args.min_return}% min_swing={args.min_swing}%")
        
        monitor = SimulatedMonitor(detector)
        monitor.load_data()
        monitor.scan_all()
        
    elif args.mode == 'live':
        if not args.api_key and not os.environ.get('ODDS_API_KEY'):
            print("=" * 60)
            print("FOOTBALL ARBITRAGE MONITOR — LIVE MODE")
            print("=" * 60)
            print("\n❌ No API key found.")
            print("   Get a free key at: https://the-odds-api.com/#get-access")
            print("   Then run: export ODDS_API_KEY=your_key_here")
            print(f"   Or: python3 monitor.py --mode live --api-key your_key")
            sys.exit(1)
        
        print("=" * 60)
        print("FOOTBALL ARBITRAGE MONITOR — LIVE MODE")
        print("=" * 60)
        
        monitor = LiveMonitor(detector, args.api_key)
        
        # Test connection
        print("\nTesting API connection...")
        matches = monitor.fetch_odds(sport='upcoming', regions='uk')
        if matches:
            print(f"  ✅ Connected! {len(matches)} upcoming matches found")
            # Show first match as example
            ex = monitor.api_to_snapshot(matches[0])
            print(f"  Example: {ex.home_team} vs {ex.away_team}")
            print(f"  Best odds: H={ex.best_home:.2f} D={ex.best_draw:.2f} A={ex.best_away:.2f}")
            print(f"  Implied sum: {ex.implied_sum:.4f}")
            if ex.implied_sum < 1.0:
                print(f"  ⚡ ARB: {ex.arb_return_pct:.2f}%")
        else:
            print("  ❌ Connection failed. Check your API key.")
            print("  Note: Free tier has 500 req/month limit")
            sys.exit(1)
        
        monitor.run_loop(args.interval)


if __name__ == '__main__':
    main()
