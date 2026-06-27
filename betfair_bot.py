#!/usr/bin/env python3
"""
Betfair Trading Bot — Back-to-Lay arbitrage on football.
=======================================================
Modes:
  simulate: Replays historical data as a 3-week trading simulation.
  live:     Connects to Betfair API (requires credentials).

Usage:
  # Simulation (3 weeks starting £1k):
  python3 betfair_bot.py --mode simulate --bankroll 1000 --days 21

  # Live (requires Betfair API key):
  python3 betfair_bot.py --mode live --bankroll 1000
"""

import json, time, os, sys, math
from pathlib import Path
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

SEP = "=" * 60
DATA_DIR = Path.home() / '.cache/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2'
OUT_DIR = Path('/home/burley/football-ml')
OUT_DIR.mkdir(exist_ok=True)

# ── CONFIG ──────────────────────────────────────────────────────────
CONFIG = {
    'min_swing_pct': 3.0,      # Min % odds swing to trade
    'max_odds': 20.0,           # Don't trade odds above this (illiquid)
    'min_odds': 1.3,            # Don't trade odds below this (tiny edge)
    'commission': 0.05,         # Betfair commission on winning bets
    'stake_per_trade': 50,      # £ per back bet
    'max_concurrent': 20,       # Max open positions at once
}

# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """An open back-to-lay trade."""
    match_id: int
    league: str
    home_team: str
    away_team: str
    match_date: str
    outcome: str  # 'home', 'draw', 'away'
    
    open_time: str
    open_odds: float
    back_stake: float  # £ amount backed
    
    lay_odds: Optional[float] = None
    lay_stake: Optional[float] = None
    status: str = 'open'  # 'open', 'closed'
    pnl: float = 0.0
    close_time: Optional[str] = None


@dataclass
class Portfolio:
    """Tracks bankroll, positions, and P&L."""
    bankroll: float
    peak: float = 0.0
    positions: list = field(default_factory=list)
    closed_trades: list = field(default_factory=list)
    daily_pnl: dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.peak = self.bankroll
        self.start_date = None
    
    @property
    def open_count(self):
        return sum(1 for p in self.positions if p.status == 'open')
    
    @property
    def total_pnl(self):
        return sum(t.pnl for t in self.closed_trades)
    
    @property
    def equity(self):
        return self.bankroll + self.total_pnl
    
    @property
    def drawdown_pct(self):
        if self.peak == 0: return 0
        return (self.peak - self.equity) / self.peak * 100
    
    def open_position(self, pos: Position):
        """Record a new position."""
        self.positions.append(pos)
    
    def close_position(self, pos: Position, lay_odds: float, timestamp: str):
        """Close a position and record P&L."""
        pos.lay_odds = lay_odds
        pos.close_time = timestamp
        
        # Green-book calculation
        # Back £X at open_odds, lay £Y at close_odds
        # For equal profit regardless: Y = X * open_odds / close_odds
        pos.lay_stake = pos.back_stake * pos.open_odds / lay_odds
        
        # Profit if outcome wins: back_profit - lay_loss
        profit_if_wins = pos.back_stake * (pos.open_odds - 1) - pos.lay_stake * (lay_odds - 1)
        # Profit if outcome loses: -back_stake + lay_stake (lay wins)
        profit_if_loses = -pos.back_stake + pos.lay_stake
        
        # Guaranteed profit = the smaller of the two (we green the book)
        pos.pnl = min(profit_if_wins, profit_if_loses) * (1 - CONFIG['commission'])
        pos.status = 'closed'
        
        # Record daily P&L
        day = timestamp[:10]
        self.daily_pnl[day] = self.daily_pnl.get(day, 0) + pos.pnl
        
        self.closed_trades.append(pos)
        
        # Update peak for drawdown tracking
        if self.equity > self.peak:
            self.peak = self.equity
    
    def summary(self):
        """Print portfolio summary."""
        days = len(self.daily_pnl)
        wins = sum(1 for t in self.closed_trades if t.pnl > 0)
        losses = sum(1 for t in self.closed_trades if t.pnl <= 0)
        
        print(f"\n{'─'*50}")
        print(f"PORTFOLIO SUMMARY")
        print(f"{'─'*50}")
        print(f"  Starting bankroll:  £{self.bankroll:,.2f}")
        print(f"  Current equity:     £{self.equity:,.2f}")
        print(f"  Total P&L:          £{self.total_pnl:+,.2f}")
        print(f"  Return:             {(self.equity/self.bankroll-1)*100:+.2f}%")
        print(f"  Trades closed:      {len(self.closed_trades)}")
        print(f"  Winning trades:     {wins} ({wins/len(self.closed_trades)*100:.0f}%)" if self.closed_trades else "  Winning trades:     0")
        print(f"  Max drawdown:       {self.drawdown_pct:.2f}%")
        print(f"  Trading days:       {days}")
        if days > 0:
            daily_returns = list(self.daily_pnl.values())
            avg_day = sum(daily_returns) / days
            print(f"  Avg daily P&L:      £{avg_day:+.2f}")
            print(f"  Best day:           £{max(daily_returns):+.2f}")
            print(f"  Worst day:          £{min(daily_returns):+.2f}")
        print(f"{'─'*50}\n")


# ═══════════════════════════════════════════════════════════════════
# TRADING ENGINE
# ═══════════════════════════════════════════════════════════════════

class TradingBot:
    """Back-to-Lay trading bot. Simulated or live Betfair execution."""
    
    def __init__(self, portfolio: Portfolio, config: dict = None):
        self.portfolio = portfolio
        self.config = config or CONFIG
    
    def evaluate(self, match_id, league, home, away, match_date,
                 open_home, open_draw, open_away,
                 close_home, close_draw, close_away,
                 timestamp) -> list:
        """
        Evaluate a match for trading opportunities.
        
        Args:
            All odds are (price, bookie_name) tuples or None.
            
        Returns:
            List of Position objects for identified trades.
        """
        positions = []
        
        outcomes = [
            ('home', open_home, close_home),
            ('draw', open_draw, close_draw),
            ('away', open_away, close_away),
        ]
        
        for outcome, open_odds, close_odds in outcomes:
            if open_odds is None or close_odds is None:
                continue
            
            open_p, open_bk = open_odds if isinstance(open_odds, tuple) else (open_odds, '')
            close_p, close_bk = close_odds if isinstance(close_odds, tuple) else (close_odds, '')
            
            if not (self.config['min_odds'] <= open_p <= self.config['max_odds']):
                continue
            
            # We need odds to shorten (open > close) for back-to-lay profit
            if open_p <= close_p:
                continue
            
            swing = (open_p / close_p - 1) * 100
            if swing < self.config['min_swing_pct']:
                continue
            
            # Check capacity
            if self.portfolio.open_count >= self.config['max_concurrent']:
                continue
            
            # Place the trade
            pos = Position(
                match_id=match_id,
                league=league,
                home_team=home,
                away_team=away,
                match_date=match_date,
                outcome=outcome,
                open_time=timestamp,
                open_odds=open_p,
                back_stake=self.config['stake_per_trade'],
            )
            positions.append(pos)
        
        return positions
    
    def close_position(self, pos: Position, lay_odds: float, timestamp: str):
        """Close an open position at given lay odds."""
        self.portfolio.close_position(pos, lay_odds, timestamp)
    
    def report(self):
        """Print current state."""
        print(f"  Equity: £{self.portfolio.equity:,.2f}  |  "
              f"Open: {self.portfolio.open_count}  |  "
              f"Trades: {len(self.portfolio.closed_trades)}  |  "
              f"P&L: £{self.portfolio.total_pnl:+,.2f}  |  "
              f"DD: {self.portfolio.drawdown_pct:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════

class SimEngine:
    """
    Replays historical time-series odds day-by-day.
    
    Each trading day:
      1. 'Morning' — check opening odds for today's matches
      2. 'Evening' — check closing odds (matches have finished)
      3. Place back bets on morning odds
      4. Close positions with evening odds
    """
    
    def __init__(self, bot: TradingBot, days: int = 21):
        self.bot = bot
        self.days = days
        self.df = None
    
    def load_data(self):
        """Load and prepare time-series data."""
        import pandas as pd
        
        print("Loading time-series odds data...")
        basic = ['match_id','match_date','match_time','score_home','score_away']
        time_cols = []
        for outcome in ['home','draw','away']:
            for t in [71, 0]:
                for b in range(1, 9):
                    time_cols.append(f'{outcome}_b{b}_{t}')
        
        df = pd.read_csv(
            DATA_DIR / 'odds_series.csv.gz',
            compression='gzip', encoding='latin1',
            usecols=basic + time_cols
        )
        df = df.dropna(subset=['score_home', 'score_away'])
        df['match_date'] = pd.to_datetime(df['match_date'])
        df = df.sort_values('match_date').reset_index(drop=True)
        
        print(f"  {len(df):,} matches loaded ({df['match_date'].min().date()} to {df['match_date'].max().date()})")
        self.df = df
        return self
    
    def _best_odds(self, row, outcome, t):
        """Get best odds across 8 bookmakers for outcome at time t."""
        import pandas as pd
        cols = [f'{outcome}_b{b}_{t}' for b in range(1, 9)]
        vals = {f'b{b}': row.get(c) for b, c in enumerate(cols, 1)}
        best = None
        best_bk = None
        for bk, v in vals.items():
            if pd.notna(v) and (best is None or v > best):
                best = v
                best_bk = bk
        if best is not None:
            return (best, best_bk)
        return None
    
    def run(self):
        """Run the simulation day by day."""
        import pandas as pd
        
        if self.df is None:
            self.load_data()
        
        df = self.df
        unique_dates = sorted(df['match_date'].dt.date.unique())
        
        # Limit to requested number of days
        unique_dates = unique_dates[:self.days]
        
        print(f"\n{'='*60}")
        print(f"BACK-TO-LAY SIMULATION — {self.days} trading days")
        print(f"Starting bankroll: £{self.bot.portfolio.bankroll:,.2f}")
        print(f"Stake per trade:   £{CONFIG['stake_per_trade']}")
        print(f"Min swing:         {CONFIG['min_swing_pct']}%")
        print(f"{'='*60}\n")
        
        total_placed = 0
        total_closed = 0
        
        for day_idx, day in enumerate(unique_dates):
            day_matches = df[df['match_date'].dt.date == day]
            
            if len(day_matches) == 0:
                continue
            
            day_str = day.isoformat()
            day_pnl_before = self.bot.portfolio.total_pnl
            
            # ── OPENING: Place back bets on today's matches ──
            for _, row in day_matches.iterrows():
                match_id = int(row['match_id'])
                
                open_home = self._best_odds(row, 'home', 71)
                open_draw = self._best_odds(row, 'draw', 71)
                open_away = self._best_odds(row, 'away', 71)
                close_home = self._best_odds(row, 'home', 0)
                close_draw = self._best_odds(row, 'draw', 0)
                close_away = self._best_odds(row, 'away', 0)
                
                if not all([open_home, open_draw, open_away, close_home, close_draw, close_away]):
                    continue
                
                positions = self.bot.evaluate(
                    match_id=match_id,
                    league='',
                    home='',
                    away='',
                    match_date=day_str,
                    open_home=open_home, open_draw=open_draw, open_away=open_away,
                    close_home=close_home, close_draw=close_draw, close_away=close_away,
                    timestamp=f'{day_str}T00:00:00',
                )
                
                for pos in positions:
                    self.bot.portfolio.open_position(pos)
                    total_placed += 1
            
            # ── CLOSING: Close all open positions at closing odds ──
            for _, row in day_matches.iterrows():
                match_id = int(row['match_id'])
                close_home = self._best_odds(row, 'home', 0)
                close_draw = self._best_odds(row, 'draw', 0)
                close_away = self._best_odds(row, 'away', 0)
                
                close_odds_map = {}
                if close_home: close_odds_map['home'] = close_home[0]
                if close_draw: close_odds_map['draw'] = close_draw[0]
                if close_away: close_odds_map['away'] = close_away[0]
                
                for pos in list(self.bot.portfolio.positions):
                    if pos.status != 'open':
                        continue
                    if int(pos.match_id) != match_id:
                        continue
                    
                    lay_odds = close_odds_map.get(pos.outcome)
                    if lay_odds and lay_odds > 0 and lay_odds > 1.0:
                        self.bot.close_position(pos, lay_odds, f'{day_str}T23:59:00')
                        total_closed += 1
            
            # ── Daily report ──
            day_pnl = self.bot.portfolio.total_pnl - day_pnl_before
            if (day_idx + 1) % 3 == 0 or day_idx == 0 or day_idx == len(unique_dates) - 1:
                print(f"  Day {day_idx+1:2d}/{len(unique_dates)} ({day_str}): "
                      f"placed={len(day_matches)} "
                      f"trades={self.bot.portfolio.open_count} "
                      f"closed={total_closed} "
                      f"day P&L={day_pnl:+.2f} "
                      f"equity=£{self.bot.portfolio.equity:,.2f}")
        
        # ── Final report ──
        self.bot.portfolio.summary()
        
        # Save results
        results = {
            'config': CONFIG,
            'starting_bankroll': self.bot.portfolio.bankroll,
            'final_equity': self.bot.portfolio.equity,
            'total_pnl': self.bot.portfolio.total_pnl,
            'trades_placed': total_placed,
            'trades_closed': total_closed,
            'winning_trades': sum(1 for t in self.bot.portfolio.closed_trades if t.pnl > 0),
            'max_drawdown_pct': self.bot.portfolio.drawdown_pct,
            'daily_pnl': self.bot.portfolio.daily_pnl,
            'run_date': datetime.now().isoformat(),
            'days_simulated': len(unique_dates),
        }
        
        with open(OUT_DIR / 'simulation_result.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save trades CSV
        import csv
        with open(OUT_DIR / 'simulation_trades.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['match_date','outcome','open_odds','lay_odds','back_stake',
                       'lay_stake','pnl','status'])
            for t in self.bot.portfolio.closed_trades:
                w.writerow([t.match_date, t.outcome, t.open_odds, t.lay_odds,
                          t.back_stake, t.lay_stake, t.pnl, t.status])
        
        print(f"  Results saved to {OUT_DIR}/")
        return results


# ═══════════════════════════════════════════════════════════════════
# LIVE MODE (Betfair API stub)
# ═══════════════════════════════════════════════════════════════════

class LiveEngine:
    """
    Connects to real Betfair exchange via betfairlightweight.
    
    To use:
      1. pip install betfairlightweight
      2. Create app key at https://developer.betfair.com
      3. Run: python3 betfair_bot.py --mode live
    """
    
    def __init__(self, bot: TradingBot):
        self.bot = bot
        self.client = None
    
    def connect(self, username: str = None, password: str = None, app_key: str = None):
        """Connect to Betfair API."""
        username = username or os.environ.get('BF_USERNAME', '')
        password = password or os.environ.get('BF_PASSWORD', '')
        app_key = app_key or os.environ.get('BF_APP_KEY', '')
        
        if not all([username, password, app_key]):
            print("❌ Betfair credentials required.")
            print("   Set env vars: BF_USERNAME, BF_PASSWORD, BF_APP_KEY")
            print("   Or pass: --username --password --app-key")
            return False
        
        try:
            from betfairlightweight import APIClient
            self.client = APIClient(username, password, app_key=app_key)
            self.client.login()
            print(f"  ✅ Connected to Betfair as {username}")
            return True
        except Exception as e:
            print(f"  ❌ Connection failed: {e}")
            return False
    
    def run_loop(self, interval: int = 300):
        """Continuous monitoring loop (live mode)."""
        if not self.client:
            print("Not connected. Run connect() first.")
            return
        
        print("\nStarting live trading loop...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                self._tick()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            self.bot.portfolio.summary()
    
    def _tick(self):
        """One polling tick."""
        ts = datetime.now().isoformat()
        
        # Get soccer markets
        try:
            markets = self.client.betting.list_market_catalogue(
                filter={'eventTypeIds': ['1']},  # Soccer
                max_results=200,
                market_projection=['RUNNER_DESCRIPTION', 'MARKET_START_TIME'],
            )
        except Exception as e:
            print(f"  [{ts[:19]}] API error: {e}")
            return
        
        print(f"  [{ts[:19]}] {len(markets)} markets available")
        
        # Collect prices for each market
        for market in markets:
            try:
                prices = self.client.betting.list_market_book(
                    market_ids=[market.market_id],
                    price_projection={'priceData': ['EX_BEST_OFFERS']},
                )
                if not prices:
                    continue
                
                book = prices[0]
                runners = book.runners
                
                # Build back/lay prices for each runner
                for runner in runners:
                    back_prices = [p.price for p in (runner.ex.available_to_back or [])]
                    lay_prices = [p.price for p in (runner.ex.available_to_lay or [])]
                    
                    if not back_prices or not lay_prices:
                        continue
                    
                    best_back = back_prices[0]  # Highest back price
                    best_lay = lay_prices[-1] if lay_prices else 0  # Lowest lay price
                    
                    if best_lay <= 0 or best_back <= 0:
                        continue
                    
                    # Check if back-to-lay spread is profitable
                    ratio = best_back / best_lay
                    if ratio >= 1 + CONFIG['min_swing_pct'] / 100:
                        print(f"    ⚡ TRADE: {runner.description} "
                              f"back={best_back:.2f} lay={best_lay:.2f} "
                              f"ratio={ratio:.3f}")
                        
                        # Place back bet (in production, use place_orders)
                        # self.client.betting.place_orders(...)
            
            except Exception as e:
                pass


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Betfair Trading Bot')
    parser.add_argument('--mode', choices=['simulate', 'live'], default='simulate')
    parser.add_argument('--bankroll', type=float, default=1000)
    parser.add_argument('--days', type=int, default=21)
    parser.add_argument('--stake', type=float, default=50)
    parser.add_argument('--min-swing', type=float, default=3.0)
    parser.add_argument('--max-concurrent', type=int, default=20)
    # Live mode args
    parser.add_argument('--username', default='')
    parser.add_argument('--password', default='')
    parser.add_argument('--app-key', default='')
    
    args = parser.parse_args()
    
    CONFIG['stake_per_trade'] = args.stake
    CONFIG['min_swing_pct'] = args.min_swing
    CONFIG['max_concurrent'] = args.max_concurrent
    
    portfolio = Portfolio(bankroll=args.bankroll)
    bot = TradingBot(portfolio, CONFIG)
    
    if args.mode == 'simulate':
        print(SEP)
        print("BETFAIR TRADING BOT — SIMULATION MODE")
        print(SEP)
        print(f"  Bankroll: £{args.bankroll:,.2f}")
        print(f"  Days:     {args.days}")
        print(f"  Stake:    £{CONFIG['stake_per_trade']}")
        print(f"  Min swing:{CONFIG['min_swing_pct']}%")
        
        engine = SimEngine(bot, days=args.days)
        engine.load_data()
        engine.run()
        
    elif args.mode == 'live':
        print(SEP)
        print("BETFAIR TRADING BOT — LIVE MODE")
        print(SEP)
        
        engine = LiveEngine(bot)
        if engine.connect(args.username, args.password, args.app_key):
            engine.run_loop()


if __name__ == '__main__':
    main()
