#!/usr/bin/env python3
"""
Paper Trader — Pre-Match Consensus Strategy (first-seen tracking)
================================================================
Simulates value betting against live odds from The Odds API.
Uses FIRST-SEEN odds only (cached) to match the validated back-test approach.
Optimised: edge ≥8%, odds 1.3-2.5, £10 flat stake.

Strategy:
  - When a match first appears in the API, record all bookmaker odds = "opening odds"
  - For value detection: use ONLY the first-seen odds (never re-scan)
  - Auto-settle via The Odds API /scores endpoint
  - Agent can manually settle via web search if needed

Usage:
  python3 paper_trader.py              # one-shot poll + settle
  python3 paper_trader.py --watch       # continuous loop
  python3 paper_trader.py --report      # portfolio + pending results
  python3 paper_trader.py --settle <home> <away> <hscore> <ascore>
  python3 paper_trader.py --odds-cache  # show cached opening odds

Requires: ODDS_API_KEY env var
"""

import json, os, sys, time, csv
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional
import statistics

# ── Config ──
BANKROLL = 1000.0
STAKE_FRAC = 0.02
MIN_EDGE_PCT = 8.0
MIN_ODDS = 1.3
MAX_ODDS = 2.5
MIN_BOOKMAKERS = 3
POLL_INTERVAL = 600

OUT_DIR = Path('/home/burley/football-ml')
OUT_DIR.mkdir(exist_ok=True)
TRADE_LOG = OUT_DIR / 'paper_trades.csv'
PORTFOLIO_FILE = OUT_DIR / 'paper_portfolio.json'
ODDS_CACHE_FILE = OUT_DIR / 'first_seen_odds.json'
SEP = "=" * 60


@dataclass
class PaperTrade:
    timestamp: str
    match_id: str
    sport: str
    league: str
    home_team: str
    away_team: str
    commence_time: str
    outcome: str
    consensus_odds: float
    best_odds: float
    best_bookie: str
    edge_pct: float
    n_bookmakers: int
    stake: float
    point: Optional[float] = None          # Total line for over/under (e.g. 2.5)
    won: Optional[bool] = None
    profit: Optional[float] = None
    settled: bool = False


# ═══════════════════════════════════════════════════════════════════
# FIRST-SEEN ODDS CACHE
# ═══════════════════════════════════════════════════════════════════

class OddsCache:
    """
    Stores the first odds seen for each match across all bookmakers.
    Once recorded, never updated — these are the "opening odds" for our purposes.
    """
    
    def __init__(self):
        self.data = {}  # {match_id: {home_team, away_team, sport_key, 
                        #             first_seen, commence_time, 
                        #             odds: {outcome: {bookie: price}}}}
        self._load()
    
    def record(self, match: dict):
        """Record first-seen odds for a match if we haven't seen it before."""
        mid = match.get('id', '')
        if not mid or mid in self.data:
            return False  # Already recorded
        
        home = match.get('home_team', '')
        away = match.get('away_team', '')
        bookmakers = match.get('bookmakers', [])
        
        if len(bookmakers) < MIN_BOOKMAKERS:
            return False
        
        odds = {'home': {}, 'draw': {}, 'away': {}}
        totals = {}  # {point: {'over': {bookie: price}, 'under': {bookie: price}}}
        for bk in bookmakers:
            name = bk.get('title', '?')
            taken_first_h2h = False
            for market in bk.get('markets', []):
                if market.get('key') == 'h2h':
                    if taken_first_h2h:
                        continue
                    taken_first_h2h = True
                    for o in market.get('outcomes', []):
                        p = o.get('price', 0)
                        if p <= 0: continue
                        on = o.get('name', '')
                        if on == home: odds['home'][name] = p
                        elif on == away: odds['away'][name] = p
                        elif on == 'Draw': odds['draw'][name] = p
                elif market.get('key') == 'totals':
                    point = market.get('outcomes', [{}])[0].get('point', 2.5) if market.get('outcomes') else 2.5
                    if point not in totals:
                        totals[point] = {'over': {}, 'under': {}}
                    for o in market.get('outcomes', []):
                        p = o.get('price', 0)
                        if p <= 0: continue
                        on = o.get('name', '').lower()
                        if 'over' in on:
                            totals[point]['over'][name] = p
                        elif 'under' in on:
                            totals[point]['under'][name] = p

        # Only record if we got enough bookmaker data
        if not odds['home'] and not odds['draw'] and not odds['away']:
            return False
        
        self.data[mid] = {
            'home_team': home,
            'away_team': away,
            'sport_key': match.get('sport_key', ''),
            'sport_title': match.get('sport_title', ''),
            'first_seen': datetime.now().isoformat(),
            'commence_time': match.get('commence_time', ''),
            'odds': odds,
            'totals': totals,
            'surface': detect_surface(match.get('sport_key', ''), match.get('sport_title', '')),
        }
        self._save()
        return True
    
    def get_odds(self, match_id: str, outcome: str) -> dict:
        """Get first-seen odds for a specific match+outcome. Returns {bookie: price}."""
        entry = self.data.get(match_id)
        if not entry:
            return {}
        return entry.get('odds', {}).get(outcome, {})
    
    def get_match_info(self, match_id: str) -> Optional[dict]:
        return self.data.get(match_id)
    
    def has(self, match_id: str) -> bool:
        return match_id in self.data
    
    def count(self) -> int:
        return len(self.data)
    
    def untraded_matches(self, portfolio) -> list:
        """Return cached matches that haven't been bet on yet."""
        bet_ids = {t.match_id for t in portfolio.trades}
        return [mid for mid in self.data if mid not in bet_ids]
    
    def _load(self):
        if ODDS_CACHE_FILE.exists():
            try:
                with open(ODDS_CACHE_FILE) as f:
                    self.data = json.load(f)
                print(f"  Loaded {len(self.data)} first-seen odds records")
            except (json.JSONDecodeError, IOError):
                self.data = {}
    
    def _save(self):
        with open(ODDS_CACHE_FILE, 'w') as f:
            json.dump(self.data, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO
# ═══════════════════════════════════════════════════════════════════

class PaperPortfolio:
    def __init__(self, bankroll: float = BANKROLL):
        self.bankroll = bankroll
        self.peak = bankroll
        self.trades = []
        self.load()

    @property
    def equity(self):
        return self.bankroll + sum(t.profit for t in self.trades if t.profit is not None)

    def place_bet(self, trade: PaperTrade):
        self.trades.append(trade)
        self._append_trade(trade)
        self._save_portfolio()

    def settle(self, trade: PaperTrade, won: bool):
        trade.won = won
        trade.profit = round(trade.stake * (trade.best_odds - 1), 2) if won else -trade.stake
        trade.settled = True
        self._save_all_trades()
        self._save_portfolio()

    def find_unsettled(self, home: str, away: str) -> list:
        h, a = home.lower(), away.lower()
        return [t for t in self.trades if not t.settled
                and (h in t.home_team.lower() or h in t.away_team.lower())
                and (a in t.away_team.lower() or a in t.home_team.lower())]

    def unsettled_count(self):
        return sum(1 for t in self.trades if not t.settled)

    def summary(self):
        settled = [t for t in self.trades if t.settled]
        unsettled = self.unsettled_count()

        print(f"\n{SEP}")
        print("📊 PORTFOLIO")
        print(SEP)
        print(f"  Bankroll:  £{self.bankroll:,.2f}")
        print(f"  Equity:    £{self.equity:,.2f}")
        print(f"  P&L:       £{self.total_pnl:+,.2f}")
        print(f"  Return:    {(self.equity/self.bankroll-1)*100:+.2f}%")
        print(f"  Bets:      {len(settled)} settled, {unsettled} unsettled")

        if settled:
            wins = sum(1 for t in settled if t.won)
            total_staked = sum(t.stake for t in settled)
            print(f"  Win rate:  {wins}/{len(settled)} ({wins/len(settled)*100:.1f}%)")
            if total_staked > 0:
                print(f"  ROI:       {self.total_pnl/total_staked*100:.2f}%")
            print(f"\n  Last {min(5, len(settled))} settled:")
            for t in settled[-5:]:
                sym = "✅" if t.won else "❌"
                print(f"    {sym} {t.home_team[:20]:20s} vs {t.away_team[:20]:20s} "
                      f"| {t.outcome:5s} @ {t.best_odds:.2f} | £{t.profit:+.2f}")
        else:
            print(f"  No settled trades yet")

    @property
    def total_pnl(self):
        return sum(t.profit for t in self.trades if t.profit is not None)

    def load(self):
        if not TRADE_LOG.exists():
            return
        with open(TRADE_LOG) as f:
            for row in csv.DictReader(f):
                t = PaperTrade(
                    timestamp=row['timestamp'], match_id=row['match_id'],
                    sport=row.get('sport', ''), league=row.get('league', ''),
                    home_team=row['home_team'], away_team=row['away_team'],
                    commence_time=row['commence_time'], outcome=row['outcome'],
                    consensus_odds=float(row['consensus_odds']),
                    best_odds=float(row['best_odds']),
                    best_bookie=row['best_bookie'],
                    edge_pct=float(row['edge_pct']),
                    n_bookmakers=int(row['n_bookmakers']),
                    stake=float(row['stake']),
                    point=float(row['point']) if row.get('point', '') else None,
                    won={'True': True, 'False': False, '': None}.get(row.get('won', ''), None),
                    profit=float(row['profit']) if row.get('profit', '') else None,
                    settled=row.get('settled', 'False') == 'True',
                )
                self.trades.append(t)
        self.peak = self.bankroll
        for i, t in enumerate(self.trades):
            if t.profit is not None:
                eq = self.bankroll + sum(tt.profit for tt in self.trades[:i+1] if tt.profit is not None)
                if eq > self.peak:
                    self.peak = eq

    def _append_trade(self, t):
        fn = ['timestamp','match_id','sport','league','home_team','away_team',
              'commence_time','outcome','consensus_odds','best_odds','best_bookie',
              'edge_pct','n_bookmakers','stake','point','won','profit','settled']
        exists = TRADE_LOG.exists()
        with open(TRADE_LOG, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fn)
            if not exists: w.writeheader()
            w.writerow(self._row(t))

    def _save_all_trades(self):
        fn = ['timestamp','match_id','sport','league','home_team','away_team',
              'commence_time','outcome','consensus_odds','best_odds','best_bookie',
              'edge_pct','n_bookmakers','stake','point','won','profit','settled']
        with open(TRADE_LOG, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            for t in self.trades:
                w.writerow(self._row(t))

    def _row(self, t):
        return {
            'timestamp': t.timestamp, 'match_id': t.match_id,
            'sport': t.sport, 'league': t.league,
            'home_team': t.home_team, 'away_team': t.away_team,
            'commence_time': t.commence_time, 'outcome': t.outcome,
            'consensus_odds': f"{t.consensus_odds:.2f}",
            'best_odds': f"{t.best_odds:.2f}",
            'best_bookie': t.best_bookie,
            'edge_pct': f"{t.edge_pct:.1f}",
            'n_bookmakers': str(t.n_bookmakers),
            'stake': f"{t.stake:.2f}",
            'point': f"{t.point:.1f}" if t.point is not None else '',
            'won': str(t.won) if t.won is not None else '',
            'profit': f"{t.profit:.2f}" if t.profit is not None else '',
            'settled': str(t.settled),
        }

    def _save_portfolio(self):
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump({
                'bankroll': self.bankroll, 'peak': self.peak,
                'equity': self.equity, 'trades': len(self.trades),
                'updated': datetime.now().isoformat(),
            }, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
# ODDS API
# ═══════════════════════════════════════════════════════════════════

class OddsAPI:
    def __init__(self, key):
        self.key = key
        self.base = 'https://api.the-odds-api.com/v4'

    def fetch(self, url):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PaperTrader/1.0'})
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code not in (422,):
                print(f"  ⚠️  API {e.code}: {e.reason}")
            return []
        except Exception as e:
            return []

    def get_matches(self, sport='soccer', regions='uk,eu', markets='h2h'):
        return self.fetch(f'{self.base}/sports/{sport}/odds'
                          f'?apiKey={self.key}&regions={regions}'
                          f'&markets={markets}&oddsFormat=decimal')

    def get_all_sports(self):
        """Fetch all available sports to find active leagues."""
        return self.fetch(f'{self.base}/sports/?apiKey={self.key}')

    def get_scores(self, sport_key):
        return self.fetch(f'{self.base}/sports/{sport_key}/scores'
                          f'?apiKey={self.key}&daysFrom=3')


# ═══════════════════════════════════════════════════════════════════
# SURFACE DETECTOR — Tennis tournament surface lookup
# ═══════════════════════════════════════════════════════════════════

# Known tennis tournaments and their surfaces
# Format: patterns that appear in sport_key or sport_title → surface
SURFACE_PATTERNS = {
    'clay': [
        'french open', 'roland garros', 'monte carlo', 'madrid', 'rome',
        'barcelona', 'hamburg', 'munich', 'geneva', 'lyon', 'gstaad',
        'bastad', 'umag', 'kitzbuhel', 'stuttgart', 'rio open',
        'buenos aires', 'santiago', 'cordoba', 'bogota', 'sao paulo',
        'marrakech', 'casablanca', 'estoril', 'bucharest', 'belgrade',
        'paris masters', 'savannah', 'aix en provence', 'heilbronn',
        'clay', 'red clay',
    ],
    'grass': [
        'wimbledon', 'queen\'s club', 'halle', 'eastbourne',
        's-hertogenbosch', 'rosmalen', 'mallorca', 'newport',
        'nottingham', 'grass', 'surbiton', 'stuttgart grass',
    ],
    'hard': [
        'us open', 'australian open', 'indian wells', 'miami',
        'cincinnati', 'canada masters', 'toronto', 'montreal',
        'shanghai', 'paris masters indoor', 'beijing', 'tokyo',
        'vienna', 'basel', 'stockholm', 'antwerp', 'moscow',
        'washington', 'acapulco', 'dubai', 'doha', 'abu dhabi',
        'zhuhai', 'chengdu', 'astana', 'sofia', 'marseille',
        'montpellier', 'milan', 'london', 'hard', 'atp finals',
    ],
}

# Which surfaces we should value-bet
VALIDATED_TENNIS = {
    'atp': ['clay'],      # ATP clay only
    'wta': ['hard'],      # WTA hard only
}


def detect_surface(sport_key: str, sport_title: str) -> str:
    """
    Detect tennis court surface from sport key and title.
    Returns 'clay', 'grass', 'hard', or 'unknown'.
    """
    text = f"{sport_key.lower()} {sport_title.lower()}"
    
    for surface, patterns in SURFACE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in text:
                return surface
    
    # Default for ATP/WTA events if nothing matched
    if 'atp' in text or 'wta' in text:
        return 'hard'  # Most non-clay/grass ATP/WTA events are hard
    
    return 'unknown'


def should_bet_tennis(sport_key: str, sport_title: str) -> bool:
    """
    Check if a tennis match should be value-bet based on surface + gender.
    Returns True if this match type is validated.
    """
    surface = detect_surface(sport_key, sport_title)
    text = f"{sport_key.lower()} {sport_title.lower()}"
    
    gender = 'atp' if 'atp' in text else 'wta' if 'wta' in text else None
    if gender is None:
        return False
    
    allowed = VALIDATED_TENNIS.get(gender, [])
    return surface in allowed

def record_new_matches(matches: list, cache: OddsCache):
    """Record first-seen odds for any new matches."""
    new = 0
    for m in matches:
        if cache.record(m):
            new += 1
    return new


def scan_cached_matches(cache: OddsCache, portfolio: PaperPortfolio) -> list:
    """
    Scan cached first-seen odds for value bets.
    Only considers matches we haven't already bet on.
    """
    found = []
    
    for mid in cache.untraded_matches(portfolio):
        entry = cache.get_match_info(mid)
        if not entry:
            continue
        
        odds = entry.get('odds', {})
        home = entry['home_team']
        away = entry['away_team']
        sport_key = entry.get('sport_key', '')
        sport_title = entry.get('sport_title', '')
        commence = entry.get('commence_time', '')
        
        for outcome in ['home', 'draw', 'away']:
            bookie_odds = odds.get(outcome, {})
            prices = list(bookie_odds.values())
            
            if len(prices) < MIN_BOOKMAKERS:
                continue
            
            # Tennis surface filtering
            if 'tennis' in sport_key or 'tennis' in sport_title.lower():
                # Skip draws (tennis has no draws)
                if outcome == 'draw':
                    continue
                # Only bet validated tennis surfaces
                surface = entry.get('surface', 'unknown')
                text = f"{sport_key.lower()} {sport_title.lower()}"
                gender = 'atp' if 'atp' in text else 'wta' if 'wta' in text else None
                allowed = VALIDATED_TENNIS.get(gender, [])
                if surface not in allowed:
                    continue
            
            consensus = statistics.mean(prices)
            best = max(prices)
            
            if best <= consensus:
                continue
            if not (MIN_ODDS <= best <= MAX_ODDS):
                continue
            
            edge = (best / consensus - 1) * 100
            if edge < MIN_EDGE_PCT:
                continue
            
            stake = round(portfolio.equity * STAKE_FRAC, 2)
            if stake < 1:
                continue
            
            best_bookie = max(bookie_odds, key=bookie_odds.get)
            
            t = PaperTrade(
                timestamp=datetime.now().isoformat(),
                match_id=mid,
                sport=sport_title,
                league=sport_key,
                home_team=home,
                away_team=away,
                commence_time=commence[:19],
                outcome=outcome,
                consensus_odds=round(consensus, 2),
                best_odds=round(best, 2),
                best_bookie=best_bookie,
                edge_pct=round(edge, 1),
                n_bookmakers=len(prices),
                stake=stake,
            )
            portfolio.place_bet(t)
            found.append(t)
    
    return found


# ═══════════════════════════════════════════════════════════════════
# OVER/UNDER 2.5 SCANNER
# ═══════════════════════════════════════════════════════════════════

# From backtest: Over 2.5 only, edge ≥8%, 6+ bookmakers → +13.49% ROI
OU_MIN_EDGE = 8.0
OU_MIN_BOOKMAKERS = 6
OU_MIN_ODDS = 1.3
OU_MAX_ODDS = 3.0
OU_POINT = 2.5  # Standard total goals line


def scan_totals(cache: OddsCache, portfolio: PaperPortfolio) -> list:
    """
    Scan cached first-seen totals (over/under) odds for value bets.
    Validated for Over 2.5 only at ≥8% edge with 6+ bookmakers.
    """
    found = []

    for mid in cache.untraded_matches(portfolio):
        entry = cache.get_match_info(mid)
        if not entry:
            continue

        totals = entry.get('totals', {})
        if OU_POINT not in totals:
            continue

        point_data = totals[OU_POINT]
        home = entry['home_team']
        away = entry['away_team']
        sport_key = entry.get('sport_key', '')
        sport_title = entry.get('sport_title', '')
        commence = entry.get('commence_time', '')

        # Only soccer has validated over/under data
        if 'soccer' not in sport_key and 'soccer' not in sport_title.lower():
            continue

        over_odds = point_data.get('over', {})
        under_odds = point_data.get('under', {})

        # Check Over 2.5 (validated direction)
        over_prices = list(over_odds.values())
        if len(over_prices) >= OU_MIN_BOOKMAKERS:
            consensus = statistics.mean(over_prices)
            best = max(over_prices)

            if best > consensus and OU_MIN_ODDS <= best <= OU_MAX_ODDS:
                edge = (best / consensus - 1) * 100
                if edge >= OU_MIN_EDGE:
                    stake = round(portfolio.equity * STAKE_FRAC, 2)
                    if stake >= 1:
                        best_bookie = max(over_odds, key=over_odds.get)
                        t = PaperTrade(
                            timestamp=datetime.now().isoformat(),
                            match_id=mid,
                            sport=f"Over {OU_POINT} - {sport_title}",
                            league=sport_key,
                            home_team=home,
                            away_team=away,
                            commence_time=commence[:19],
                            outcome='over_2.5',
                            consensus_odds=round(consensus, 2),
                            best_odds=round(best, 2),
                            best_bookie=best_bookie,
                            edge_pct=round(edge, 1),
                            n_bookmakers=len(over_prices),
                            stake=stake,
                            point=OU_POINT,
                        )
                        portfolio.place_bet(t)
                        found.append(t)

    return found


def auto_settle(portfolio, api_key):
    """Try to settle via The Odds API /scores endpoint."""
    unsettled = [t for t in portfolio.trades if not t.settled]
    if not unsettled:
        return []

    api = OddsAPI(api_key)
    settled = []

    groups = {}
    for t in unsettled:
        sk = t.league if t.league and t.league.startswith('soccer_') else 'soccer'
        groups.setdefault(sk, []).append(t)

    for sk, trades in groups.items():
        # Determine which sport key to use for scores
        if sk.startswith('tennis_') or sk.startswith('soccer_'):
            score_sport = sk
        elif sk == 'tennis':
            score_sport = 'tennis'
        else:
            score_sport = 'soccer'
        results = api.get_scores(score_sport)
        if not results:
            continue

        scores_map = {}
        for match in results:
            if not match.get('completed') and not match.get('scores'):
                continue
            sc = match.get('scores') or []
            ht, at = match.get('home_team', ''), match.get('away_team', '')
            hs, as_ = None, None
            for s in sc:
                try:
                    v = int(float(s.get('score', -1)))
                    if s.get('name') == ht: hs = v
                    elif s.get('name') == at: as_ = v
                except: pass
            if hs is not None and as_ is not None:
                scores_map[(ht, at)] = (hs, as_)

        for t in trades:
            key = (t.home_team, t.away_team)
            if key in scores_map:
                hs, as_ = scores_map[key]
                # Over/Under settlement
                if t.outcome == 'over_2.5':
                    total_goals = hs + as_
                    point = t.point or 2.5
                    won = total_goals > point
                elif t.outcome == 'under_2.5':
                    total_goals = hs + as_
                    point = t.point or 2.5
                    won = total_goals < point
                # Tennis: no draws
                elif t.sport.startswith('ATP') or t.sport.startswith('WTA') or 'tennis' in t.league:
                    won = (t.outcome == 'home' and hs > as_) or (t.outcome == 'away' and as_ > hs)
                # Standard h2h
                else:
                    won = {'home': hs > as_, 'draw': hs == as_, 'away': as_ > hs}[t.outcome]
                portfolio.settle(t, won)
                sym = "✅" if won else "❌"
                detail = f"{hs}-{as_}" if t.outcome in ('over_2.5', 'under_2.5') else f"{hs}-{as_}"
                print(f"  {sym} {t.home_team} {detail} {t.away_team} → {t.outcome} "
                      f"{'WON' if won else 'LOST'} (£{t.profit:+.2f})")
                settled.append(t)

    return settled


def show_pending(portfolio):
    unsettled = [t for t in portfolio.trades if not t.settled]
    if not unsettled:
        print(f"\n  ✅ All trades settled!")
        return

    print(f"\n{'─' * 55}")
    print(f"🔍 {len(unsettled)} PENDING RESULTS")
    print(f"{'─' * 55}")
    for i, t in enumerate(unsettled, 1):
        if t.outcome in ('over_2.5', 'under_2.5'):
            direction = f"{'OVER' if 'over' in t.outcome else 'UNDER'} {t.point:.1f}"
            settle_cmd = f"python3 paper_trader.py --settle \"{t.home_team}\" \"{t.away_team}\" <hscore> <ascore>"
        else:
            direction = t.outcome.upper()
            settle_cmd = f"python3 paper_trader.py --settle \"{t.home_team}\" \"{t.away_team}\" <hscore> <ascore>"
        print(f"\n  [{i}] {t.home_team} vs {t.away_team}")
        print(f"      Placed: {t.timestamp[:19]}  |  Match: {t.commence_time[:10]}")
        print(f"      Back {direction} @ {t.best_odds:.2f} ({t.best_bookie})")
        print(f"      Stake: £{t.stake:.2f}  |  Edge: {t.edge_pct:.1f}%")
        print(f"      → {settle_cmd}")
    print()


def show_odds_cache(cache: OddsCache, portfolio: PaperPortfolio):
    """Show the first-seen odds cache contents."""
    bet_ids = {t.match_id for t in portfolio.trades}
    print(f"\n{'─' * 55}")
    print(f"📝 FIRST-SEEN ODDS CACHE ({cache.count()} matches)")
    print(f"{'─' * 55}")
    
    # Show matches not yet bet on (awaiting scan)
    unscanned = [mid for mid in cache.data if mid not in bet_ids]
    if unscanned:
        print(f"\n  Unscanned ({len(unscanned)}):")
        for mid in sorted(unscanned)[:10]:
            e = cache.data[mid]
            print(f"    • {e['home_team'][:20]:20s} vs {e['away_team'][:20]:20s} "
                  f"| {e.get('sport_key','')}")
        if len(unscanned) > 10:
            print(f"    ... and {len(unscanned)-10} more")
    
    # Show scanned (already bet on)
    scanned = [mid for mid in cache.data if mid in bet_ids]
    if scanned:
        print(f"\n  Scanned ({len(scanned)}):")
        for mid in sorted(scanned)[:10]:
            e = cache.data[mid]
            t = next((t for t in portfolio.trades if t.match_id == mid), None)
            status = "✅ settled" if t and t.settled else "⏳ pending"
            print(f"    • {e['home_team'][:20]:20s} vs {e['away_team'][:20]:20s} | {status}")
    print()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    key = os.environ.get('ODDS_API_KEY', '')
    if not key:
        print("❌ ODDS_API_KEY not set. See: https://the-odds-api.com")
        sys.exit(1)

    import argparse
    p = argparse.ArgumentParser(description='Paper Trader — Opening Odds Consensus')
    p.add_argument('--watch', action='store_true', help='Continuous polling')
    p.add_argument('--report', action='store_true', help='Portfolio + pending')
    p.add_argument('--odds-cache', action='store_true', help='Show first-seen odds cache')
    p.add_argument('--settle', nargs=4, metavar=('HOME','AWAY','HGOALS','AGOALS'),
                   help='Settle: --settle "Team A" "Team B" 2 1')

    args = p.parse_args()

    if args.settle:
        home, away, hg, ag = args.settle
        portfolio = PaperPortfolio()
        matches = portfolio.find_unsettled(home, away)
        if not matches:
            print(f"  No unsettled trade for {home} vs {away}")
            return
        t = matches[0]
        try:
            hs, as_ = int(hg), int(ag)
        except ValueError:
            print("  Scores must be integers")
            return
        if t.outcome == 'over_2.5':
            won = (hs + as_) > (t.point or 2.5)
        elif t.outcome == 'under_2.5':
            won = (hs + as_) < (t.point or 2.5)
        else:
            won = {'home': hs > as_, 'draw': hs == as_, 'away': as_ > hs}[t.outcome]
        portfolio.settle(t, won)
        sym = "✅" if won else "❌"
        print(f"  {sym} Settled: {t.home_team} {hs}-{as_} {t.away_team} → "
              f"{'WON' if won else 'LOST'} (£{t.profit:+.2f})")
        portfolio.summary()
        return

    portfolio = PaperPortfolio()
    cache = OddsCache()

    if args.report:
        portfolio.summary()
        show_pending(portfolio)
        show_odds_cache(cache, portfolio)
        return

    if args.odds_cache:
        show_odds_cache(cache, portfolio)
        return

    # === One-shot poll ===
    api = OddsAPI(key)

    # 1. Settle finished trades
    n_settled = len(auto_settle(portfolio, key))
    if n_settled:
        print(f"  Auto-settled {n_settled} trades\n")

    # 2. Poll API for new matches (soccer + tennis)
    print(f"📡 Polling The Odds API...")
    all_matches = []
    
    for sport in ['soccer', 'tennis']:
        # Request both h2h and totals markets for over/under
        m = api.get_matches(sport, markets='h2h,totals')
        if m:
            all_matches.extend(m)
            print(f"  {sport}: {len(m)} matches")
    
    if not all_matches:
        print("  No matches returned")
    else:
        # Deduplicate by match_id
        seen = set()
        unique = []
        for m in all_matches:
            mid = m.get('id', '')
            if mid not in seen:
                seen.add(mid)
                unique.append(m)
        print(f"  Total: {len(unique)} unique matches")
        
        # Record first-seen odds for any NEW matches
        new = record_new_matches(unique, cache)
        if new:
            print(f"  Recorded first-seen odds for {new} new matches")
        else:
            print(f"  No new matches (cache has {cache.count()} total)")

    # 3. Scan cached matches for value bets (using first-seen odds)
    print(f"\n🔍 Scanning cached odds for value (odds {MIN_ODDS}-{MAX_ODDS}, edge ≥{MIN_EDGE_PCT}%)...")
    bets = scan_cached_matches(cache, portfolio)
    
    if bets:
        print(f"\n{'─' * 55}")
        print(f"⚡ {len(bets)} NEW VALUE BET{'S' if len(bets) > 1 else ''} (from first-seen odds)")
        print(f"{'─' * 55}")
        for b in bets:
            print(f"  {b.home_team:22s} vs {b.away_team:22s}")
            print(f"  Back {b.outcome.upper():5s} @ {b.best_odds:.2f} ({b.best_bookie})")
            print(f"  Edge: {b.edge_pct:.1f}%  |  Stake: £{b.stake:.2f}  |  "
                  f"{b.n_bookmakers} bookmakers\n")
    else:
        print(f"  No h2h value bets")
    
    # 4. Scan for over/under 2.5 value bets
    print(f"\n🔍 Scanning totals for Over {OU_POINT} value (edge ≥{OU_MIN_EDGE}%, ≥{OU_MIN_BOOKMAKERS} books)...")
    ou_bets = scan_totals(cache, portfolio)
    
    if ou_bets:
        print(f"\n{'─' * 55}")
        print(f"⚡ {len(ou_bets)} NEW OVER/UNDER BET{'S' if len(ou_bets) > 1 else ''}")
        print(f"{'─' * 55}")
        for b in ou_bets:
            print(f"  {b.home_team:22s} vs {b.away_team:22s}")
            print(f"  Back OVER {b.point:.1f} @ {b.best_odds:.2f} ({b.best_bookie})")
            print(f"  Edge: {b.edge_pct:.1f}%  |  Stake: £{b.stake:.2f}  |  "
                  f"{b.n_bookmakers} bookmakers\n")
    else:
        print(f"  No over/under value bets")
        untraded = len(cache.untraded_matches(portfolio))
        if untraded:
            print(f"  ({untraded} cached matches — none met totals criteria)")

    portfolio.summary()
    show_pending(portfolio)

    # === Watch mode ===
    if args.watch:
        print(f"\n👁️  Watch mode: polling every {POLL_INTERVAL}s...")
        poll = 1
        try:
            while True:
                time.sleep(POLL_INTERVAL)
                poll += 1
                now = datetime.now().strftime('%H:%M')
                print(f"[{now}] Poll #{poll}...")
                
                # Settle
                auto_settle(portfolio, key)
                
                # Record new matches only (don't re-scan old ones)
                for sport in ['soccer', 'tennis']:
                    m2 = api.get_matches(sport, markets='h2h,totals')
                    if m2:
                        new2 = record_new_matches(m2, cache)
                        if new2:
                            print(f"  {sport}: {new2} new matches recorded")
                            bets2 = scan_cached_matches(cache, portfolio)
                            if bets2:
                                print(f"  ⚡ {len(bets2)} new h2h bets from fresh odds!")
                                for b in bets2:
                                    print(f"     {b.home_team[:20]:20s} vs {b.away_team[:20]:20s} | "
                                          f"{b.outcome:5s} @ {b.best_odds:.2f}")
                            ou2 = scan_totals(cache, portfolio)
                            if ou2:
                                print(f"  ⚡ {len(ou2)} new over/under bets from fresh odds!")
                                for b in ou2:
                                    print(f"     {b.home_team[:20]:20s} vs {b.away_team[:20]:20s} | "
                                          f"Over {b.point:.1f} @ {b.best_odds:.2f}")
                
                if poll % 6 == 0:
                    portfolio.summary()
                    show_pending(portfolio)
        except KeyboardInterrupt:
            print(f"\n  Stopped after {poll} polls.")
            portfolio.summary()
            show_pending(portfolio)


if __name__ == '__main__':
    main()
