#!/usr/bin/env python3
"""
Over/Under 2.5 Goals Consensus Backtest
=========================================
Tests whether consensus pricing on over/under 2.5 goals
identifies value bets across multiple European leagues.

Two eras:
  2015-2019: BetBrain consensus (BbAv>2.5), max price (BbMx>2.5)
  2020-2025: Market avg consensus (Avg>2.5), Bet365 price (B365>2.5)
"""

import csv
import io
import urllib.request
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

SEP = "=" * 70
START_BANKROLL = 1000
STAKE_FRAC = 0.02
MIN_EDGE_PCT = 5.0
MIN_ODDS = 1.3
MAX_ODDS = 3.0
MIN_BOOKMAKERS = 3

# Leagues to test
LEAGUES = {
    'E0': 'England Premier League',
    'D1': 'Germany Bundesliga 1',
    'I1': 'Italy Serie A',
    'SP1': 'Spain La Liga',
    'F1': 'France Ligue 1',
}

# Seasons to test
SEASONS_OLD = [f"{y}{y+1}" for y in range(15, 20)]    # 2015-2019 (BetBrain era)
SEASONS_NEW = [f"{y}{y+1}" for y in range(20, 25)]    # 2020-2025 (individual era)

# URLs: format changed over time
URL_PATTERN_OLD = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
URL_PATTERN_NEW = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


@dataclass
class Bet:
    match_date: str
    league: str
    home: str
    away: str
    bet_type: str          # 'over' or 'under'
    odds: float
    consensus: float
    edge_pct: float
    total_goals: int
    result: str            # 'win' or 'loss'
    profit: float
    total_bookmakers: int  # How many bookmakers in consensus


@dataclass
class Result:
    bets: list = field(default_factory=list)
    bankroll: float = START_BANKROLL


def fetch_csv(url: str, max_retries=2) -> list:
    """Fetch a CSV from football-data.co.uk with retries."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=15)
            data = resp.read().decode('utf-8-sig')
            return list(csv.DictReader(io.StringIO(data)))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None


def process_betbrain_era(rows: list, league_name: str) -> list:
    """Process rows with BetBrain columns (2015-2019)."""
    bets = []
    for row in rows:
        try:
            # Need at least 3 bookmakers for meaningful consensus
            n_bookies = int(row.get('BbOU', '0') or '0')
            if n_bookies < MIN_BOOKMAKERS:
                continue

            total_goals = int(row.get('FTHG', '0') or '0') + int(row.get('FTAG', '0') or '0')

            # BetBrain consensus (average)
            cons_over = float(row.get('BbAv>2.5', '') or '0')
            cons_under = float(row.get('BbAv<2.5', '') or '0')
            if cons_over <= 0 or cons_under <= 0:
                continue

            # Best available price (maximum)
            best_over = float(row.get('BbMx>2.5', '') or '0')
            best_under = float(row.get('BbMx<2.5', '') or '0')

            # Check Over bet
            if best_over >= MIN_ODDS and best_over <= MAX_ODDS:
                implied_cons = 1.0 / cons_over
                implied_best = 1.0 / best_over
                edge = (implied_cons - implied_best) / implied_best * 100
                if edge >= MIN_EDGE_PCT:
                    outcome = 'win' if total_goals > 2.5 else 'loss'
                    stake = STAKE_FRAC
                    profit = stake * (best_over - 1) if outcome == 'win' else -stake
                    if profit != 0:
                        bets.append(Bet(
                            match_date=row.get('Date', ''),
                            league=league_name,
                            home=row.get('HomeTeam', ''),
                            away=row.get('AwayTeam', ''),
                            bet_type='over',
                            odds=best_over,
                            consensus=cons_over,
                            edge_pct=round(edge, 1),
                            total_goals=total_goals,
                            result=outcome,
                            profit=round(profit * 1000, 2),
                            total_bookmakers=n_bookies,
                        ))

            # Check Under bet
            if best_under >= MIN_ODDS and best_under <= MAX_ODDS:
                implied_cons = 1.0 / cons_under
                implied_best = 1.0 / best_under
                edge = (implied_cons - implied_best) / implied_best * 100
                if edge >= MIN_EDGE_PCT:
                    outcome = 'win' if total_goals < 2.5 else 'loss'
                    stake = STAKE_FRAC
                    profit = stake * (best_under - 1) if outcome == 'win' else -stake
                    if profit != 0:
                        bets.append(Bet(
                            match_date=row.get('Date', ''),
                            league=league_name,
                            home=row.get('HomeTeam', ''),
                            away=row.get('AwayTeam', ''),
                            bet_type='under',
                            odds=best_under,
                            consensus=cons_under,
                            edge_pct=round(edge, 1),
                            total_goals=total_goals,
                            result=outcome,
                            profit=round(profit * 1000, 2),
                            total_bookmakers=n_bookies,
                        ))

        except (ValueError, TypeError):
            continue

    return bets


def process_individual_era(rows: list, league_name: str) -> list:
    """Process rows with individual bookmaker odds (2020-2025).
    Uses Avg>2.5 as consensus, B365>2.5 as individual bookmaker price."""
    bets = []
    for row in rows:
        try:
            total_goals = int(row.get('FTHG', '0') or '0') + int(row.get('FTAG', '0') or '0')

            # Use market average as consensus
            cons_over = float(row.get('Avg>2.5', '') or '0')
            cons_under = float(row.get('Avg<2.5', '') or '0')
            if cons_over <= 0 or cons_under <= 0:
                # Try closing
                cons_over = float(row.get('AvgC>2.5', '') or '0')
                cons_under = float(row.get('AvgC<2.5', '') or '0')
                if cons_over <= 0 or cons_under <= 0:
                    continue

            # Bet365 price
            b365_over = float(row.get('B365>2.5', '') or '0')
            b365_under = float(row.get('B365<2.5', '') or '0')

            # Also try Pinnacle as alternative
            p_over = float(row.get('P>2.5', '') or '0')
            p_under = float(row.get('P<2.5', '') or '0')

            # Test each available bookmaker
            bookmakers = []
            if b365_over > 0 and b365_under > 0:
                bookmakers.append(('Bet365', b365_over, b365_under))
            if p_over > 0 and p_under > 0:
                bookmakers.append(('Pinnacle', p_over, p_under))

            for bk_name, bk_over, bk_under in bookmakers:
                # Over bet
                if bk_over >= MIN_ODDS and bk_over <= MAX_ODDS:
                    implied_cons = 1.0 / cons_over
                    implied_bk = 1.0 / bk_over
                    edge = (implied_cons - implied_bk) / implied_bk * 100
                    if edge >= MIN_EDGE_PCT:
                        outcome = 'win' if total_goals > 2.5 else 'loss'
                        stake = STAKE_FRAC
                        profit = stake * (bk_over - 1) if outcome == 'win' else -stake
                        if profit != 0:
                            bets.append(Bet(
                                match_date=row.get('Date', ''),
                                league=f"{league_name} [{bk_name}]",
                                home=row.get('HomeTeam', ''),
                                away=row.get('AwayTeam', ''),
                                bet_type='over',
                                odds=bk_over,
                                consensus=cons_over,
                                edge_pct=round(edge, 1),
                                total_goals=total_goals,
                                result=outcome,
                                profit=round(profit * 1000, 2),
                                total_bookmakers=0,
                            ))

                # Under bet
                if bk_under >= MIN_ODDS and bk_under <= MAX_ODDS:
                    implied_cons = 1.0 / cons_under
                    implied_bk = 1.0 / bk_under
                    edge = (implied_cons - implied_bk) / implied_bk * 100
                    if edge >= MIN_EDGE_PCT:
                        outcome = 'win' if total_goals < 2.5 else 'loss'
                        stake = STAKE_FRAC
                        profit = stake * (bk_under - 1) if outcome == 'win' else -stake
                        if profit != 0:
                            bets.append(Bet(
                                match_date=row.get('Date', ''),
                                league=f"{league_name} [{bk_name}]",
                                home=row.get('HomeTeam', ''),
                                away=row.get('AwayTeam', ''),
                                bet_type='under',
                                odds=bk_under,
                                consensus=cons_under,
                                edge_pct=round(edge, 1),
                                total_goals=total_goals,
                                result=outcome,
                                profit=round(profit * 1000, 2),
                                total_bookmakers=0,
                            ))

        except (ValueError, TypeError):
            continue

    return bets


def calculate_returns(bets: list) -> dict:
    """Calculate performance metrics."""
    if not bets:
        return {'total_bets': 0, 'wins': 0, 'losses': 0, 'profit': 0, 'roi': 0}

    bankroll = 1000
    peak = 1000
    max_drawdown = 0
    wins = sum(1 for b in bets if b.result == 'win')
    losses = sum(1 for b in bets if b.result == 'loss')
    total_staked = len(bets) * STAKE_FRAC * 1000
    profit = sum(b.profit for b in bets)

    # Track drawdown
    equity = 1000
    for b in bets:
        equity += b.profit
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_drawdown:
            max_drawdown = dd

    avg_odds = sum(b.odds for b in bets) / len(bets)

    return {
        'total_bets': len(bets),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(bets) * 100 if bets else 0,
        'profit': round(profit, 2),
        'roi': round(profit / total_staked * 100, 2),
        'avg_odds': round(avg_odds, 3),
        'max_drawdown': round(max_drawdown, 1),
        'final_bankroll': round(1000 + profit, 2),
    }


def split_by_edge(bets: list) -> dict:
    """Split performance by edge size."""
    buckets = {
        '5-10%': [b for b in bets if 5 <= b.edge_pct < 10],
        '10-20%': [b for b in bets if 10 <= b.edge_pct < 20],
        '20%+': [b for b in bets if b.edge_pct >= 20],
    }
    results = {}
    for label, bucket in buckets.items():
        results[label] = calculate_returns(bucket)
    return results


def main():
    t0 = time.time()
    all_bets = []

    # Process BetBrain era (2015-2019)
    print("BetBrain Era (2015-2019):")
    print(SEP)
    for league_code, league_name in LEAGUES.items():
        for season in SEASONS_OLD:
            url = URL_PATTERN_OLD.format(season=season, league=league_code)
            rows = fetch_csv(url)
            if not rows or len(rows) < 10:
                continue
            bets = process_betbrain_era(rows, league_name)
            all_bets.extend(bets)
            if bets:
                print(f"  {league_name} {season}: {len(bets)} bets")

    # Process individual bookmaker era (2020-2025)
    print(f"\nIndividual Bookmaker Era (2020-2025):")
    print(SEP)
    for league_code, league_name in LEAGUES.items():
        for season in SEASONS_NEW:
            url = URL_PATTERN_NEW.format(season=season, league=league_code)
            rows = fetch_csv(url)
            if not rows or len(rows) < 10:
                continue
            bets = process_individual_era(rows, league_name)
            all_bets.extend(bets)
            if bets:
                print(f"  {league_name} {season}: {len(bets)} bets")

    print(f"\n{'=' * 70}")
    print("OVER/UNDER 2.5 GOALS — CONSENSUS BACKTEST RESULTS")
    print(f"{'=' * 70}")
    print(f"Period: 2015-2025 (5 European leagues)")
    print(f"Edge threshold: {MIN_EDGE_PCT}%")
    print(f"Odds range: {MIN_ODDS}-{MAX_ODDS}")
    print(f"Stake: {STAKE_FRAC*100:.0f}% per bet")
    print(f"Min bookmakers for consensus: {MIN_BOOKMAKERS}")
    print()

    # Overall results
    overall = calculate_returns(all_bets)
    print(f"Total bets: {overall['total_bets']}")
    print(f"Wins: {overall['wins']}  Losses: {overall['losses']}")
    print(f"Win rate: {overall['win_rate']:.1f}%")
    print(f"Average odds: {overall['avg_odds']}")
    print(f"Profit: £{overall['profit']:+.2f}")
    print(f"ROI: {overall['roi']:+.2f}%")
    print(f"Final bankroll: £{overall['final_bankroll']:.2f}")
    print(f"Max drawdown: {overall['max_drawdown']:.1f}%")

    # Split by direction
    over_bets = [b for b in all_bets if b.bet_type == 'over']
    under_bets = [b for b in all_bets if b.bet_type == 'under']
    print(f"\n--- By Direction ---")
    over_r = calculate_returns(over_bets)
    under_r = calculate_returns(under_bets)
    print(f"Over bets:  {over_r['total_bets']:>4d} bets, WR {over_r['win_rate']:.1f}%, P&L £{over_r['profit']:+.2f}, ROI {over_r['roi']:+.2f}%")
    print(f"Under bets: {under_r['total_bets']:>4d} bets, WR {under_r['win_rate']:.1f}%, P&L £{under_r['profit']:+.2f}, ROI {under_r['roi']:+.2f}%")

    # Split by edge
    print(f"\n--- By Edge Size ---")
    edge_res = split_by_edge(all_bets)
    for label, res in edge_res.items():
        if res['total_bets'] > 0:
            print(f"Edge {label}: {res['total_bets']:>4d} bets, WR {res['win_rate']:.1f}%, P&L £{res['profit']:+.2f}, ROI {res['roi']:+.2f}%")

    # By league
    print(f"\n--- By League ---")
    for league_code, league_name in LEAGUES.items():
        league_bets = [b for b in all_bets if league_name in b.league]
        if league_bets:
            lr = calculate_returns(league_bets)
            print(f"{league_name:30s}: {lr['total_bets']:>4d} bets, WR {lr['win_rate']:.1f}%, P&L £{lr['profit']:+.2f}, ROI {lr['roi']:+.2f}%")

    # Era comparison
    bb_bets = [b for b in all_bets if '[' not in b.league]
    ind_bets = [b for b in all_bets if '[' in b.league]
    if bb_bets:
        bb_r = calculate_returns(bb_bets)
        print(f"\n--- By Era ---")
        print(f"BetBrain consensus (2015-2019): {bb_r['total_bets']:>4d} bets, WR {bb_r['win_rate']:.1f}%, P&L £{bb_r['profit']:+.2f}, ROI {bb_r['roi']:+.2f}%")
    if ind_bets:
        ind_r = calculate_returns(ind_bets)
        print(f"Individual bookmaker (2020-2025): {ind_r['total_bets']:>4d} bets, WR {ind_r['win_rate']:.1f}%, P&L £{ind_r['profit']:+.2f}, ROI {ind_r['roi']:+.2f}%")

    # Save to CSV for analysis
    out_path = Path('/home/burley/football-ml')
    out_path.mkdir(exist_ok=True)
    if all_bets:
        with open(out_path / 'over_under_trades.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Date', 'League', 'Home', 'Away', 'Type', 'Odds', 'Consensus', 'Edge%', 'TotalGoals', 'Result', 'Profit', 'N_Bookies'])
            for b in sorted(all_bets, key=lambda x: x.match_date):
                w.writerow([b.match_date, b.league, b.home, b.away, b.bet_type,
                           b.odds, b.consensus, b.edge_pct, b.total_goals,
                           b.result, b.profit, b.total_bookmakers])
        print(f"\nTrades saved to {out_path / 'over_under_trades.csv'}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
