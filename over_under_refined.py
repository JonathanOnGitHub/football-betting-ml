#!/usr/bin/env python3
"""
Refined Over/Under 2.5 Backtest
Only BetBrain era (2015-2019) — cleaner consensus signal.
Tests different edge thresholds and direction filters.
Also adds more leagues and older seasons.
"""

import csv
import io
import urllib.request
import time
from pathlib import Path
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

SEP = "=" * 60
START_BANKROLL = 1000
STAKE_FRAC = 0.02
MIN_ODDS = 1.3
MAX_ODDS = 3.0
MIN_BOOKMAKERS = 3

LEAGUES = {
    'E0': 'England Premier League',
    'E1': 'England Championship',
    'E2': 'England League 1',
    'E3': 'England League 2',
    'EC': 'England Conference',
    'D1': 'Germany Bundesliga 1',
    'D2': 'Germany Bundesliga 2',
    'I1': 'Italy Serie A',
    'I2': 'Italy Serie B',
    'SP1': 'Spain La Liga',
    'SP2': 'Spain Segunda',
    'F1': 'France Ligue 1',
    'F2': 'France Ligue 2',
    'N1': 'Netherlands Eredivisie',
    'B1': 'Belgium Jupiler',
    'P1': 'Portugal Liga',
    'SC0': 'Scotland Premiership',
    'T1': 'Turkey Super Lig',
}

# BetBrain era seasons
SEASONS = [f"{y}{y+1}" for y in range(15, 20)]
URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


@dataclass
class Bet:
    match_date: str
    league: str
    home: str
    away: str
    bet_type: str
    odds: float
    consensus: float
    edge_pct: float
    total_goals: int
    result: str
    profit: float
    n_bookies: int


def fetch_csv(url, max_retries=2):
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=15)
            data = resp.read().decode('utf-8-sig')
            return list(csv.DictReader(io.StringIO(data)))
        except:
            if attempt < max_retries - 1:
                time.sleep(1)
            return None


def process_bets(rows, league_name, edge_threshold):
    bets = []
    for row in rows:
        try:
            n_bookies = int(row.get('BbOU', '0') or '0')
            if n_bookies < MIN_BOOKMAKERS:
                continue

            total_goals = int(row.get('FTHG', '0') or '0') + int(row.get('FTAG', '0') or '0')
            cons_over = float(row.get('BbAv>2.5', '') or '0')
            cons_under = float(row.get('BbAv<2.5', '') or '0')
            if cons_over <= 0 or cons_under <= 0:
                continue

            best_over = float(row.get('BbMx>2.5', '') or '0')
            best_under = float(row.get('BbMx<2.5', '') or '0')

            # Check Over
            if best_over >= MIN_ODDS and best_over <= MAX_ODDS:
                implied_cons = 1.0 / cons_over
                implied_best = 1.0 / best_over
                edge = (implied_cons - implied_best) / implied_best * 100
                if edge >= edge_threshold:
                    outcome = 'win' if total_goals > 2.5 else 'loss'
                    profit = STAKE_FRAC * (best_over - 1) if outcome == 'win' else -STAKE_FRAC
                    bets.append(Bet(row.get('Date',''), league_name, row.get('HomeTeam',''),
                                   row.get('AwayTeam',''), 'over', best_over, cons_over,
                                   round(edge,1), total_goals, outcome,
                                   round(profit*1000,2), n_bookies))

            # Check Under
            if best_under >= MIN_ODDS and best_under <= MAX_ODDS:
                implied_cons = 1.0 / cons_under
                implied_best = 1.0 / best_under
                edge = (implied_cons - implied_best) / implied_best * 100
                if edge >= edge_threshold:
                    outcome = 'win' if total_goals < 2.5 else 'loss'
                    profit = STAKE_FRAC * (best_under - 1) if outcome == 'win' else -STAKE_FRAC
                    bets.append(Bet(row.get('Date',''), league_name, row.get('HomeTeam',''),
                                   row.get('AwayTeam',''), 'under', best_under, cons_under,
                                   round(edge,1), total_goals, outcome,
                                   round(profit*1000,2), n_bookies))

        except (ValueError, TypeError):
            continue
    return bets


def calc(bets):
    if not bets:
        return {}
    wins = sum(1 for b in bets if b.result == 'win')
    losses = sum(1 for b in bets if b.result == 'loss')
    profit = sum(b.profit for b in bets)
    total_staked = len(bets) * STAKE_FRAC * 1000
    avg_odds = sum(b.odds for b in bets) / len(bets) if bets else 0

    equity = 1000
    peak = 1000
    max_dd = 0
    for b in bets:
        equity += b.profit
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd

    return {
        'n': len(bets), 'wins': wins, 'losses': losses,
        'wr': wins/len(bets)*100, 'profit': round(profit,2),
        'roi': round(profit/total_staked*100,2), 'avg_odds': round(avg_odds,3),
        'max_dd': round(max_dd,1), 'final': round(1000+profit,2)
    }


def main():
    t0 = time.time()

    for edge_threshold in [3, 5, 8, 10, 15]:
        all_bets = []
        for league_code, league_name in LEAGUES.items():
            for season in SEASONS:
                url = URL.format(season=season, league=league_code)
                rows = fetch_csv(url)
                if not rows or len(rows) < 10:
                    continue
                bets = process_bets(rows, league_name, edge_threshold)
                all_bets.extend(bets)

        r = calc(all_bets)
        if r:
            # By direction
            over_r = calc([b for b in all_bets if b.bet_type == 'over'])
            under_r = calc([b for b in all_bets if b.bet_type == 'under'])

            # By N bookmakers
            big_cons = calc([b for b in all_bets if b.n_bookies >= 6])
            small_cons_bets = [b for b in all_bets if 3 <= b.n_bookies < 6]
            small_cons = calc(small_cons_bets) if small_cons_bets else {'n':0,'wr':0,'profit':0,'roi':0,'avg_odds':0}

            sm_str = f"{small_cons['n']:>5d}b  WR {small_cons['wr']:5.1f}%  P&L £{small_cons['profit']:>+7.2f}  ROI {small_cons.get('roi',0):>+6.2f}%" if small_cons['n'] > 0 else "    0 bets"
            print(f"6+ books:  {big_cons['n']:>5d}b  WR {big_cons['wr']:5.1f}%  P&L £{big_cons['profit']:>+7.2f}  ROI {big_cons['roi']:>+6.2f}%")
            print(f"3-5 books: {sm_str}")

            print(f"\n{'='*70}")
            print(f"EDGE THRESHOLD: {edge_threshold}%  |  {r['n']} total bets across {len(LEAGUES)} leagues (2015-2019)")
            print(f"{'='*70}")
            print(f"Overall:   {r['n']:>5d}b  WR {r['wr']:5.1f}%  P&L £{r['profit']:>+7.2f}  ROI {r['roi']:>+6.2f}%  DD {r['max_dd']:5.1f}%  AVG {r['avg_odds']}")
            print(f"Over:      {over_r['n']:>5d}b  WR {over_r['wr']:5.1f}%  P&L £{over_r['profit']:>+7.2f}  ROI {over_r['roi']:>+6.2f}%")
            print(f"Under:     {under_r['n']:>5d}b  WR {under_r['wr']:5.1f}%  P&L £{under_r['profit']:>+7.2f}  ROI {under_r['roi']:>+6.2f}%")
            print(f"6+ books:  {big_cons['n']:>5d}b  WR {big_cons['wr']:5.1f}%  P&L £{big_cons['profit']:>+7.2f}  ROI {big_cons['roi']:>+6.2f}%")
            print(f"3-5 books: {small_cons['n']:>5d}b  WR {small_cons['wr']:5.1f}%  P&L £{small_cons['profit']:>+7.2f}  ROI {small_cons['roi']:>+6.2f}%")

    # Best combination: overs only, 8% edge
    print(f"\n\n{'='*70}")
    print(f"BEST COMBINATION: Overs Only, Edge ≥8%, 6+ Bookmakers")
    print(f"{'='*70}")
    best_bets = []
    for league_code, league_name in LEAGUES.items():
        for season in SEASONS:
            url = URL.format(season=season, league=league_code)
            rows = fetch_csv(url)
            if not rows or len(rows) < 10:
                continue
            for row in rows:
                try:
                    n_bookies = int(row.get('BbOU', '0') or '0')
                    if n_bookies < 6:
                        continue
                    total_goals = int(row.get('FTHG','0') or '0') + int(row.get('FTAG','0') or '0')
                    cons_over = float(row.get('BbAv>2.5','') or '0')
                    if cons_over <= 0:
                        continue
                    best_over = float(row.get('BbMx>2.5','') or '0')
                    if best_over < MIN_ODDS or best_over > MAX_ODDS:
                        continue
                    implied_cons = 1.0 / cons_over
                    implied_best = 1.0 / best_over
                    edge = (implied_cons - implied_best) / implied_best * 100
                    if edge >= 8:
                        outcome = 'win' if total_goals > 2.5 else 'loss'
                        profit = STAKE_FRAC * (best_over - 1) if outcome == 'win' else -STAKE_FRAC
                        best_bets.append(Bet(row.get('Date',''), league_name, row.get('HomeTeam',''),
                                           row.get('AwayTeam',''), 'over', best_over, cons_over,
                                           round(edge,1), total_goals, outcome,
                                           round(profit*1000,2), n_bookies))
                except:
                    continue

    r = calc(best_bets)
    print(f"Total: {r['n']} bets")
    print(f"Win rate: {r['wr']:.1f}%  (breakeven: {100/r['avg_odds']:.1f}%)")
    print(f"Average odds: {r['avg_odds']}")
    print(f"Profit: £{r['profit']:+.2f}")
    print(f"ROI: {r['roi']:+.2f}%")
    print(f"Max drawdown: {r['max_dd']:.1f}%")
    print(f"Final bankroll: £{r['final']:.2f}")

    # Best leagues for overs
    print(f"\n--- By League (Overs ≥8%, 6+ books) ---")
    for league_code, league_name in LEAGUES.items():
        lb = [b for b in best_bets if b.league == league_name]
        if lb:
            lr = calc(lb)
            print(f"{league_name:30s}: {lr['n']:>4d}b  WR {lr['wr']:5.1f}%  P&L £{lr['profit']:>+7.2f}  ROI {lr['roi']:>+6.2f}%")

    # Save
    out_path = Path('/home/burley/football-ml')
    if best_bets:
        with open(out_path / 'over_under_best.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Date','League','Home','Away','Odds','Consensus','Edge%','Goals','Result','Profit','N_Books'])
            for b in sorted(best_bets, key=lambda x: x.match_date):
                w.writerow([b.match_date, b.league, b.home, b.away, b.odds,
                           b.consensus, b.edge_pct, b.total_goals, b.result, b.profit, b.n_bookies])
        print(f"\nSaved: {out_path / 'over_under_best.csv'}")

    print(f"\nTime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
