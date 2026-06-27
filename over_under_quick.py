#!/usr/bin/env python3
"""Quick check of edge 10% and 15% thresholds on BetBrain era."""
import csv, io, urllib.request, time
import warnings
warnings.filterwarnings('ignore')

STAKE = 0.02
MIN_ODDS = 1.3
MAX_ODDS = 3.0
MIN_B = 3

LEAGUES = {
    'E0':'EPL','E1':'Champ','E2':'Lg1','E3':'Lg2','EC':'Conf',
    'D1':'BL1','D2':'BL2','I1':'SA','I2':'SB',
    'SP1':'LL','SP2':'Seg','F1':'L1','F2':'L2',
    'N1':'Ered','B1':'Jup','P1':'Port','SC0':'SPL','T1':'Turk',
}

class Bet:
    __slots__ = ('direction','odds','consensus','edge','total_goals','n_bookies')
    def __init__(self, direction, odds, consensus, edge, total_goals, n_bookies):
        self.direction = direction; self.odds = odds; self.consensus = consensus
        self.edge = edge; self.total_goals = total_goals; self.n_bookies = n_bookies
    @property
    def result(self): return 'win' if (self.direction=='over' and self.total_goals>2.5) or (self.direction=='under' and self.total_goals<2.5) else 'loss'
    @property
    def profit(self):
        return round((STAKE*(self.odds-1) if self.result=='win' else -STAKE)*1000, 2)

def fetch(url):
    for _ in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=15)
            return list(csv.DictReader(io.StringIO(resp.read().decode('utf-8-sig'))))
        except:
            time.sleep(1)
    return None

def scan(edge_min):
    bets = []
    for lc, ln in LEAGUES.items():
        for s in [f"{y}{y+1}" for y in range(15,20)]:
            rows = fetch(f"https://www.football-data.co.uk/mmz4281/{s}/{lc}.csv")
            if not rows or len(rows) < 10: continue
            for row in rows:
                try:
                    nb = int(row.get('BbOU','0') or '0')
                    if nb < MIN_B: continue
                    tg = int(row.get('FTHG','0') or '0') + int(row.get('FTAG','0') or '0')
                    co = float(row.get('BbAv>2.5','') or '0')
                    cu = float(row.get('BbAv<2.5','') or '0')
                    if co <= 0 or cu <= 0: continue
                    bo = float(row.get('BbMx>2.5','') or '0')
                    bu = float(row.get('BbMx<2.5','') or '0')
                    if bo >= MIN_ODDS and bo <= MAX_ODDS:
                        e = (1.0/co - 1.0/bo)/(1.0/bo)*100
                        if e >= edge_min: bets.append(Bet('over',bo,co,e,tg,nb))
                    if bu >= MIN_ODDS and bu <= MAX_ODDS:
                        e = (1.0/cu - 1.0/bu)/(1.0/bu)*100
                        if e >= edge_min: bets.append(Bet('under',bu,cu,e,tg,nb))
                except: pass
    return bets

def stats(bets, label):
    if not bets: print(f"{label:30s}: 0 bets"); return
    n = len(bets); w = sum(1 for b in bets if b.result=='win')
    p = sum(b.profit for b in bets); ts = n * STAKE * 1000
    avg_o = sum(b.odds for b in bets)/n
    eq=1000; pk=1000; dd=0
    for b in bets:
        eq+=b.profit
        if eq>pk: pk=eq
        d=(pk-eq)/pk*100
        if d>dd: dd=d
    print(f"{label:30s}: {n:>5d}b  WR {w/n*100:5.1f}%  AVG {avg_o:.3f}  P&L £{p:>+8.2f}  ROI {p/ts*100:>+6.2f}%  DD {dd:5.1f}%")

for edge in [10, 15]:
    bets = scan(edge)
    overs = [b for b in bets if b.direction=='over']
    unders = [b for b in bets if b.direction=='under']
    print(f"\n{'='*70}")
    print(f"EDGE ≥ {edge}%  (BetBrain era, 18 leagues, 2015-2019)")
    print(f"{'='*70}")
    stats(bets, "All bets")
    stats(overs, "Overs only")
    stats(unders, "Unders only")

print(f"\n{'='*70}")
print("BEST: Over 2.5, Edge ≥8%, 6+ Bookmakers")
print(f"{'='*70}")
bets = scan(8)
filt = [b for b in bets if b.direction=='over' and b.n_bookies >= 6]
stats(filt, "Over 2.5, ≥8% edge, ≥6 books")
