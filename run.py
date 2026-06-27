#!/usr/bin/env python3
"""
Football ML Betting System — Asian vs European Markets
======================================================
Data: Beat The Bookie closing_odds (479K matches, 2005-2015)
Tests whether Asian leagues have exploitable pricing inefficiencies
compared to efficient European top-5 markets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings, time
warnings.filterwarnings('ignore')
t0 = time.time()

SEP = "=" * 60
DATA = Path.home() / '.cache/kagglehub/datasets/austro/beat-the-bookie-worldwide-football-dataset/versions/2'
OUT = Path('/home/burley/football-ml')
OUT.mkdir(exist_ok=True)

# ── Config ──
TRAIN_END = '2013-06-30'
CAL_END   = '2013-12-31'
BANKROLL = 10000
STAKE = 0.02  # fixed 2% per bet (no Kelly - avoids miscalibration issues)

# Asian leagues (target markets)
ASIAN_LEAGUES = [
    'Japan: J-League', 'Japan: J-League Division 2',
    'South Korea: K-League', 'South Korea: National League',
    'China: Super League', 'China: Jia League',
    'Thailand: Thai Premier League',
    'Australia: A-League',
    'Singapore: S-League', 'Singapore: S.League',
    'India: I-League',
    'Malaysia: Super League',
    'Indonesia: Super Liga',
    'Vietnam: V-League',
    'Asia: AFC Champions League',
    'Hong Kong: Premier League',
]

# European top-5 leagues (efficient baseline)
EURO_TOP5 = [
    'England: Premier League', 'England: Championship',
    'Spain: La Liga', 'Spain: Segunda Division',
    'Germany: Bundesliga', 'Germany: 2. Bundesliga',
    'Italy: Serie A', 'Italy: Serie B',
    'France: Ligue 1', 'France: Ligue 2',
]

# ── 1. LOAD ──
print(SEP)
print("FOOTBALL BETTING ML — ASIAN vs EUROPEAN")
print(SEP)
print("\n[1] Loading closing odds...")

df = pd.read_csv(DATA / 'closing_odds.csv.gz', compression='gzip',
                 encoding='latin1', low_memory=False)
df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
print(f"  Total: {len(df):,} matches, {df['league'].nunique():,} leagues")
print(f"  Date:  {df['match_date'].min().date()} to {df['match_date'].max().date()}")

# Tag regions
df['region'] = 'Other'
df.loc[df['league'].isin(ASIAN_LEAGUES), 'region'] = 'Asia'
df.loc[df['league'].isin(EURO_TOP5), 'region'] = 'Europe_Top5'

print(f"\n  Asia:       {df[df['region']=='Asia'].shape[0]:>7,} matches ({len(ASIAN_LEAGUES)} leagues)")
print(f"  Europe Top5:{df[df['region']=='Europe_Top5'].shape[0]:>7,} matches ({len(EURO_TOP5)} leagues)")
print(f"  Other:      {df[df['region']=='Other'].shape[0]:>7,} matches")

# ── 2. CLEAN ──
print(f"\n[2] Cleaning & feature engineering...")

# Parse scores
df[['home_score','away_score']] = df[['home_score','away_score']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['home_score','away_score'])

# Target: home win, draw, away win
df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
df['draw'] = (df['home_score'] == df['away_score']).astype(int)
df['away_win'] = (df['home_score'] < df['away_score']).astype(int)

# Clean odds columns
for c in ['avg_odds_home_win','avg_odds_draw','avg_odds_away_win',
          'max_odds_home_win','max_odds_draw','max_odds_away_win']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Market-implied probabilities (from average odds, inverse of overround-adjusted odds)
# The overround = 1/home + 1/draw + 1/away - 1
df['market_home'] = 1.0 / df['avg_odds_home_win']
df['market_draw'] = 1.0 / df['avg_odds_draw']
df['market_away'] = 1.0 / df['avg_odds_away_win']

# Drop rows with missing odds
df = df.dropna(subset=['avg_odds_home_win','avg_odds_draw','avg_odds_away_win',
                        'home_win','draw','away_win'])
print(f"  Usable: {len(df):,} matches")

# ── Sort for rolling features ──
df = df.sort_values(['home_team','match_date']).reset_index(drop=True)

# ── TEAM FORM FEATURES (rolling, look-back only) ──
print("  Building team form features...")

def team_form_features(match_df, team_col, target_cols, windows=[5]):
    """Add rolling form features for a team column.
    Features are computed per team, sorted by date, shifted to avoid look-ahead.
    """
    result = {}
    for w in windows:
        for t in target_cols:
            # Points equivalent: home_win=3, draw=1, away_win=0... 
            # Actually for a given team, we want: points, goals_for, goals_against
            pass
    return result

# Simpler approach: for each match, look up each team's last N results
# We'll do this with a manual join-like approach

# First, create a unified "team event" dataframe
home_events = df[['match_date','home_team','home_score','away_score','home_win']].copy()
home_events.columns = ['date','team','goals_for','goals_against','won']
home_events['points'] = home_events['won'] * 3 + 0  # will fix draws
home_events.loc[home_events['won'] == 0, 'points'] = 0  # loss
home_events['played'] = 1

away_events = df[['match_date','away_team','away_score','home_score','away_win']].copy()
away_events.columns = ['date','team','goals_for','goals_against','won']
away_events['points'] = away_events['won'] * 3  # away wins = 3 pts
away_events['played'] = 1

team_events = pd.concat([home_events, away_events], ignore_index=True)
team_events = team_events.sort_values(['team','date']).reset_index(drop=True)

# Rolling features per team
print("  Computing rolling form (this may take a minute)...")
t1 = time.time()
team_events['pts_ma5'] = team_events.groupby('team')['points'].transform(
    lambda x: x.rolling(6, min_periods=1).mean().shift(1))
team_events['gf_ma5'] = team_events.groupby('team')['goals_for'].transform(
    lambda x: x.rolling(6, min_periods=1).mean().shift(1))
team_events['ga_ma5'] = team_events.groupby('team')['goals_against'].transform(
    lambda x: x.rolling(6, min_periods=1).mean().shift(1))

# Fill first entries
for c in ['pts_ma5','gf_ma5','ga_ma5']:
    team_events[c] = team_events[c].fillna(team_events[c].median())

print(f"  Rolling features: {time.time()-t1:.0f}s")

# Merge back: home team form
home_form = team_events[['date','team','pts_ma5','gf_ma5','ga_ma5']].copy()
home_form.columns = ['match_date','home_team','home_pts_ma5','home_gf_ma5','home_ga_ma5']
df = df.merge(home_form, on=['match_date','home_team'], how='left')

# Away team form
away_form = team_events[['date','team','pts_ma5','gf_ma5','ga_ma5']].copy()
away_form.columns = ['match_date','away_team','away_pts_ma5','away_gf_ma5','away_ga_ma5']
df = df.merge(away_form, on=['match_date','away_team'], how='left')

# Form difference features
df['form_diff'] = df['home_pts_ma5'] - df['away_pts_ma5']
df['gf_diff'] = df['home_gf_ma5'] - df['away_gf_ma5']
df['ga_diff'] = df['home_ga_ma5'] - df['away_ga_ma5']

# League encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['league_enc'] = le.fit_transform(df['league'].astype(str))

# Encode team names
df['home_enc'] = le.fit_transform(df['home_team'].astype(str))
# For away teams, we need consistency - let's fit on all teams
all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
team_le = LabelEncoder()
team_le.fit(all_teams.astype(str))
df['home_enc2'] = team_le.transform(df['home_team'].astype(str))
df['away_enc2'] = team_le.transform(df['away_team'].astype(str))

# Market features (as odds ratios for model)
df['market_home_ratio'] = df['market_home'] / (df['market_home'] + df['market_draw'] + df['market_away'])
df['market_draw_ratio'] = df['market_draw'] / (df['market_home'] + df['market_draw'] + df['market_away'])
df['market_away_ratio'] = df['market_away'] / (df['market_home'] + df['market_draw'] + df['market_away'])

# Drop rows with missing form features
df = df.dropna(subset=['home_pts_ma5','away_pts_ma5','form_diff'])
print(f"  Final usable: {len(df):,} matches [{time.time()-t0:.0f}s]")

# ── 3. SPLIT ──
print(f"\n[3] Temporal split...")

train = df[df['match_date'] <= TRAIN_END].copy()
cal = df[(df['match_date'] > TRAIN_END) & (df['match_date'] <= CAL_END)].copy()
test = df[df['match_date'] > CAL_END].copy()

print(f"  Train:       {len(train):,} ({train['match_date'].min().date()} to {train['match_date'].max().date()})")
print(f"  Calibrate:   {len(cal):,} ({cal['match_date'].min().date()} to {cal['match_date'].max().date()})")
print(f"  Backtest:    {len(test):,} ({test['match_date'].min().date()} to {test['match_date'].max().date()})")

# Features for Home Win prediction
FEATURES = [
    'home_pts_ma5','home_gf_ma5','home_ga_ma5',
    'away_pts_ma5','away_gf_ma5','away_ga_ma5',
    'form_diff','gf_diff','ga_diff',
    'market_home_ratio','market_draw_ratio','market_away_ratio',
    'league_enc','home_enc2','away_enc2',
]

X_train = train[FEATURES].values.astype(np.float32)
y_train = train['home_win'].values.astype(np.float32)
X_cal = cal[FEATURES].values.astype(np.float32)
y_cal = cal['home_win'].values.astype(np.float32)
X_test = test[FEATURES].values.astype(np.float32)
y_test = test['home_win'].values.astype(np.float32)

# ── 4. TRAIN ──
print(f"\n[4] Training XGBoost...")

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
print(f"  Home win rate: {pos/len(y_train)*100:.1f}% (class weight: {neg/pos:.1f})")

model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=neg/pos,
    reg_lambda=2.0, reg_alpha=0.5,
    eval_metric='logloss', random_state=42, verbosity=0,
)
model.fit(X_train, y_train, verbose=False)

# Calibrate
calibrator = LogisticRegression()
calibrator.fit(model.predict_proba(X_cal)[:,1].reshape(-1,1), y_cal)

def calibrate(p):
    return calibrator.predict_proba(p.reshape(-1,1))[:,1]

train_prob = calibrate(model.predict_proba(X_train)[:,1])
test_prob = calibrate(model.predict_proba(X_test)[:,1])

print(f"\n  ROC-AUC: train={roc_auc_score(y_train, train_prob):.4f}  test={roc_auc_score(y_test, test_prob):.4f}")
print(f"  Brier:   train={brier_score_loss(y_train, train_prob):.4f}  test={brier_score_loss(y_test, test_prob):.4f}")

# Feature importance
imp = pd.DataFrame({'feat': FEATURES, 'imp': model.feature_importances_}).sort_values('imp', ascending=False)
print("\n  Top features:")
for _, r in imp.head(10).iterrows():
    print(f"    {r['feat']:25s}  {r['imp']:.4f}")

# ── 5. EVALUATE BY REGION ──
print(f"\n{SEP}")
print("EVALUATION BY REGION")
print(SEP)

test['prob'] = test_prob
test['edge'] = test['prob'] - test['market_home_ratio']

for region, label in [('Asia', 'Asian Leagues'), ('Europe_Top5', 'Europe Top 5'), ('Other', 'Other')]:
    sub = test[test['region'] == region]
    if len(sub) < 100:
        continue
    y = sub['home_win'].values
    p = sub['prob'].values
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    mkt = sub['market_home_ratio'].mean()
    actual = y.mean()
    print(f"\n  {label:20s} (n={len(sub):,}):")
    print(f"    ROC-AUC:      {auc:.4f}")
    print(f"    Brier:        {brier:.4f}")
    print(f"    Market avg:   {mkt:.3f}")
    print(f"    Actual H win: {actual:.3f}")
    print(f"    Model avg:    {p.mean():.3f}")

# ── 6. BACKTEST ──
print(f"\n{SEP}")
print("BACKTEST SIMULATION")
print(SEP)

def backtest(df_sub, label, min_edge=0.0):
    """Backtest a simple strategy: bet on home wins with edge > min_edge."""
    sub = df_sub.copy()
    sub['bet_signal'] = (
        (sub['prob'] > sub['market_home_ratio'] + min_edge) &
        (sub['prob'] >= 0.25)  # minimum confidence
    )
    bets = sub[sub['bet_signal']]
    
    if len(bets) < 10:
        print(f"  {label:20s}: too few bets ({len(bets)})")
        return
    
    bankroll = BANKROLL
    wins = 0
    for _, row in bets.iterrows():
        stake = bankroll * STAKE
        if row['home_win']:
            bankroll += stake * (row['avg_odds_home_win'] - 1) * 0.95
            wins += 1
        else:
            bankroll -= stake
    
    n = len(bets)
    ret = (bankroll / BANKROLL - 1) * 100
    print(f"  {label:20s} (n={n:>4,}): "
          f"£{BANKROLL:,.0f} → £{bankroll:,.0f} "
          f"({ret:+.0f}%)  win={wins/n*100:.0f}%  "
          f"avg_odds={bets['avg_odds_home_win'].mean():.1f}  "
          f"avg_edge={bets['edge'].mean():.3f}")
    return bankroll

# Global backtest
print(f"\n  All matches (global):")
backtest(test, 'No edge filter')
backtest(test, 'Edge > 1%', min_edge=0.01)
backtest(test, 'Edge > 3%', min_edge=0.03)
backtest(test, 'Edge > 5%', min_edge=0.05)

# By region
print(f"\n  By region (edge > 3%):")
for region, label in [('Asia', 'Asia'), ('Europe_Top5', 'EU Top5'), ('Other', 'Other')]:
    sub = test[test['region'] == region]
    backtest(sub, label, min_edge=0.03)

# Asian leagues individually
print(f"\n  Individual Asian leagues (edge > 3%):")
for league in ASIAN_LEAGUES:
    sub = test[test['league'] == league]
    if len(sub) >= 200:
        backtest(sub, league.split(':')[0][:18], min_edge=0.03)

# European comparison
print(f"\n  Individual European top-5 (edge > 3%):")
for league in EURO_TOP5:
    sub = test[test['league'] == league]
    n = len(sub)
    if n >= 200:
        backtest(sub, league.split(':')[0][:18], min_edge=0.03)

# ═══════════════════════════════════════════════════════════════
# 7. SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"\n  Data: {len(df):,} matches from {len(ASIAN_LEAGUES)} Asian + {len(EURO_TOP5)} European leagues")
print(f"  Model: XGBoost + Platt scaling")
print(f"  Global ROC-AUC: {roc_auc_score(y_test, test_prob):.4f}")
print(f"  Runtime: {time.time()-t0:.0f}s")

# Save
test.to_csv(OUT / 'predictions.csv', index=False)
imp.to_csv(OUT / 'feature_importance.csv', index=False)
print(f"\n  Results saved to {OUT}/")
print(f"{SEP}")
