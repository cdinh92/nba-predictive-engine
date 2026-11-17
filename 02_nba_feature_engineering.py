import os
import pandas as pd
import numpy as np

def calculate_opponent_adjusted_rtg(df):
    print("Computing opponent-adjusted Offensive Rating metrics...")
    home_off = df[['game_id', 'home_team', 'home_off_rtg']].rename(
        columns={'home_team': 'team', 'home_off_rtg': 'off_rtg'}
    )
    away_off = df[['game_id', 'away_team', 'away_off_rtg']].rename(
        columns={'away_team': 'team', 'away_off_rtg': 'off_rtg'}
    )
    all_team_games = pd.concat([home_off, away_off])

    league_avg_rtg = all_team_games['off_rtg'].mean()
    team_off_factor = all_team_games.groupby('team', as_index=False)['off_rtg'].mean()
    team_off_factor['adj_off_rtg'] = team_off_factor['off_rtg'] - league_avg_rtg

    return team_off_factor[['team', 'adj_off_rtg']]

def calculate_rolling_and_rest_features(df):
    print("Calculating time-shifted EMA rolling averages (Span=10) & Rest Days / Schedule Density...")

    home_cols = [
        'game_id', 'season', 'gameday', 'home_team',
        'home_score', 'away_score', 'pace',
        'home_off_rtg', 'home_def_rtg', 'home_net_rtg',
        'home_efg_pct', 'home_ts_pct', 'home_tov_pct', 'home_oreb_pct', 'home_ftr'
    ]
    home_df = df[home_cols].copy().rename(columns={
        'home_team': 'team', 'home_score': 'pts_scored', 'away_score': 'pts_allowed',
        'home_off_rtg': 'off_rtg', 'home_def_rtg': 'def_rtg', 'home_net_rtg': 'net_rtg',
        'home_efg_pct': 'efg_pct', 'home_ts_pct': 'ts_pct', 'home_tov_pct': 'tov_pct',
        'home_oreb_pct': 'oreb_pct', 'home_ftr': 'ftr'
    })
    home_df['is_home'] = 1

    away_cols = [
        'game_id', 'season', 'gameday', 'away_team',
        'away_score', 'home_score', 'pace',
        'away_off_rtg', 'away_def_rtg', 'away_net_rtg',
        'away_efg_pct', 'away_ts_pct', 'away_tov_pct', 'away_oreb_pct', 'away_ftr'
    ]
    away_df = df[away_cols].copy().rename(columns={
        'away_team': 'team', 'away_score': 'pts_scored', 'home_score': 'pts_allowed',
        'away_off_rtg': 'off_rtg', 'away_def_rtg': 'def_rtg', 'away_net_rtg': 'net_rtg',
        'away_efg_pct': 'efg_pct', 'away_ts_pct': 'ts_pct', 'away_tov_pct': 'tov_pct',
        'away_oreb_pct': 'oreb_pct', 'away_ftr': 'ftr'
    })
    away_df['is_home'] = 0

    team_df = pd.concat([home_df, away_df]).sort_values(by=['team', 'gameday', 'game_id']).reset_index(drop=True)
    team_df['gameday_dt'] = pd.to_datetime(team_df['gameday'])

    # Calculate Rest Days & Schedule Density (B2B and 3-in-4 nights)
    team_df['prev_gameday'] = team_df.groupby(['team', 'season'])['gameday_dt'].shift(1)
    team_df['prev2_gameday'] = team_df.groupby(['team', 'season'])['gameday_dt'].shift(2)
    
    team_df['rest_days'] = (team_df['gameday_dt'] - team_df['prev_gameday']).dt.days.fillna(4).clip(upper=5)
    team_df['is_b2b'] = (team_df['rest_days'] == 1).astype(int)
    
    # 3 games in 4 nights flag
    team_df['days_since_prev2'] = (team_df['gameday_dt'] - team_df['prev2_gameday']).dt.days
    team_df['is_3_in_4'] = (team_df['days_since_prev2'] <= 3).astype(int)

    metrics_to_roll = [
        'pts_scored', 'pts_allowed', 'pace',
        'off_rtg', 'def_rtg', 'net_rtg',
        'efg_pct', 'ts_pct', 'tov_pct', 'oreb_pct', 'ftr'
    ]

    SPAN_VAL = 10
    for metric in metrics_to_roll:
        team_df[f'roll_{metric}'] = (
            team_df.groupby('team')[metric]
            .transform(lambda x: x.shift(1).ewm(span=SPAN_VAL, adjust=False).mean())
        ).fillna(0)

    stat_cols = [f'roll_{m}' for m in metrics_to_roll] + ['rest_days', 'is_b2b', 'is_3_in_4']

    home_stats = team_df[team_df['is_home'] == 1][['game_id'] + stat_cols].rename(
        columns={c: f'home_{c}' for c in stat_cols}
    )
    away_stats = team_df[team_df['is_home'] == 0][['game_id'] + stat_cols].rename(
        columns={c: f'away_{c}' for c in stat_cols}
    )

    fully_merged = df.merge(home_stats, on='game_id').merge(away_stats, on='game_id')

    adj_rtg = calculate_opponent_adjusted_rtg(fully_merged)
    fully_merged = fully_merged.merge(
        adj_rtg.rename(columns={'team': 'home_team', 'adj_off_rtg': 'home_adj_off_rtg'}),
        on='home_team', how='left'
    ).merge(
        adj_rtg.rename(columns={'team': 'away_team', 'adj_off_rtg': 'away_adj_off_rtg'}),
        on='away_team', how='left'
    )

    # Core Differentials
    fully_merged['diff_pts_scored'] = fully_merged['home_roll_pts_scored'] - fully_merged['away_roll_pts_scored']
    fully_merged['diff_pts_allowed'] = fully_merged['away_roll_pts_allowed'] - fully_merged['home_roll_pts_allowed']
    fully_merged['diff_off_rtg'] = fully_merged['home_roll_off_rtg'] - fully_merged['away_roll_off_rtg']
    fully_merged['diff_def_rtg'] = fully_merged['away_roll_def_rtg'] - fully_merged['home_roll_def_rtg']
    fully_merged['diff_net_rtg'] = fully_merged['home_roll_net_rtg'] - fully_merged['away_roll_net_rtg']
    fully_merged['diff_efg_pct'] = fully_merged['home_roll_efg_pct'] - fully_merged['away_roll_efg_pct']
    fully_merged['diff_ts_pct'] = fully_merged['home_roll_ts_pct'] - fully_merged['away_roll_ts_pct']
    fully_merged['diff_tov_pct'] = fully_merged['away_roll_tov_pct'] - fully_merged['home_roll_tov_pct']
    fully_merged['diff_oreb_pct'] = fully_merged['home_roll_oreb_pct'] - fully_merged['away_roll_oreb_pct']
    fully_merged['diff_adj_off_rtg'] = fully_merged['home_adj_off_rtg'] - fully_merged['away_adj_off_rtg']
    fully_merged['rest_diff'] = fully_merged['home_rest_days'] - fully_merged['away_rest_days']

    # Volume Sum Features
    fully_merged['sum_pts_scored'] = fully_merged['home_roll_pts_scored'] + fully_merged['away_roll_pts_scored']
    fully_merged['sum_pts_allowed'] = fully_merged['home_roll_pts_allowed'] + fully_merged['away_roll_pts_allowed']
    fully_merged['sum_pace'] = fully_merged['home_roll_pace'] + fully_merged['away_roll_pace']
    fully_merged['sum_off_rtg'] = fully_merged['home_roll_off_rtg'] + fully_merged['away_roll_off_rtg']
    fully_merged['sum_def_rtg'] = fully_merged['home_roll_def_rtg'] + fully_merged['away_roll_def_rtg']
    fully_merged['sum_efg_pct'] = fully_merged['home_roll_efg_pct'] + fully_merged['away_roll_efg_pct']
    fully_merged['sum_ts_pct'] = fully_merged['home_roll_ts_pct'] + fully_merged['away_roll_ts_pct']

    # Drop first game of history where shifted EMAs are 0
    fully_merged = fully_merged[
        (fully_merged['home_roll_pts_scored'] > 0) & (fully_merged['away_roll_pts_scored'] > 0)
    ].reset_index(drop=True)

    print("✅ All NBA rolling, differential, rest, and volume sum features assembled.")
    return fully_merged

if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

    input_path = os.path.join(DATA_DIR, 'raw_nba_games.csv')
    df = pd.read_csv(input_path)

    final_features_df = calculate_rolling_and_rest_features(df)

    output_path = os.path.join(DATA_DIR, 'nba_model_features.csv')
    final_features_df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(final_features_df)} rows with EMA differential features to {output_path}")
