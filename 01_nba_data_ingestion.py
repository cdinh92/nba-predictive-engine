import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from dotenv import load_dotenv

load_dotenv()

def fetch_nba_historical_games(seasons):
    """
    Fetches historical NBA regular season and playoff games using nba_api,
    calculates Four Factors and Possession/Pace-based efficiency metrics
    (the NBA equivalent of EPA and CPOE), and merges into Home/Away matchups.
    """
    all_games = []
    for season in seasons:
        print(f"Downloading NBA game logs for season: {season}...")
        for season_type in ['Regular Season', 'Playoffs']:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable='00',
                season_type_nullable=season_type
            )
            df_season = finder.get_data_frames()[0]
            if not df_season.empty:
                df_season['SEASON_STR'] = season
                all_games.append(df_season)
            time.sleep(0.6)  # Respect NBA API rate limits

    raw_df = pd.concat(all_games, ignore_index=True)
    raw_df = raw_df.dropna(subset=['WL', 'PTS', 'FGA', 'FTA', 'TOV', 'OREB']).copy()

    # Estimate Possessions using standard Dean Oliver formula:
    # Poss = FGA + 0.44 * FTA - OREB + TOV
    raw_df['POSS'] = raw_df['FGA'] + (0.44 * raw_df['FTA']) - raw_df['OREB'] + raw_df['TOV']
    raw_df['POSS'] = raw_df['POSS'].replace(0, 1)

    # Four Factors Offensive Metrics (NBA equivalent of CPOE & Net Yards/Play)
    raw_df['EFG_PCT'] = (raw_df['FGM'] + 0.5 * raw_df['FG3M']) / raw_df['FGA'].replace(0, 1)
    raw_df['TS_PCT'] = raw_df['PTS'] / (2 * (raw_df['FGA'] + 0.44 * raw_df['FTA']).replace(0, 1))
    raw_df['TOV_PCT'] = raw_df['TOV'] / raw_df['POSS']
    raw_df['FTR'] = raw_df['FTA'] / raw_df['FGA'].replace(0, 1)
    raw_df['OFF_RTG'] = (raw_df['PTS'] / raw_df['POSS']) * 100.0

    # Split into Home ('vs.') and Away ('@') rows to pair game_id matchups
    home_df = raw_df[raw_df['MATCHUP'].str.contains(r'vs\.')].copy()
    away_df = raw_df[raw_df['MATCHUP'].str.contains(r'@')].copy()

    home_cols = {
        'GAME_ID': 'game_id',
        'SEASON_STR': 'season',
        'GAME_DATE': 'gameday',
        'TEAM_ABBREVIATION': 'home_team',
        'PTS': 'home_score',
        'POSS': 'home_poss',
        'OFF_RTG': 'home_off_rtg',
        'EFG_PCT': 'home_efg_pct',
        'TS_PCT': 'home_ts_pct',
        'TOV_PCT': 'home_tov_pct',
        'OREB': 'home_oreb',
        'DREB': 'home_dreb',
        'FTR': 'home_ftr',
        'AST': 'home_ast',
        'FG3M': 'home_fg3m'
    }

    away_cols = {
        'GAME_ID': 'game_id',
        'TEAM_ABBREVIATION': 'away_team',
        'PTS': 'away_score',
        'POSS': 'away_poss',
        'OFF_RTG': 'away_off_rtg',
        'EFG_PCT': 'away_efg_pct',
        'TS_PCT': 'away_ts_pct',
        'TOV_PCT': 'away_tov_pct',
        'OREB': 'away_oreb',
        'DREB': 'away_dreb',
        'FTR': 'away_ftr',
        'AST': 'away_ast',
        'FG3M': 'away_fg3m'
    }

    home_df = home_df[list(home_cols.keys())].rename(columns=home_cols)
    away_df = away_df[list(away_cols.keys())].rename(columns=away_cols)

    df = pd.merge(home_df, away_df, on='game_id', how='inner')

    # Cross-compute Defensive Rating (Opponent's Off Rating) and OREB%
    df['home_def_rtg'] = df['away_off_rtg']
    df['away_def_rtg'] = df['home_off_rtg']
    df['home_net_rtg'] = df['home_off_rtg'] - df['home_def_rtg']
    df['away_net_rtg'] = df['away_off_rtg'] - df['away_def_rtg']

    df['home_oreb_pct'] = df['home_oreb'] / (df['home_oreb'] + df['away_dreb']).replace(0, 1)
    df['away_oreb_pct'] = df['away_oreb'] / (df['away_oreb'] + df['home_dreb']).replace(0, 1)
    df['pace'] = (df['home_poss'] + df['away_poss']) / 2.0

    # Placeholder moneylines if historical odds CSV is not merged
    df['home_moneyline'] = -110
    df['away_moneyline'] = -110

    df['gameday'] = pd.to_datetime(df['gameday']).dt.strftime('%Y-%m-%d')
    df = df.sort_values(by=['gameday', 'game_id']).reset_index(drop=True)
    return df

if __name__ == "__main__":
    # Include up through the active 2026-27 NBA season
    target_seasons = ['2023-24', '2024-25', '2025-26', '2026-27']
    games_df = fetch_nba_historical_games(target_seasons)

    print("\n--- Games per Season ---")
    print(games_df['season'].value_counts())

    print("\n--- Latest Data Preview ---")
    print(games_df.tail())

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    output_path = os.path.join(DATA_DIR, "raw_nba_games.csv")
    games_df.to_csv(output_path, index=False)
    print(f"\n✅ Successfully saved {len(games_df)} NBA games to {output_path}")
