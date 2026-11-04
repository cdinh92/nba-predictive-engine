from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import os
import re
import pandas as pd
import joblib
from scipy.stats import norm
from nba_api.stats.endpoints import scoreboardv2

NBA_TEAM_NAMES = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards',
}

TEAM_ID_TO_ABBR = {
    1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
    1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
    1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
    1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
    1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
    1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
    1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
    1610612762: 'UTA', 1610612764: 'WAS'
}

def get_full_name(abbr):
    return NBA_TEAM_NAMES.get(abbr, abbr)

def format_gametime_utc(gameday_val, gametime_str):
    """
    Parses NBA API Eastern Time tipoff strings (e.g., '7:30 pm ET') and converts
    them into standardized ISO-8601 UTC format ('YYYY-MM-DDTHH:MM:SSZ').
    """
    if not gametime_str or gametime_str == 'TBD' or pd.isna(gametime_str) or pd.isna(gameday_val):
        return 'TBD'
    try:
        date_str = pd.to_datetime(gameday_val).strftime('%Y-%m-%d')
        raw_str = str(gametime_str).strip()

        # Match standard NBA API tipoff string like '7:30 pm ET' or '10:00 pm ET'
        et_match = re.search(r'(\d{1,2}:\d{2})\s*(am|pm)', raw_str, re.IGNORECASE)
        if et_match:
            time_part = f"{et_match.group(1)} {et_match.group(2).upper()}"
            et_zone = ZoneInfo("America/New_York")
            local_dt = datetime.strptime(f"{date_str} {time_part}", "%Y-%m-%d %I:%M %p").replace(tzinfo=et_zone)

            # Convert to Mountain Time (uncomment if needed directly in JSON)
            # mt_zone = ZoneInfo("America/Denver")
            # return local_dt.astimezone(mt_zone).strftime('%I:%M %p')

            return local_dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

        # Fallback for 24-hour 'HH:MM' format
        time_str = raw_str[:5]
        et_zone = ZoneInfo("America/New_York")
        local_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=et_zone)
        return local_dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        return str(gametime_str)

def american_odds_to_raw_prob(ml):
    if pd.isna(ml):
        return 0.50
    ml = float(ml)
    return (-ml) / (-ml + 100) if ml < 0 else 100 / (ml + 100)

def get_team_latest_stats(df, team, target_date_str):
    team_games = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    if team_games.empty:
        return None

    last_game = team_games.iloc[-1]
    is_home = last_game['home_team'] == team
    prefix = 'home_' if is_home else 'away_'

    last_date = pd.to_datetime(last_game['gameday'])
    target_dt = pd.to_datetime(target_date_str)
    rest_days = max(1, min(5, (target_dt - last_date).days))
    is_b2b = 1 if rest_days == 1 else 0

    return {
        'roll_pts_scored': last_game.get(f'{prefix}roll_pts_scored', 112.0),
        'roll_pts_allowed': last_game.get(f'{prefix}roll_pts_allowed', 112.0),
        'roll_pace': last_game.get(f'{prefix}roll_pace', 99.5),
        'roll_off_rtg': last_game.get(f'{prefix}roll_off_rtg', 113.0),
        'roll_def_rtg': last_game.get(f'{prefix}roll_def_rtg', 113.0),
        'roll_net_rtg': last_game.get(f'{prefix}roll_net_rtg', 0.0),
        'roll_efg_pct': last_game.get(f'{prefix}roll_efg_pct', 0.54),
        'roll_ts_pct': last_game.get(f'{prefix}roll_ts_pct', 0.58),
        'roll_tov_pct': last_game.get(f'{prefix}roll_tov_pct', 0.13),
        'roll_oreb_pct': last_game.get(f'{prefix}roll_oreb_pct', 0.25),
        'roll_adj_off_rtg': last_game.get(f'{prefix}adj_off_rtg', 0.0),
        'rest_days': rest_days,
        'is_b2b': is_b2b
    }

def fetch_daily_slate(target_date):
    """Pulls the NBA daily schedule via ScoreboardV2 for YYYY-MM-DD."""
    sb = scoreboardv2.ScoreboardV2(game_date=target_date)
    header_df = sb.game_header.get_data_frame()
    if header_df.empty:
        return pd.DataFrame()

    games = []
    for _, row in header_df.iterrows():
        home_abbr = TEAM_ID_TO_ABBR.get(row['HOME_TEAM_ID'])
        away_abbr = TEAM_ID_TO_ABBR.get(row['VISITOR_TEAM_ID'])
        games.append({
            'game_id': str(row['GAME_ID']),
            'gameday': target_date,
            'gametime': str(row.get('GAME_STATUS_TEXT', 'TBD')).strip(),
            'home_team': home_abbr,
            'away_team': away_abbr,
            'home_moneyline': -110,
            'away_moneyline': -110
        })
    return pd.DataFrame(games)

if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

    TARGET_SEASON = '2026-27'
    TARGET_DATE = '2026-11-04'  # Set target date for NBA slate

    PREDICTIONS_BASE = os.path.join(PROJECT_ROOT, 'predictions')
    DATE_FOLDER_NAME = f'date_{TARGET_DATE}'
    PREDICTIONS_DIR = os.path.join(PREDICTIONS_BASE, DATE_FOLDER_NAME)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    current_time_utc = datetime.now(timezone.utc)
    mt_time = current_time_utc.astimezone(ZoneInfo("America/Denver"))
    formatted_timestamp = mt_time.strftime('%I:%M %p (Mountain Time), %Y-%m-%d')

    print('=========================================================================================================')
    print(f' NBA PREDICTIVE ENGINE | RUN TIMESTAMP: {formatted_timestamp}')
    print('=========================================================================================================')

    print(f'Pulling NBA Schedule for {TARGET_DATE}...')
    target_games = fetch_daily_slate(TARGET_DATE)

    if target_games.empty:
        print('No NBA games found for the selected date.')
        exit()

    features_df = pd.read_csv(os.path.join(DATA_DIR, 'nba_model_features.csv'))

    margin_model_path = os.path.join(MODEL_DIR, 'nba_xgb_margin_model.joblib')
    total_model_path = os.path.join(MODEL_DIR, 'nba_xgb_total_points_model.joblib')

    if not os.path.exists(margin_model_path) or not os.path.exists(total_model_path):
        print(f"❌ Error: Required NBA model artifacts missing in {MODEL_DIR}.")
        exit()

    margin_model = joblib.load(margin_model_path)
    total_points_model = joblib.load(total_model_path)

    batch_payload = {
        'last_updated': current_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'season': TARGET_SEASON,
        'target_date': TARGET_DATE,
        'games_by_date': {},
    }

    processed_games_by_date = {}

    for _, game in target_games.iterrows():
        home_team_abbr = game['home_team']
        away_team_abbr = game['away_team']
        game_id = game['game_id']

        gameday = pd.to_datetime(game['gameday']).strftime('%Y-%m-%d') if pd.notna(game.get('gameday')) else 'TBD'
        formatted_time = format_gametime_utc(gameday, game.get('gametime', 'TBD'))

        home_ml = game.get('home_moneyline', -110)
        away_ml = game.get('away_moneyline', -110)
        home_raw = american_odds_to_raw_prob(home_ml)
        away_raw = american_odds_to_raw_prob(away_ml)
        vegas_implied_prob = home_raw / (home_raw + away_raw) if (home_raw + away_raw) > 0 else 0.50

        home_stats = get_team_latest_stats(features_df, home_team_abbr, TARGET_DATE)
        away_stats = get_team_latest_stats(features_df, away_team_abbr, TARGET_DATE)

        if not home_stats or not away_stats:
            continue

        rest_diff = home_stats['rest_days'] - away_stats['rest_days']

        # 1. 11-Feature Matrix for NBA Margin Model
        margin_matchup_data = pd.DataFrame([{
            'diff_pts_scored': home_stats['roll_pts_scored'] - away_stats['roll_pts_scored'],
            'diff_pts_allowed': away_stats['roll_pts_allowed'] - home_stats['roll_pts_allowed'],
            'diff_off_rtg': home_stats['roll_off_rtg'] - away_stats['roll_off_rtg'],
            'diff_def_rtg': away_stats['roll_def_rtg'] - home_stats['roll_def_rtg'],
            'diff_net_rtg': home_stats['roll_net_rtg'] - away_stats['roll_net_rtg'],
            'diff_efg_pct': home_stats['roll_efg_pct'] - away_stats['roll_efg_pct'],
            'diff_ts_pct': home_stats['roll_ts_pct'] - away_stats['roll_ts_pct'],
            'diff_tov_pct': away_stats['roll_tov_pct'] - home_stats['roll_tov_pct'],
            'diff_oreb_pct': home_stats['roll_oreb_pct'] - away_stats['roll_oreb_pct'],
            'diff_adj_off_rtg': home_stats['roll_adj_off_rtg'] - away_stats['roll_adj_off_rtg'],
            'rest_diff': rest_diff
        }])

        # 2. 20-Feature Matrix for NBA Total Points Model
        total_matchup_data = pd.DataFrame([{
            'sum_pts_scored': home_stats['roll_pts_scored'] + away_stats['roll_pts_scored'],
            'sum_pts_allowed': home_stats['roll_pts_allowed'] + away_stats['roll_pts_allowed'],
            'sum_pace': home_stats['roll_pace'] + away_stats['roll_pace'],
            'sum_off_rtg': home_stats['roll_off_rtg'] + away_stats['roll_off_rtg'],
            'sum_def_rtg': home_stats['roll_def_rtg'] + away_stats['roll_def_rtg'],
            'sum_efg_pct': home_stats['roll_efg_pct'] + away_stats['roll_efg_pct'],
            'sum_ts_pct': home_stats['roll_ts_pct'] + away_stats['roll_ts_pct'],
            'diff_pts_scored': home_stats['roll_pts_scored'] - away_stats['roll_pts_scored'],
            'diff_pts_allowed': away_stats['roll_pts_allowed'] - home_stats['roll_pts_allowed'],
            'diff_off_rtg': home_stats['roll_off_rtg'] - away_stats['roll_off_rtg'],
            'diff_def_rtg': away_stats['roll_def_rtg'] - home_stats['roll_def_rtg'],
            'diff_net_rtg': home_stats['roll_net_rtg'] - away_stats['roll_net_rtg'],
            'diff_efg_pct': home_stats['roll_efg_pct'] - away_stats['roll_efg_pct'],
            'diff_ts_pct': home_stats['roll_ts_pct'] - away_stats['roll_ts_pct'],
            'diff_tov_pct': away_stats['roll_tov_pct'] - home_stats['roll_tov_pct'],
            'diff_oreb_pct': home_stats['roll_oreb_pct'] - away_stats['roll_oreb_pct'],
            'diff_adj_off_rtg': home_stats['roll_adj_off_rtg'] - away_stats['roll_adj_off_rtg'],
            'rest_diff': rest_diff,
            'home_is_b2b': home_stats['is_b2b'],
            'away_is_b2b': away_stats['is_b2b']
        }])

        predicted_margin = float(margin_model.predict(margin_matchup_data)[0])
        STD_DEV = 12.0  # Standard NBA game margin volatility
        stats_home_win_prob = float(norm.cdf(predicted_margin / STD_DEV))
        model_home_win_prob = (0.80 * stats_home_win_prob) + (0.20 * vegas_implied_prob)

        edge = model_home_win_prob - vegas_implied_prob
        pick_abbr = home_team_abbr if model_home_win_prob > 0.5 else away_team_abbr

        model_picks_home = model_home_win_prob > 0.5
        vegas_picks_home = vegas_implied_prob >= 0.5
        same_winner = model_picks_home == vegas_picks_home

        model_winner_prob = model_home_win_prob if model_picks_home else (1.0 - model_home_win_prob)
        vegas_winner_prob = vegas_implied_prob if vegas_picks_home else (1.0 - vegas_implied_prob)
        projected_margin = round(abs(predicted_margin), 1)

        if same_winner and (model_winner_prob > 0.70) and (vegas_winner_prob >= 0.50):
            confidence_tier = 'High'
        elif same_winner and (0.60 <= model_winner_prob) and (vegas_winner_prob >= 0.50):
            confidence_tier = 'Moderate'
        else:
            confidence_tier = 'Low'

        projected_total = float(total_points_model.predict(total_matchup_data)[0])

        home_full = get_full_name(home_team_abbr)
        away_full = get_full_name(away_team_abbr)
        pick_full = get_full_name(pick_abbr)

        game_info = {
            'game_id': game_id,
            'gametime': formatted_time,
            'home_team': {'abbr': home_team_abbr, 'name': home_full, 'moneyline': home_ml},
            'away_team': {'abbr': away_team_abbr, 'name': away_full, 'moneyline': away_ml},
            'prediction': {
                'model_home_win_prob': round(model_home_win_prob, 3),
                'vegas_home_implied_prob': round(vegas_implied_prob, 3),
                'edge': round(edge, 3),
                'pick_abbr': str(pick_abbr),
                'pick_name': str(pick_full),
                'projected_margin': projected_margin,
                'projected_total_points': round(projected_total, 1),
                'confidence_tier': confidence_tier,
            },
        }

        if gameday not in processed_games_by_date:
            processed_games_by_date[gameday] = []
        processed_games_by_date[gameday].append(game_info)

    for gameday, games in sorted(processed_games_by_date.items()):
        print(f'\nDATE: {gameday}')
        batch_payload['games_by_date'][gameday] = games
        for g in games:
            away = g['away_team']['name']
            home = g['home_team']['name']
            p = g['prediction']
            pick_display = f"{p['pick_name']} (by {p['projected_margin']} pts)"
            time_slice = g["gametime"][-9:-1] if g["gametime"].endswith('Z') else g["gametime"]
            print(f'  [{time_slice:<10}] {away:>13} @ {home:<13} | Total: {p["projected_total_points"]:<5.1f} | Pick: {pick_display:<25} | Tier: {p["confidence_tier"]:<8}')

    print('\n=========================================================================================================')

    output_filename = f'nba_predictions_{TARGET_DATE}.json'
    json_path = os.path.join(PREDICTIONS_DIR, output_filename)
    with open(json_path, 'w') as f:
        json.dump(batch_payload, f, indent=2, default=int)

    print(f'✅ Exported NBA daily JSON slate (ISO-8601 UTC) to {os.path.abspath(json_path)}')
