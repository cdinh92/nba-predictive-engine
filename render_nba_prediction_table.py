from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import os
import re
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR
PREDICTIONS_BASE = os.path.join(PROJECT_ROOT, 'predictions')

target_date = '2026-11-04'
predictor_path = os.path.join(SCRIPT_DIR, '07_predict_nba_slate.py')

if os.path.exists(predictor_path):
    with open(predictor_path, 'r') as f:
        content = f.read()
        match = re.search(r"TARGET_DATE\s*=\s*['\"]([^'\"]+)['\"]", content)
        if match:
            target_date = match.group(1)
            print(f"✅ Automatically synced target date to {target_date} from 07_predict_nba_slate.py")

DATE_FOLDER_NAME = f'date_{target_date}'
PREDICTIONS_DIR = os.path.join(PREDICTIONS_BASE, DATE_FOLDER_NAME)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

JSON_FILENAME = f'nba_predictions_{target_date}.json'
json_path = os.path.join(PREDICTIONS_DIR, JSON_FILENAME)

if not os.path.exists(json_path):
    print(f'❌ Error: Could not find prediction file at {json_path}')
    exit()

with open(json_path, 'r') as f:
    data = json.load(f)

games_by_date = data.get('games_by_date', {})

high_tier, mod_tier, low_tier = [], [], []

et_tz = ZoneInfo("America/New_York")
mt_tz = ZoneInfo("America/Denver")

def format_to_mountain_time(gameday_str, raw_time_str):
    clean_date = str(gameday_str).split('T')[0].split(' ')[0]
    if not raw_time_str or raw_time_str == 'TBD':
        try:
            dt_date = datetime.strptime(clean_date, '%Y-%m-%d')
            return f"{dt_date.strftime('%b %d')} • TBD"
        except ValueError:
            return f"{clean_date} • TBD"

    raw_str = str(raw_time_str).strip()

    iso_match = re.match(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?', raw_str)
    if iso_match:
        yr, mo, dy, hr, mi, sc = iso_match.groups()
        sc = sc if sc else '00'
        try:
            dt_utc = datetime(int(yr), int(mo), int(dy), int(hr), int(mi), int(sc), tzinfo=timezone.utc)
            dt_mt = dt_utc.astimezone(mt_tz)
            return f"{dt_mt.strftime('%b %d')} • {dt_mt.strftime('%I:%M %p').lstrip('0')}"
        except Exception:
            pass

    et_match = re.search(r'(\d{1,2}:\d{2})\s*(am|pm)', raw_str, re.IGNORECASE)
    if et_match:
        time_part = f"{et_match.group(1)} {et_match.group(2).upper()}"
        try:
            dt_et = datetime.strptime(f"{clean_date} {time_part}", "%Y-%m-%d %I:%M %p").replace(tzinfo=et_tz)
            dt_mt = dt_et.astimezone(mt_tz)
            return f"{dt_mt.strftime('%b %d')} • {dt_mt.strftime('%I:%M %p').lstrip('0')}"
        except ValueError:
            pass

    try:
        dt_date = datetime.strptime(clean_date, '%Y-%m-%d')
        return f"{dt_date.strftime('%b %d')} • {raw_str}"
    except ValueError:
        return f"{clean_date} • {raw_str}"

for gameday, games in games_by_date.items():
    for g in games:
        away_full = g['away_team']['name']
        home_full = g['home_team']['name']
        matchup_str = f'{away_full} @ {home_full}'
        datetime_display = format_to_mountain_time(gameday, g.get('gametime', 'TBD'))

        p = g['prediction']
        model_prob = f"{p['model_home_win_prob']*100:.1f}%"
        vegas_prob = f"{p['vegas_home_implied_prob']*100:.1f}%"
        edge_val = f"{p['edge']*100:+.1f}%"

        pick_name = p['pick_name']
        projected_margin = p.get('projected_margin', 0.0)
        pick = f"{pick_name} (+{projected_margin})" if projected_margin > 0 else pick_name
        projected_total = f"{p.get('projected_total_points', 0.0):.1f}"
        tier = p['confidence_tier']

        row_item = (matchup_str, datetime_display, model_prob, vegas_prob, edge_val, projected_total, pick)

        if tier == 'High':
            high_tier.append(row_item)
        elif tier == 'Moderate':
            mod_tier.append(row_item)
        else:
            low_tier.append(row_item)

tiers_data = [
    ('HIGH CONFIDENCE', high_tier, '#d4edda', '#155724'),
    ('MODERATE CONFIDENCE', mod_tier, '#fff3cd', '#856404'),
    ('LOW CONFIDENCE', low_tier, '#f8d7da', '#721c24'),
]

fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('off')

table_data = []
row_styles = []

for tier_title, matches, bg_color, text_color in tiers_data:
    table_data.append([tier_title, '', '', '', '', '', ''])
    row_styles.append(('header', bg_color, text_color))

    if not matches:
        table_data.append(['No games in this tier', '', '', '', '', '', ''])
        row_styles.append(('data', '#ffffff', '#6c757d'))
    else:
        for match, datetime_str, model, vegas, edge, total, pick in matches:
            table_data.append([match, datetime_str, model, vegas, edge, total, pick])
            row_styles.append(('data', bg_color, '#ffffff'))

table = ax.table(
    cellText=table_data,
    colLabels=['Matchup', 'Date & Time (MT)', 'Model', 'Vegas', 'Edge', 'O/U Total', 'Pick'],
    cellLoc='center',
    loc='center',
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.3)

col_widths = [0.24, 0.20, 0.10, 0.10, 0.09, 0.11, 0.16]
for i, col in enumerate(col_widths):
    for row in range(len(table_data) + 1):
        table[(row, i)].set_width(col)

for j in range(7):
    header_cell = table[(0, j)]
    header_cell.set_facecolor('#343a40')
    header_cell.set_text_props(weight='bold', color='#ffffff', size=11)

for idx, (row_type, bg_color, color_val) in enumerate(row_styles):
    row_idx = idx + 1
    if row_type == 'header':
        for j in range(7):
            cell = table[(row_idx, j)]
            cell.set_facecolor(bg_color)
            if j == 0:
                cell.set_text_props(weight='bold', color=color_val, size=10.5, ha='left')
                cell.get_text().set_x(0.04)
            else:
                cell.get_text().set_text('')
    else:
        for j in range(7):
            cell = table[(row_idx, j)]
            cell.set_facecolor('#ffffff')
            cell.set_text_props(color='#212529', size=10)

ax.set_title(
    f'NBA Slate ({target_date}) Predictions & Total Points Slices',
    weight='bold',
    size=15,
    pad=10,
    y=1.02,
)

output_image_name = f'nba_predictions_{target_date}.png'
output_image_path = os.path.join(PREDICTIONS_DIR, output_image_name)
plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0.15)
print(f'✅ Successfully generated and saved NBA table image (Mountain Time) to: {os.path.abspath(output_image_path)}')