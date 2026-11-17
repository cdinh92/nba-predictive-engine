import os
import joblib
import pandas as pd
import xgboost as xgb
import optuna
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss

MARGIN_FEATURES = [
    'diff_pts_scored', 'diff_pts_allowed',
    'diff_off_rtg', 'diff_def_rtg', 'diff_net_rtg',
    'diff_efg_pct', 'diff_ts_pct', 'diff_tov_pct',
    'diff_oreb_pct', 'diff_adj_off_rtg', 'rest_diff'
]

TOTAL_FEATURES = [
    'sum_pts_scored', 'sum_pts_allowed', 'sum_pace',
    'sum_off_rtg', 'sum_def_rtg', 'sum_efg_pct', 'sum_ts_pct',
    'diff_pts_scored', 'diff_pts_allowed',
    'diff_off_rtg', 'diff_def_rtg', 'diff_net_rtg',
    'diff_efg_pct', 'diff_ts_pct', 'diff_tov_pct',
    'diff_oreb_pct', 'diff_adj_off_rtg',
    'rest_diff', 'home_is_b2b', 'away_is_b2b',
    'home_is_3_in_4', 'away_is_3_in_4' # NEW FATIGUE METRICS
]

def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=30):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'reg:absoluteerror',
            'random_state': 42
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def train_models(df):
    print("Prepping NBA data for XGBoost Margin and Total Points Regressors...")
    df['home_margin'] = df['home_score'] - df['away_score']
    df['total_points'] = df['home_score'] + df['away_score']
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

    clean_df = df.dropna(subset=TOTAL_FEATURES).copy()

    latest_season = clean_df['season'].max()
    train_df = clean_df[clean_df['season'] != latest_season].copy()
    test_df = clean_df[clean_df['season'] == latest_season].copy()

    if len(test_df) < 50:
        split_idx = int(len(clean_df) * 0.8)
        train_df = clean_df.iloc[:split_idx].copy()
        test_df = clean_df.iloc[split_idx:].copy()

    print(f"Training on {len(train_df)} games | Evaluating on {len(test_df)} out-of-sample games...")
    print("\n🔍 Running Optuna Hyperparameter Tuning for Margin Model...")
    
    best_margin_params = optimize_hyperparameters(
        train_df[MARGIN_FEATURES], train_df['home_margin'],
        test_df[MARGIN_FEATURES], test_df['home_margin'],
        n_trials=30
    )
    
    print("\n🔍 Running Optuna Hyperparameter Tuning for Total Points Model...")
    best_total_params = optimize_hyperparameters(
        train_df[TOTAL_FEATURES], train_df['total_points'],
        test_df[TOTAL_FEATURES], test_df['total_points'],
        n_trials=30
    )

    # 1. Train Margin Model with Best Params
    best_margin_params['objective'] = 'reg:absoluteerror'
    margin_model = xgb.XGBRegressor(**best_margin_params)
    margin_model.fit(train_df[MARGIN_FEATURES], train_df['home_margin'])

    margin_preds = margin_model.predict(test_df[MARGIN_FEATURES])
    STD_DEV = test_df['home_margin'].std() if len(test_df) > 30 else 12.0
    prob_preds = norm.cdf(margin_preds / STD_DEV)
    binary_preds = (prob_preds > 0.5).astype(int)

    mae_margin = mean_absolute_error(test_df['home_margin'], margin_preds)
    acc = accuracy_score(test_df['home_win'], binary_preds)
    brier = brier_score_loss(test_df['home_win'], prob_preds)

    # 2. Train Total Points Model with Best Params
    best_total_params['objective'] = 'reg:absoluteerror'
    total_model = xgb.XGBRegressor(**best_total_params)
    total_model.fit(train_df[TOTAL_FEATURES], train_df['total_points'])
    total_preds = total_model.predict(test_df[TOTAL_FEATURES])
    mae_total = mean_absolute_error(test_df['total_points'], total_preds)

    print("\n==========================================")
    print("       NBA REGRESSION MODEL RESULTS       ")
    print("==========================================")
    print(f"Margin MAE (Optimized)     : {mae_margin:.2f} points")
    print(f"Win/Loss Accuracy          : {acc * 100:.2f}%")
    print(f"Brier Score                : {brier:.4f}")
    print(f"Totals MAE (Optimized)     : {mae_total:.2f} points")
    print("==========================================\n")

    # Fit final models on full dataset for live deployment
    margin_model.fit(clean_df[MARGIN_FEATURES], clean_df['home_margin'])
    total_model.fit(clean_df[TOTAL_FEATURES], clean_df['total_points'])

    return margin_model, total_model

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    input_path = os.path.join(DATA_DIR, "nba_model_features.csv")
    if not os.path.exists(input_path):
        print("Error: nba_model_features.csv not found! Run 02_nba_feature_engineering.py first.")
    else:
        df = pd.read_csv(input_path)
        margin_model, total_model = train_models(df)

        margin_path = os.path.join(MODEL_DIR, "nba_xgb_margin_model.joblib")
        total_path = os.path.join(MODEL_DIR, "nba_xgb_total_points_model.joblib")
        joblib.dump(margin_model, margin_path)
        joblib.dump(total_model, total_path)
        print(f"✅ Saved Optimized NBA Margin & Total Points models to {MODEL_DIR}")