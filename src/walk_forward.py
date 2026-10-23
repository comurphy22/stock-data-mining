"""
Pooled walk-forward evaluation.

Replaces the per-stock, per-year 80/20 split (15-35 test days per experiment)
with one model trained on every ticker at once and evaluated month by month on
data it has never seen. This yields thousands of out-of-sample predictions, so
accuracy differences can be tested against baselines with confidence intervals.

Pipeline:
    1. build_panel(): stack each ticker's features into one DataFrame
    2. walk_forward(): expanding-window train, predict the next calendar month
    3. summarize() / compare(): metrics with date-block bootstrap CIs

The bootstrap resamples whole trading days rather than individual rows,
because predictions for different tickers on the same day are correlated
(they share the market move), and treating them as independent would make
the intervals too narrow.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Columns produced by feature_engineering.create_features() when politician
# data is supplied (basic aggregation + advanced_politician_features)
POLITICIAN_FEATURES = [
    'politician_buy_count', 'politician_trade_amount',
    'buy_count', 'sell_count', 'full_sale_count', 'partial_sale_count',
    'total_amount', 'buy_amount', 'sell_amount', 'net_trade_index',
    'net_dollar_flow', 'buy_percentage',
    'trades_last_30d', 'amount_last_30d', 'net_flow_last_30d',
    'trades_last_60d', 'amount_last_60d', 'net_flow_last_60d',
    'trades_last_90d', 'amount_last_90d', 'net_flow_last_90d',
    'trade_momentum_30d', 'dollar_momentum_30d',
    'days_since_last_trade', 'conviction_score',
]

# Value meaning "no trades" for tickers without any politician activity
POLITICIAN_FILL = {'days_since_last_trade': 180}

# Dollar-denominated columns: heavy-tailed, so compressed with a signed log
DOLLAR_FEATURES = [
    'politician_trade_amount', 'total_amount', 'buy_amount', 'sell_amount',
    'net_dollar_flow', 'amount_last_30d', 'net_flow_last_30d',
    'amount_last_60d', 'net_flow_last_60d', 'amount_last_90d',
    'net_flow_last_90d',
]

# Columns measured in price units; meaningless when pooling a $30 stock with a
# $1000 stock, so they are divided by that day's close
PRICE_LEVEL_FEATURES = ['SMA_10', 'SMA_20', 'SMA_50', 'MACD', 'MACD_signal',
                        'MACD_diff', 'ATR_14']

META_COLUMNS = ['Date', 'ticker', 'target', 'next_return']


def prepare_ticker_frame(X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                         close: pd.Series, ticker: str) -> pd.DataFrame:
    """
    Turn create_features() output for one ticker into rows ready for pooling.

    Parameters:
    -----------
    X, y, dates : outputs of feature_engineering.create_features()
    close : pd.Series
        Closing prices aligned with X's index (used to scale price-level
        features and to compute next-day returns)
    ticker : str

    Returns:
    --------
    pd.DataFrame with META_COLUMNS plus scale-free numeric features
    """
    frame = X.select_dtypes(include=[np.number]).copy()
    close = close.loc[frame.index].astype(float)

    for col in PRICE_LEVEL_FEATURES:
        if col in frame.columns:
            if col.startswith('SMA_'):
                frame[col] = frame[col] / close - 1
            else:
                frame[col] = frame[col] / close

    for col in DOLLAR_FEATURES:
        if col in frame.columns:
            frame[col] = np.sign(frame[col]) * np.log1p(frame[col].abs())

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.insert(0, 'Date', pd.to_datetime(dates.loc[frame.index]).values)
    frame.insert(1, 'ticker', ticker)
    frame['target'] = y.loc[frame.index].astype(int).values
    frame['next_return'] = (close.shift(-1) / close - 1).values
    return frame


def pool_frames(frames: list) -> pd.DataFrame:
    """
    Stack per-ticker frames. Tickers without politician trades lack those
    columns entirely; they are filled with "no activity" values rather than
    dropped so every ticker shares one feature space.
    """
    panel = pd.concat(frames, ignore_index=True, sort=False)
    for col in POLITICIAN_FEATURES:
        if col in panel.columns:
            panel[col] = panel[col].fillna(POLITICIAN_FILL.get(col, 0))
    # The last row of each ticker has a target but no next-day return
    # inside the window; keep it for classification metrics.
    return panel.sort_values(['Date', 'ticker']).reset_index(drop=True)


def feature_sets(panel: pd.DataFrame) -> dict:
    """Feature columns for each configuration being compared."""
    all_features = [c for c in panel.columns if c not in META_COLUMNS]
    politician = [c for c in all_features if c in POLITICIAN_FEATURES]
    return {
        'all': all_features,
        'no_politician': [c for c in all_features if c not in politician],
    }


def make_logreg():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000))


def make_xgboost(params: dict = None):
    import xgboost as xgb
    if params is None:
        from config import XGBOOST_PARAMS
        params = XGBOOST_PARAMS
    params = {k: v for k, v in params.items() if k != 'objective'}
    return xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                             tree_method='hist', **params)


def walk_forward(panel: pd.DataFrame, models: dict, min_train_months: int = 12,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation.

    For each calendar month after the first `min_train_months`, every model is
    fit on all rows dated strictly before that month (minus the final training
    day, whose label depends on the first test day's close) and predicts every
    row in the month.

    Parameters:
    -----------
    panel : pd.DataFrame
        Output of pool_frames()
    models : dict
        {name: (factory, feature_columns)} where factory() returns an
        unfitted sklearn-style classifier
    min_train_months : int
        Months of history required before the first test month

    Returns:
    --------
    pd.DataFrame: one row per (test row, model) with Date, ticker, target,
        next_return, model, prob_up, pred, train_end, plus the three baselines
        (always_up, train_majority, momentum) as additional models
    """
    panel = panel.sort_values(['Date', 'ticker']).reset_index(drop=True)
    months = panel['Date'].dt.to_period('M')
    test_months = sorted(months.unique())[min_train_months:]
    if not test_months:
        raise ValueError(f"Need more than {min_train_months} months of data")

    outputs = []
    for month in test_months:
        month_start = month.to_timestamp()
        train_dates = panel.loc[panel['Date'] < month_start, 'Date']
        # Embargo: drop the last training day (its target uses the first
        # test day's close)
        train_mask = panel['Date'] < train_dates.max()
        test_mask = months == month
        train, test = panel[train_mask], panel[test_mask]
        if train['target'].nunique() < 2 or test.empty:
            continue

        base = test[['Date', 'ticker', 'target', 'next_return']].copy()
        base['train_end'] = train['Date'].max()

        predictions = {}
        for name, (factory, columns) in models.items():
            model = factory()
            model.fit(train[columns], train['target'])
            predictions[name] = model.predict_proba(test[columns])[:, 1]

        majority = float(train['target'].mean() >= 0.5)
        predictions['always_up'] = np.ones(len(test))
        predictions['train_majority'] = np.full(len(test), majority)
        if 'Price_change' in test.columns:
            # Yesterday's direction carries forward (scored on a 0/1 scale)
            predictions['momentum'] = (test['Price_change'] > 0).astype(float).values

        for name, prob in predictions.items():
            out = base.copy()
            out['model'] = name
            out['prob_up'] = prob
            out['pred'] = (prob >= 0.5).astype(int)
            outputs.append(out)

        if verbose:
            print(f"   [OK] {month}: train={len(train):,} rows, test={len(test):,} rows")

    return pd.concat(outputs, ignore_index=True)


def _date_bootstrap(per_date: pd.DataFrame, value_cols: list, ratio,
                    n_boot: int, seed: int) -> np.ndarray:
    """Resample trading days with replacement; return n_boot statistics."""
    rng = np.random.default_rng(seed)
    sums = per_date[value_cols].to_numpy()
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    totals = sums[idx].sum(axis=1)  # (n_boot, len(value_cols))
    return ratio(totals)


def accuracy_ci(preds: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
                alpha: float = 0.05) -> tuple:
    """Accuracy with a date-block bootstrap confidence interval."""
    correct = (preds['pred'] == preds['target']).astype(int)
    per_date = (pd.DataFrame({'Date': preds['Date'], 'correct': correct, 'n': 1})
                .groupby('Date')[['correct', 'n']].sum())
    stats = _date_bootstrap(per_date, ['correct', 'n'],
                            lambda t: t[:, 0] / t[:, 1], n_boot, seed)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return correct.mean(), lo, hi


def summarize(predictions: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """
    One row per model: accuracy (with 95% CI), AUC, and the gross daily
    return of a long-when-predicted-up / flat-otherwise rule (no costs).
    """
    rows = []
    for name, preds in predictions.groupby('model', sort=False):
        acc, lo, hi = accuracy_ci(preds, n_boot=n_boot)
        try:
            auc = roc_auc_score(preds['target'], preds['prob_up'])
        except ValueError:
            auc = np.nan
        if preds['prob_up'].nunique() < 2:
            auc = np.nan

        # Equal-weight portfolio across tickers each day
        traded = preds.dropna(subset=['next_return'])
        daily = (traded.assign(strat=traded['pred'] * traded['next_return'])
                 .groupby('Date')[['strat', 'next_return']].mean())
        sharpe = (daily['strat'].mean() / daily['strat'].std() * np.sqrt(252)
                  if daily['strat'].std() > 0 else np.nan)

        rows.append({
            'model': name,
            'n_predictions': len(preds),
            'n_days': preds['Date'].nunique(),
            'accuracy': acc,
            'acc_ci_low': lo,
            'acc_ci_high': hi,
            'auc': auc,
            'pct_pred_up': preds['pred'].mean(),
            'gross_mean_daily_return': daily['strat'].mean(),
            'gross_sharpe': sharpe,
            'buy_hold_mean_daily_return': daily['next_return'].mean(),
        })
    return pd.DataFrame(rows)


def compare(predictions: pd.DataFrame, model_a: str, model_b: str,
            n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Paired accuracy difference (a - b) on the same test rows, with a
    date-block bootstrap CI and the share of resamples where a beats b.
    """
    a = predictions[predictions['model'] == model_a].set_index(['Date', 'ticker'])
    b = predictions[predictions['model'] == model_b].set_index(['Date', 'ticker'])
    joined = a[['pred', 'target']].join(b[['pred']].rename(columns={'pred': 'pred_b'}), how='inner')
    diff = ((joined['pred'] == joined['target']).astype(int)
            - (joined['pred_b'] == joined['target']).astype(int))
    per_date = (pd.DataFrame({'diff': diff.values, 'n': 1},
                             index=joined.index.get_level_values('Date'))
                .groupby(level=0)[['diff', 'n']].sum())
    stats = _date_bootstrap(per_date, ['diff', 'n'],
                            lambda t: t[:, 0] / t[:, 1], n_boot, seed)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {
        'model_a': model_a,
        'model_b': model_b,
        'n_predictions': len(joined),
        'accuracy_diff': diff.mean(),
        'diff_ci_low': lo,
        'diff_ci_high': hi,
        'share_boot_a_better': float((stats > 0).mean()),
    }


def per_ticker(predictions: pd.DataFrame, model: str,
               n_boot: int = 2000) -> pd.DataFrame:
    """Accuracy and CI for each ticker under one model."""
    rows = []
    preds = predictions[predictions['model'] == model]
    for ticker, group in preds.groupby('ticker'):
        acc, lo, hi = accuracy_ci(group, n_boot=n_boot)
        rows.append({'model': model, 'ticker': ticker, 'n_predictions': len(group),
                     'accuracy': acc, 'acc_ci_low': lo, 'acc_ci_high': hi,
                     'share_up_days': group['target'].mean()})
    return pd.DataFrame(rows)
