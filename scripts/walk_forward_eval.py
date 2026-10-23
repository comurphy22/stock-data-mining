"""
Pooled Walk-Forward Evaluation

Trains one model on all validation stocks together and tests it month by month
on unseen data, instead of a separate 80/20 split per stock-year (which left
only 15-35 test days per experiment). Reports accuracy with confidence
intervals against simple baselines, and compares the full feature set with
the same model trained without politician features.

Usage:
    python scripts/walk_forward_eval.py
    python scripts/walk_forward_eval.py --start 2018-01-01 --end 2020-12-31
    python scripts/walk_forward_eval.py --use-cached-panel   # skip data fetching

Outputs (results/):
    walk_forward_summary.csv       accuracy/AUC/CI per model and baseline
    walk_forward_comparisons.csv   paired accuracy differences
    walk_forward_per_ticker.csv    per-stock accuracy with CIs
    walk_forward_predictions.csv   every out-of-sample prediction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from datetime import datetime

import pandas as pd

from config import VALIDATION_STOCKS, DATA_DIR, RESULTS_DIR, check_api_keys
from walk_forward import (
    prepare_ticker_frame, pool_frames, feature_sets, make_logreg, make_xgboost,
    walk_forward, summarize, compare, per_ticker
)

# Calendar days fetched before --start so 50-day SMAs and 60-day betas are
# populated on the first evaluated day
WARMUP_DAYS = 120
PANEL_CACHE = DATA_DIR / 'walk_forward_panel.csv'


def build_panel(tickers: list, start: str, end: str, use_news: bool) -> pd.DataFrame:
    from data_loader import (fetch_stock_data, fetch_politician_trades,
                             fetch_historical_news_kaggle, aggregate_daily_sentiment)
    from feature_engineering import create_features

    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
    frames, trade_counts = [], {}

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] {ticker}")
        stock = fetch_stock_data(ticker, fetch_start, end)
        stock['Date'] = pd.to_datetime(stock['Date']).dt.tz_localize(None).dt.normalize()

        news_sentiment = None
        if use_news:
            news = fetch_historical_news_kaggle(ticker, fetch_start, end)
            news_sentiment = aggregate_daily_sentiment(news)

        trades = fetch_politician_trades(ticker)
        if not trades.empty:
            trades['date'] = pd.to_datetime(trades['date'])
            trades = trades[trades['date'] <= end]
        trade_counts[ticker] = int((trades['date'] >= start).sum()) if not trades.empty else 0

        X, y, dates = create_features(stock, news_sentiment,
                                      trades if not trades.empty else None,
                                      ticker, add_market_features=True)
        close = dates.map(stock.set_index('Date')['Close'])
        frame = prepare_ticker_frame(X, y, dates, close, ticker)
        frames.append(frame[frame['Date'] >= start])

    print("\nPolitician trades in evaluation window:")
    for ticker, count in trade_counts.items():
        print(f"   {ticker}: {count}")
    if sum(trade_counts.values()) == 0:
        raise SystemExit("[ERROR] No politician trades loaded; the with/without "
                         "comparison would be meaningless. Check QUIVER_API_KEY.")

    return pool_frames(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tickers', nargs='+', default=VALIDATION_STOCKS)
    parser.add_argument('--start', default='2018-01-01')
    parser.add_argument('--end', default='2020-12-31')
    parser.add_argument('--min-train-months', type=int, default=12)
    parser.add_argument('--no-news', action='store_true',
                        help='Skip news sentiment (e.g. if data/archive/ is absent)')
    parser.add_argument('--use-cached-panel', action='store_true',
                        help=f'Load features from {PANEL_CACHE.name} instead of fetching')
    parser.add_argument('--n-boot', type=int, default=2000)
    args = parser.parse_args()

    print("=" * 80)
    print("POOLED WALK-FORWARD EVALUATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers: {args.tickers}")
    print(f"Window: {args.start} to {args.end}, first {args.min_train_months} months train-only\n")

    if args.use_cached_panel:
        panel = pd.read_csv(PANEL_CACHE, parse_dates=['Date'])
        panel = panel[panel['ticker'].isin(args.tickers)]
        print(f"[OK] Loaded cached panel: {len(panel):,} rows")
    else:
        if not check_api_keys():
            sys.exit(1)
        panel = build_panel(args.tickers, args.start, args.end, use_news=not args.no_news)
        PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(PANEL_CACHE, index=False)
        print(f"\n[OK] Cached panel to {PANEL_CACHE}")

    sets = feature_sets(panel)
    print(f"\nPanel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['Date'].nunique()} days")
    print(f"Features: {len(sets['all'])} total, "
          f"{len(sets['all']) - len(sets['no_politician'])} politician")
    print(f"Share of up days: {panel['target'].mean():.2%}")

    models = {}
    for set_name, columns in sets.items():
        models[f'logreg_{set_name}'] = (make_logreg, columns)
        models[f'xgboost_{set_name}'] = (make_xgboost, columns)

    print("\nRunning walk-forward...")
    predictions = walk_forward(panel, models, min_train_months=args.min_train_months)

    summary = summarize(predictions, n_boot=args.n_boot)
    comparisons = pd.DataFrame(
        [compare(predictions, f'{m}_all', f'{m}_no_politician', n_boot=args.n_boot)
         for m in ('logreg', 'xgboost')]
        + [compare(predictions, name, 'always_up', n_boot=args.n_boot)
           for name in models]
    )
    tickers = pd.concat([per_ticker(predictions, name, n_boot=args.n_boot)
                         for name in ('xgboost_all', 'logreg_all', 'always_up')])

    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / 'walk_forward_summary.csv', index=False)
    comparisons.to_csv(RESULTS_DIR / 'walk_forward_comparisons.csv', index=False)
    tickers.to_csv(RESULTS_DIR / 'walk_forward_per_ticker.csv', index=False)
    predictions.to_csv(RESULTS_DIR / 'walk_forward_predictions.csv', index=False)

    pd.set_option('display.width', 200)
    print(f"\n{'=' * 80}\nRESULTS (out-of-sample, {predictions['Date'].nunique()} test days)\n{'=' * 80}")
    print(summary[['model', 'n_predictions', 'accuracy', 'acc_ci_low', 'acc_ci_high',
                   'auc', 'pct_pred_up', 'gross_sharpe']].round(4).to_string(index=False))
    print("\nPaired accuracy differences (95% CI, resampling trading days):")
    print(comparisons.round(4).to_string(index=False))
    print("\nPer-ticker accuracy (xgboost_all):")
    print(tickers[tickers['model'] == 'xgboost_all'].round(4).to_string(index=False))
    print("\nA difference is only meaningful if its CI excludes 0.")
    print(f"Saved results to {RESULTS_DIR}/walk_forward_*.csv")


if __name__ == '__main__':
    main()
