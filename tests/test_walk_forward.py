"""
Tests for the pooled walk-forward evaluation, using synthetic data only
(no network or API keys needed).

To run tests:
    pytest tests/test_walk_forward.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import create_features
from walk_forward import (
    prepare_ticker_frame, pool_frames, feature_sets, make_logreg, make_xgboost,
    walk_forward, summarize, compare, per_ticker
)


def synthetic_panel(signal_strength=0.0, n_tickers=4, seed=0):
    """Two years of daily rows; `net_trade_index` predicts the target when
    signal_strength > 0, everything else is noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-01', '2019-12-31')
    frames = []
    for t in range(n_tickers):
        n = len(dates)
        signal = rng.normal(size=n)
        logit = signal_strength * signal
        target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        frames.append(pd.DataFrame({
            'Date': dates,
            'ticker': f'T{t}',
            'Price_change': rng.normal(0, 0.01, n),
            'RSI': rng.uniform(20, 80, n),
            'net_trade_index': signal,
            'target': target,
            'next_return': np.where(target == 1, 0.01, -0.01),
        }))
    return pool_frames(frames)


def run(panel, min_train_months=12):
    sets = feature_sets(panel)
    models = {
        'logreg_all': (make_logreg, sets['all']),
        'logreg_no_politician': (make_logreg, sets['no_politician']),
    }
    return walk_forward(panel, models, min_train_months=min_train_months, verbose=False)


class TestFeatureSets:

    def test_no_politician_excludes_politician_columns(self):
        sets = feature_sets(synthetic_panel())
        assert 'net_trade_index' in sets['all']
        assert 'net_trade_index' not in sets['no_politician']
        assert 'Price_change' in sets['no_politician']

    def test_metadata_never_used_as_features(self):
        sets = feature_sets(synthetic_panel())
        for column in ('Date', 'ticker', 'target', 'next_return'):
            assert column not in sets['all']


class TestWalkForward:

    def test_training_data_precedes_test_month(self):
        predictions = run(synthetic_panel())
        test_month_start = predictions['Date'].dt.to_period('M').dt.to_timestamp()
        assert (predictions['train_end'] < test_month_start).all()

    def test_last_pre_month_day_is_embargoed(self):
        panel = synthetic_panel()
        predictions = run(panel)
        first = predictions['Date'].min()
        last_before = panel.loc[panel['Date'] < first.to_period('M').to_timestamp(), 'Date'].max()
        assert (predictions.loc[predictions['Date'] == first, 'train_end'] < last_before).all()

    def test_only_months_after_warmup_are_tested(self):
        predictions = run(synthetic_panel(), min_train_months=12)
        assert predictions['Date'].min() >= pd.Timestamp('2019-01-01')

    def test_baselines_included(self):
        models = set(run(synthetic_panel())['model'])
        assert {'always_up', 'train_majority', 'momentum'} <= models

    def test_detects_planted_signal(self):
        predictions = run(synthetic_panel(signal_strength=2.0))
        result = compare(predictions, 'logreg_all', 'logreg_no_politician', n_boot=500)
        assert result['diff_ci_low'] > 0.1

    def test_no_false_signal_on_noise(self):
        predictions = run(synthetic_panel(signal_strength=0.0))
        result = compare(predictions, 'logreg_all', 'logreg_no_politician', n_boot=500)
        assert result['diff_ci_low'] < 0 < result['diff_ci_high']

    def test_xgboost_factory_uses_config_params(self):
        model = make_xgboost()
        assert model.get_params()['max_depth'] == 2


class TestSummaries:

    def test_ci_brackets_accuracy(self):
        summary = summarize(run(synthetic_panel(signal_strength=1.0)), n_boot=500)
        assert (summary['acc_ci_low'] <= summary['accuracy']).all()
        assert (summary['accuracy'] <= summary['acc_ci_high']).all()

    def test_always_up_accuracy_is_share_of_up_days(self):
        predictions = run(synthetic_panel())
        always_up = predictions[predictions['model'] == 'always_up']
        summary = summarize(predictions, n_boot=200).set_index('model')
        assert summary.loc['always_up', 'accuracy'] == pytest.approx(always_up['target'].mean())

    def test_per_ticker_covers_every_ticker(self):
        result = per_ticker(run(synthetic_panel()), 'logreg_all', n_boot=200)
        assert sorted(result['ticker']) == ['T0', 'T1', 'T2', 'T3']


class TestPrepareTickerFrame:

    @pytest.fixture
    def stock(self):
        rng = np.random.default_rng(1)
        dates = pd.bdate_range('2018-01-01', '2018-12-31')
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        return pd.DataFrame({'Date': dates, 'Open': close, 'High': close * 1.01,
                             'Low': close * 0.99, 'Close': close,
                             'Volume': rng.integers(1e6, 2e6, len(dates))})

    def test_scales_price_levels_and_aligns_returns(self, stock):
        X, y, dates = create_features(stock, add_market_features=False)
        close = dates.map(stock.set_index('Date')['Close'])
        frame = prepare_ticker_frame(X, y, dates, close, 'TEST')

        # SMA_10 becomes distance from close, not a dollar level
        assert frame['SMA_10'].abs().max() < 0.5
        # next_return agrees with the up/down target
        known = frame.dropna(subset=['next_return'])
        assert ((known['next_return'] > 0).astype(int) == known['target']).all()

    def test_pooling_fills_missing_politician_columns(self, stock):
        X, y, dates = create_features(stock, add_market_features=False)
        close = dates.map(stock.set_index('Date')['Close'])
        no_trades = prepare_ticker_frame(X, y, dates, close, 'A')
        with_trades = no_trades.assign(ticker='B', days_since_last_trade=5.0)

        panel = pool_frames([no_trades, with_trades])
        a = panel[panel['ticker'] == 'A']
        assert (a['days_since_last_trade'] == 180).all()
