# Analysis Scripts

Scripts for running experiments and generating results.

---

## Main Validation Scripts

### `validate_multiyear.py`
Multi-year validation across 8 stocks and 2 years (16 experiments).

```bash
python scripts/validate_multiyear.py
```
**Output:** `results/multiyear_validation_results.csv`  
**Runtime:** 10-15 minutes

---

### `walk_forward_eval.py`
Pooled walk-forward evaluation: one model trained on all 8 stocks, tested month by month on unseen data (thousands of predictions instead of ~20 per stock-year). Reports accuracy with 95% confidence intervals against baselines (always-up, majority class, yesterday's direction) and compares each model with and without politician features.

```bash
python scripts/walk_forward_eval.py              # fetch data, cache to data/walk_forward_panel.csv
python scripts/walk_forward_eval.py --no-news    # if data/archive/ is not available
python scripts/walk_forward_eval.py --use-cached-panel
```
**Output:** `results/walk_forward_{summary,comparisons,per_ticker,predictions}.csv`

---

### `economic_backtest.py`
Trading simulation with transaction costs.

```bash
python scripts/economic_backtest.py
```
**Output:** `results/economic_backtest_results.csv`  
**Runtime:** 10-15 minutes

---

### `live_prediction_demo.py`
Generate real-time BUY/SELL signals.

```bash
python scripts/live_prediction_demo.py
```
**Output:** `results/daily_predictions_log.csv`  
**Runtime:** 2-3 minutes

---

## Feature Analysis

### `analyze_feature_importance.py`
Extract and rank feature importance from models.

**Output:** `results/feature_importance_rankings.csv`

---

### `validate_with_feature_selection.py`
Validate with optimized feature set (top 20 features).

---

## Model Comparison

### `compare_models.py`
Compare XGBoost vs Random Forest vs Logistic Regression.

**Output:** `results/model_comparison_results.csv`

---

### `compare_sequence_models.py`
Compare XGBoost vs LSTM vs GRU.

---

## Visualization

### `generate_visualizations.py`
Generate all result charts.

**Output:** `visualizations/*.png`

---

### `visualize_features.py`
Feature importance visualizations.

---

### `visualize_live_predictions.py`
Live prediction visualizations.

---

## Utility

### `train_financial_sentiment.py`
Train custom sentiment classifier on FinancialPhraseBank.

**Output:** `models/financial_sentiment_classifier.pkl`

---

### `daily_prediction_tracker.py`
Track daily predictions for live testing.

**Output:** `results/daily_predictions_log.csv`

---

## Recommended Workflow

```bash
# 1. Run main validation
python scripts/validate_multiyear.py

# 2. Run economic backtest
python scripts/economic_backtest.py

# 3. Generate visualizations
python scripts/generate_visualizations.py

# 4. Try live demo
python scripts/live_prediction_demo.py
```

---

## Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```
