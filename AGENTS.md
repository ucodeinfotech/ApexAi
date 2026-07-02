# Project: High Gainer Classifier

**Goal**: Binary classifier to predict next-day open-to-close return > 2% for Indian stocks.

**Status**: COMPLETED - All 14 phases done. Final report at `FINAL_PROJECT_REPORT.pdf`.

## Key Results
- **Best AUC**: 0.6545 (Phase 8 baseline, 35 features, LightGBM with class weights)
- **Final AUC**: 0.6280 [95%CI: 0.622-0.634] (cluster-enhanced ensemble, 229 features)
- **Best recall**: 57.3% (cluster-specific model at th=0.15, F1=0.237)
- **Max lift**: 1.82x (th=0.55, precision-focused)

## Key Files
- `FINAL_PROJECT_REPORT.pdf` - Comprehensive final report
- `final_model/final_model_results.csv` - Latest run results
- `final_eval/` - Class-weights evaluation (best single model)
- `improvement_results/` - Phase 8 improvement pipeline
- `feature_selection_results/` - Phase 8 feature selection
- `phase9_split/` - Walkforward fold definitions
- `phase6_deep_cleaning/` - Data cleaning outputs
- `deep_analysis_report/` - EDA + time series analysis
- `cleaned_features.parquet` - Cleaned dataset
- `improved_features.parquet` - Post-filter + interactions dataset

## Key Decisions
- Class weights > SMOTE (SMOTE destroys calibration)
- LightGBM > XGBoost (0.645 vs 0.595 avg AUC)
- Single-stock models underperform pooled
- Cluster models improve per-stock but similar overall AUC
- 35 consensus features sufficient; adding more doesn't help

## Next Steps (future work)
1. Add fundamental/sentiment/options data
2. Try regression instead of classification
3. Deep learning (LSTM/Transformer)
4. Hierarchical: vol day → gain direction
