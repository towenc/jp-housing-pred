# JP Housing Price Predictor

> 🚧 Work in progress

Predict the sale price of second-hand homes in Japan from public transaction records.

## The idea

Japan's Ministry of Land (MLIT, 国土交通省) publishes every reported real estate transaction since 2005 through the [Real Estate Information Library](https://www.reinfolib.mlit.go.jp/) (不動産情報ライブラリ). Each record has the price, location, floor area, building age, structure and more.

This project turns those records into a model that answers: **"What is this property likely to sell for?"**

## Scope

- **Property types:** used condominiums (中古マンション等) and land with a building (宅地(土地と建物))
- **Coverage:** all 47 prefectures, 2005 Q3 onward
- **Target:** `log(TradePrice)`
- **Cleaning:** drop non-market deals (auctions, related-party sales, defects, encumbrances) and sales under ¥1M

## Approach

**Data pipeline** (medallion layout, Parquet partitioned by prefecture / year / quarter):

```
MLIT API → raw (json.gz) → bronze (parquet) → silver (cleaned, typed) → gold (model-ready)
```

**Time-based split:** to mimic predicting future sales

| Split      | Years        |
|------------|--------------|
| train      | ≤ 2023       |
| validation | 2024         |
| test       | 2025 onward  |

**Models:** start with simple baselines, then build up.

1. Global mean
2. Mean by prefecture
3. Mean by municipality
4. *Next:* gradient-boosted trees on property features

**Metrics:** RMSE, MAE and R² on log price, plus two that are easier to read in yen:

- **MdAPE:** median absolute percentage error
- **PPE10:** share of predictions within ±10% of the real price

Experiments are configured in YAML (`configs/`) and tracked with MLflow.

## Status

- [x] Ingest from MLIT API
- [x] Bronze / silver / gold transforms
- [x] Train / validation / test split
- [x] Baseline models and evaluation
- [x] Config-driven training
- [ ] MLflow tracking
- [ ] Feature-based models
- [ ] Results
