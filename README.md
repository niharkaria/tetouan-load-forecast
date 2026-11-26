# Tetouan day-ahead electricity load forecast

**Live demo:** https://tetouan-load-forecast-fk5qqdskhpsseadlpsnfri.streamlit.app/
(Free hosting: if the app has been idle it shows a "gone to sleep" page. Click the button and wait a moment.)

Forecast the next 24 hourly values of Tetouan's total electricity load, made at midnight using only past load.

## Result in one paragraph

A LightGBM model that predicts load as a ratio of the past-week average, using calendar features and three lagged-demand ratios (no weather), reduced mean absolute error (MAE) by **13.6%** against a day-type naive baseline (1,822 vs 2,109) over six walk-forward folds covering Jul 16 to Dec 30, 2017. It beat the baseline in 5 of 6 folds. The gain is **modest and not statistically clear outside Eid al-Adha week**: the block-bootstrap 95% interval for the mean daily gain is [59, 555] over all days but [-19, 269] without that week. The final model was selected among about 12 variants on the same folds, so the true gain is probably smaller. No business-savings claim is made.

## Data

- Salam, A. & El Hibaoui, A. (2018). *Power Consumption of Tetouan City* [Dataset]. UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city
- 52,416 rows at 10-minute spacing, 2017-01-01 to 2017-12-30 (364 days, so no Dec 31): total load in three zones plus weather columns.
- Checked: month/day date format (first part reaches 12, second reaches 31), no missing or duplicate timestamps, all steps exactly 10 minutes.
- Units are undocumented, so values are treated as relative power units and MAE is in the same units.
- Resampled to hourly (6 readings per hour, 8,736 rows). **Target = total of the three zones.** Per-zone forecasts are future work.
- Zone 3 has many IQR outliers (1,191), concentrated in summer evenings. They were kept as real peaks.

## Problem framing and leakage control

- Forecast origin is **midnight**: for day D, everything up to 23:00 of D-1 is known. Every demand feature is at least 24 hours old (lags 24 and 168, a 7-day mean shifted by 24 hours, yesterday's daily mean).
- Calendar features (hour as sin/cos, weekend flag, weekday) are always known in advance. Month and day-of-year are deliberately **not** used, because one year of data lets a model memorise that year.
- Weather is **not used** in the final model. Adding actual temperature changed results very little, and it would have created a train/serve mismatch, because the app would need a forecast temperature while training used actual temperature.
- The serving function was tested against the batch feature pipeline (difference 0.0), so the app and the notebooks build identical features.

## Validation

- No random splits. Six walk-forward folds (`TimeSeriesSplit`, 4-week test windows, expanding training set). The first test fold starts Jul 16, so **Jan to Jun were never tested**.
- Baselines: same hour yesterday, same hour last week, and a **day-type naive** (Tue-Fri copy yesterday, Mon/Sat/Sun copy the same day last week). The day-type naive is the one to beat.
- Metrics: MAE (primary, easy to read), RMSE and MAPE (reported in notebook 03). MAPE is shown next to MAE because a 30% higher summer load raises absolute error even when the model is equally good.

## Results

Mean MAE over the six folds (lower is better):

| Model | Mean MAE | Note |
|---|---|---|
| Day-type naive baseline | 2,109 | |
| Linear regression (with weather) | 1,934 | beats baseline in 4 of 6 folds |
| Random forest (no weather) | 2,048 | |
| LightGBM, raw target (no weather) | 2,167 | |
| LightGBM, ratio target, with weather | 1,886 | |
| **LightGBM, ratio target, no weather (final)** | **1,822** | beats baseline in 5 of 6 folds; loses fold 1 (2,406 vs 2,307) |
| GRU / LSTM | 3,479 / 3,632 | worse than baseline in all six folds |

Held-out demo days (Dec 3 to 30, never seen in training): model 1,093 vs baseline 1,281 (-14.7%). The model won on 17 of 28 days. **One day, Dec 9, supplies about 59% of the gain**, and without it the improvement is about 7%.

## Error analysis

- **Eid al-Adha (fold 2, Aug 13 to Sep 9)** is the hardest period (baseline MAE 4,082). Errors concentrate on Sep 1 to 4. Load dropped about 25,000 at an unchanged temperature, so weather does not explain it. Eid al-Adha fell around Sep 1, 2017, which fits, but one year of data cannot prove the cause or teach a model about holidays.
- **Dec 9 (a Saturday):** the baseline copied Dec 2, a low Saturday (daily mean 60,104 against 64,704 on Dec 9), so it under-predicted in every hour by about 4,600 on average. The model blends several lags and a weekly mean, so one odd day does not dominate it.
- **Where the model helps:** afternoons (hours 12 to 19), weekends, Mondays and September. **Where it does not:** Tuesday to Thursday (the baseline is hard to beat there), the 5 to 7 am ramp-up, August, and the hottest temperature quintile.
- Worst absolute errors for both are at midday (hours 10 to 15).

## Interpretability

Grouped permutation importance (LightGBM, ratio model): past demand is far above day type, hour and weather. Weather is about zero, and adding 2 degC of random noise to temperature changed MAE by only 12. Caveat: the hour score is low only because the ratio lags already carry the daily shape, and random noise is not the same as systematic forecast error.

## What I tried that did not work

- **Weather features.** I added temperature and a "heat above 20 degC" feature. For linear regression and random forest the results barely moved (under 1%), because yesterday's load already reflects the weather. In the final ratio setup, dropping weather was even 3.4% better (1,822 vs 1,886), so I dropped it. A bonus: the app needs no weather forecast.
- **Predicting the difference from the baseline** (a "residual" target). It fixed one fold for LightGBM but made three others worse, so overall it changed nothing. For linear regression it gives exactly the same answer as the normal model, so it adds nothing there.
- **Scaling last week's copy by a week-on-week ratio.** It helped only in the Eid fold and was about 14% worse everywhere else. I had built it around one event.
- **GRU and LSTM.** They lost to the simple baseline in all six folds. A training-error check showed they were underfitting, not overfitting. When I made them start from the baseline, they improved it by only about 2% on a validation block, while LightGBM improved it by 14%, so I stopped before the six-fold run (I had set that rule before looking at the results). With only 189 to 336 daily samples, this shows they did not work in my setup. It does not show that deep learning cannot work here.
- **No model beat the baseline in every fold.** My own success rule asked for that, so the honest summary is "better on average, not in every fold".

## Limitations

- One year of data: yearly seasonality and holidays cannot be learned or validated, and Jan to Jun were not tested.
- The model was chosen among about 12 variants on the same folds, so reported gains are optimistic.
- Evaluation used actual past load for the inputs. Weather is not used, so there is no forecast-vs-actual temperature issue.
- Total load only, no per-zone models. Units unknown.
- The app needs at least 8 days (192 hours) of hourly history ending at 23:00, and the model only works on data at Tetouan's scale.
- The demo offers only Dec 3 to 30 because those are the only days the model never trained on.

## Run it

```bash
conda create -n tetouan python=3.11 pip -y && conda activate tetouan
pip install -r requirements-dev.txt          # full environment (notebooks, PyTorch)
cd notebooks && jupyter notebook             # run 01 first, it creates data/processed/hourly.csv
streamlit run app.py                          # needs only requirements.txt
```

## Repository layout

```
app.py                  Streamlit app (backtest demo and upload mode)
src/forecast_lib.py     shared features, CV helpers, serving function (used by notebooks and app)
models/                 saved LightGBM model (trained until 2017-12-02)
data/raw, data/demo     original CSV and the 8+ days of demo history
notebooks/              01 data, 02 EDA, 03 baselines, 04 models, 05 importance and error analysis,
                        06 GRU/LSTM, 07 final model export
requirements.txt        pinned packages for the app only
requirements-dev.txt    full environment
```
