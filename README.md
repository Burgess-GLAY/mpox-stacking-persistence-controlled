# Persistence-controlled stacked ensembles for sub-national mpox activity classification — data and code

Reproducibility package for the manuscript *"A Persistence-Controlled Evaluation of Stacked Ensembles for Sub-National Mpox Activity Classification"* (Glay and He).

## Contents
- `data/mpox_unit_month_panel.csv` — harmonized panel, 2,860 unit-months (2,835 in the analysis sample after dropping rows with missing climate or forest covariates in `code/prep.py`) (DRC provinces, Nigerian states, Ugandan districts). Columns: country, administrative unit, year, month, reconstructed monthly new cases (`new_cases_monthly`; may be fractional where reports were apportioned), ERA5 climate (temperature in K, soil moisture, precipitation in m, with lags 1–3), WorldPop population, Hansen tree cover and cumulative forest loss. The outcome (`outbreak`) and all features are derived in `code/prep.py` and `code/feat2.py`.
- `code/` — all analysis scripts (see run order below).
- `res/` — stored out-of-fold predictions (`*.npy`), selected hyperparameters (`*_info.json`) and every result table (`res/tables/*.csv`) behind the manuscript.

## Run order (from the repository root)
```
pip install -r requirements.txt
bash code/queue.sh            # nested grouped CV (seeds 42-46), LOCO, ablations (hours on 2 cores)
python code/an_a.py           # tables: primary, baselines, bootstrap, ablation, regimes, calibration, thresholds
python code/an_b.py           # forward-in-time validation (Nigeria), horizons 1/3/6 months
python code/an_b2.py          # alternative forward scheme
python code/an_c.py           # final fit, SHAP level 1 and level 2
python code/exp2.py; python code/exp3.py   # ensemble-composition sensitivity (fixed hyperparameters)
python code/exp4.py           # train/test gap, label-permutation and capacity diagnostics (overfitting audit)
python code/exp5.py           # paired stack-vs-average, train-only standardization, capacity vs transfer
python code/figs_paper.py     # manuscript figures (writes PDFs to /tmp/paper/; edit OUT path)
```
Stored predictions in `res/` allow `an_*.py`, `exp5.py` (E3 part) and `figs_paper.py` to run without repeating the hours-long nested searches.

## Protocol summary
Unit-grouped stratified 5-fold CV (all months of an administrative unit in one fold), repeated over seeds 42–46; hyperparameter search inside every outer fold; stacking meta-features from a 4-fold inner grouped split; cluster bootstrap by administrative unit (1,000 resamples) for intervals and paired differences; forward-in-time validation on Nigeria; leave-one-country-out transfer.

## Outcome definition and key limitation
`outbreak = (new_cases_monthly.fillna(0) > 0)`. In the analysis sample the outcome is identical to the `coverage` flag: all 656 FULL-coverage unit-months are active, and all 2,107 PARTIAL and 72 MISSING unit-months are zero (`python code/audit_outcome_coverage.py`). Scores therefore measure reported activity, not transmission.

## Monthly case reconstruction
Cumulative or daily report values are converted to per-interval increments, divided by days elapsed and allocated to calendar months by days covered. Negative increments are floored at 0 and flagged (`had_monotonicity_flag`). Values can be fractional. DRC case type is notified or confirmed (uncertain); Nigeria and Uganda are confirmed.

## Data sources and terms
`sources/` lists the reports with URLs and SHA-256 hashes: `drc_reports_manifest.csv` (10 DRC reports used; 3 retrieved reports rejected as non-DRC or multi-disease; INSP N°54 returned HTTP 403), `nigeria_ncdc_wer_manifest.csv` (NCDC weekly epidemiological reports). The Uganda manifest has a known error (row 21 points to a Republic of Congo report) and is not included. Environmental covariates: ERA5 (Copernicus), WorldPop, Hansen Global Forest Change, geoBoundaries. The panel is shared under CC BY 4.0; underlying sources keep their own terms. Code is MIT-licensed.

## Citation
Please cite the manuscript (reference to be added on publication).
