# Heterogeneous stacked ensemble for sub-national mpox risk classification — data and code

Reproducibility package for the manuscript *"What does stacking add? A persistence-controlled evaluation of a heterogeneous ensemble for sub-national mpox risk classification"* (Glay and He).

## Contents
- `data/mpox_unit_month_panel.csv` — harmonized analytic panel, 2,835 unit-months (DRC provinces, Nigerian states, Ugandan districts). Columns: country, administrative unit, year, month, reconstructed monthly new cases (`new_cases_monthly`; may be fractional where reports were apportioned), ERA5 climate (temperature in K, soil moisture, precipitation in m, with lags 1–3), WorldPop population, Hansen tree cover and cumulative forest loss. The outcome (`outbreak`) and all features are derived in `code/prep.py` and `code/feat2.py`.
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

## Data sources and terms
Case counts were compiled from public situation reports of the DRC Institut National de Santé Publique, WHO and WHO AFRO, the Nigeria Centre for Disease Control, and the Uganda Ministry of Health. Environmental covariates: ERA5 (Copernicus), WorldPop, Hansen Global Forest Change, geoBoundaries (all public). The compiled panel is shared under CC BY 4.0; underlying sources keep their own terms. Code is MIT-licensed.

## Citation
Please cite the manuscript (reference to be added on publication).
