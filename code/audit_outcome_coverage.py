"""Cross-tabulate the outcome against the coverage flag in the analysis sample (run from repo root)."""
import sys; sys.path.insert(0, 'code')
import pandas as pd
from prep import load
d, _ = load()
print(len(d), d.admin_unit.nunique())
print(pd.crosstab(d['coverage'], d['outbreak'], margins=True))
print(pd.crosstab([d['country'], d['coverage']], d['outbreak']))
