# Pulsar Stochastic Kick Radius

Analysis code, Jupyter notebooks, data and results for **A 9 km Effective Kick Radius for Pulsars: Constraints from a Stochastic Natal-Kick Framework**, by Zheng Li, Zhi-Qiang You, Xiaojin Liu and Xing-Jiang Zhu.

The package includes 12 population-model configurations for the full sample of 77 pulsars and the young subsample of 47 pulsars, four full/null Bayesian evidence comparisons, the single-kick comparison and the four manuscript figures. A supplementary calculation evaluates the LSR distance-prior sensitivity with the Lorimer et al. (2006) Model C prior.

## Getting started

Use Python 3.13.9 and install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/verify_archive.py
python scripts/summarize_saved.py
python scripts/run_figures.py
```

The summary and figure commands read the supplied results. Generated files are written to `reproduced/`. Figure generation can take several minutes; verification and summary generation are faster. These commands use the data provided in the repository.

## Contents

| Path | Contents |
|---|---|
| `data/` | Observational catalog, selected source lists, periods, LSR velocities and velocity-error samples |
| `src/mcmc/` | Twelve MCMC model configurations |
| `src/evidence/` | Four full/null dynesty evidence workflows |
| `src/single_kick/` | Single-kick comparison model |
| `src/lsr_sensitivity/` | Lorimer Model C distance-prior sensitivity fit |
| `results/` | Posterior samples, evidence results, parameter summaries and execution logs |
| `results/portable_npz/` | Posterior samples and column names in NumPy NPZ format |
| `Figures/` | Manuscript PDFs and plotting scripts |
| `notebooks/` | Analysis overview, figure reproduction, LSR preprocessing and single-kick notebooks |
| `notebooks/html/` | Standalone HTML copies with saved outputs |
| `scripts/` | Data preparation, summary, figure and verification entry points |
| `environment/` | Python package versions and environment files |
| `provenance/` | File checksums and posterior format definitions |
| `docs/` | Data dictionary, figure/table mapping and references |

`docs/ARTICLE_MAP.md` maps each manuscript figure and table to its files. `docs/DATA_DICTIONARY.md` describes units, source ordering and file formats.

## Notebooks

Open the notebooks in Jupyter, or read the HTML files in a web browser:

- `00_archive_overview.ipynb`: input samples and posterior/evidence summaries.
- `01_reproduce_figures.ipynb`: four manuscript figures and the plotting command.
- `02_prepare_lsr_data.ipynb`: LSR velocity preprocessing.
- `03_single_kick_comparison.ipynb`: single-kick posterior summary and fitting entry point.
- `04_lsr_distance_prior_sensitivity.ipynb`: Lorimer Model C velocities and recorded population-fit results.

To use Jupyter:

```bash
python -m pip install -r requirements-notebooks.txt
jupyter notebook notebooks/
```

The notebooks read the provided results by default. Their `RUN_FIGURES`, `RUN_PREPROCESS` and `RUN_FULL_FIT` switches control optional computations.

## Running the analysis

```bash
python scripts/prepare_observations.py
python scripts/analyze_birth_period_posteriors.py
python src/mcmc/run_LSR_47_M_8.py --execute-fit --ncores 8 --steps 2000
python src/evidence/run_LSR_Evidence_BF_47_parallel.py both --execute-fit --ncores 8
python src/evidence/run_LSR_Evidence_BF_47_parallel.py summary
python src/single_kick/run_single_kick.py --execute-fit --ncores 8
```

Linux is recommended for the parallel fitting scripts. Full fits can take hours or longer, depending on the model and hardware. Each MCMC process runs one chain, so `--ncores` sets both the chain count and parallelism. Model scripts retain their individual default chain counts; `config.json` records the settings associated with the supplied results.

The evidence workflows use `N_sim=20000`, 1000 live points for the full model, 800 live points for the null model and `dlogz=0.05`. Numerical settings and random seeds are specified in the scripts and `config.json`.

Set `PULSAR_RUN_DIR` to choose a different output directory. The supplied result files remain in `results/`.

## LSR distance-prior sensitivity

See `docs/LSR_SENSITIVITY.md` for the Lorimer Model C calculation. For the 77-source Lognormal model, the effective-radius median is 9.2772 km with this distance prior and 9.3266 km in the main calculation, a change of about 0.53%. The supplemental input samples, run output and plots are included.

```bash
python scripts/verify_lsr_sensitivity.py
python scripts/summarize_lsr_sensitivity.py
```

## Citation and licensing

Citation metadata are provided in `CITATION.cff`.

Repository: https://github.com/Astro-ZL/Pulsar_Stochastic_Kick_Radius

Code is licensed under MIT (`LICENSE`). Author-generated posterior samples, derived tables, figures and documentation are licensed under CC BY 4.0 (`LICENSE-DATA`). Observational source references and third-party terms are described in `THIRD_PARTY_DATA.md`.
