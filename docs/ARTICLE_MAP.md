# Manuscript to archive map

The table below maps manuscript figures and tables to their data, results and producing workflows.

| Manuscript item | Canonical file / data | Producing workflow |
|---|---|---|
| Table 1, all 12 model posterior summaries | `results/mcmc/` and `results/summaries/model_parameter_summary.csv` | `src/mcmc/` and `scripts/summarize_saved.py` |
| Figure 1 | `Figures/R_eff_comparison.pdf` | `Figures/plot_radius_comparison.py` |
| Figure 2 | `Figures/corner_8param.pdf` | `Figures/plot_maxwell_corner.py` |
| Figure 3 | `Figures/posterior_predictive.pdf` | `Figures/plot_posterior_predictive.py` |
| Four full/null Bayes factors | `results/evidence/`, `results/summaries/bayes_factor_summary.csv` | four producers in `src/evidence/` |
| Birth-period discussion | `results/summaries/birth_period_posterior_summary.csv` | `scripts/analyze_birth_period_posteriors.py` |
| Appendix A, LSR calculation | `data/` error clouds and summaries | `scripts/lsr_velocity.py`, `scripts/prepare_observations.py` |
| Figure A1 | `Figures/heliocentric_vs_lsr_velocity.pdf` | `Figures/plot_lsr_comparison.py` |
| Table B1, pulsar sample | `data/catalog_selected_77.csv`, `data/catalog_selected_47.csv`, `data/parallaxes_1.xlsx` | catalog selection; see data dictionary |

| Supplementary LSR distance-prior sensitivity | `data/lorimer2006/`, `results/lsr_sensitivity/` | `src/lsr_sensitivity/`, `notebooks/04_lsr_distance_prior_sensitivity.ipynb` |
