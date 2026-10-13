# LSR distance-prior sensitivity

This supplementary calculation uses the Lorimer et al. (2006) Model C spatial distribution as a distance prior, combines it with the parallax likelihood and propagates the resulting distances through the same LSR velocity correction.

The prior is proportional to `(R/R0)^B exp[-C(R-R0)/R0] exp(-|z|/z0) d^2`, with `B=1.9`, `C=5.0` and `z0=0.33 kpc`. The adopted Galactic frame has `R0=8.122 kpc` and `z_sun=20.8 pc`. The 77 source identities and periods are the same as in the main analysis. Each source has 10,000 distance and velocity samples. For fitting, these are represented by 100 equally spaced midpoint quantiles.

The Bayesian population fit uses the Lognormal kick-velocity model with marginalized neutron-star mass and moment of inertia. Its population simulator, prior, likelihood and chain algorithm are the same as those of the corresponding main calculation. The recorded fit used 300 chains of 2,000 steps, discarded the first 30% of each chain and had a mean acceptance rate of 0.28.

| Calculation | R_eff median (km) |
|---|---:|
| Main parallax-sampling LSR calculation | 9.3266 |
| Lorimer Model C distance-prior calculation | 9.2772 |

The median difference is -0.0494 km (-0.53%). This supports stability of the effective kick-radius scale under this distance-prior choice in the 77-source Lognormal comparison. The source-level velocities can change appreciably for individual objects; `data/lorimer2006/` contains their distances, velocities and uncertainty samples.

## Files

- `data/lorimer2006/`: three 77×10,000 sample matrices, three 77×100 quantile clouds, medians, asymmetric errors and source summaries. Full matrix filenames use `77x10000` to identify their actual shapes.
- `scripts/lorimer_distance_velocity.py`: distance posterior and LSR velocity functions.
- `scripts/prepare_lorimer_data.py`: catalog-to-distance/velocity workflow.
- `src/lsr_sensitivity/run_lorimer2006_77_LN_11.py`: population-fitting script.
- `results/lsr_sensitivity/`: recorded parameter medians, run output, corner plot and predictive plot.
- `notebooks/04_lsr_distance_prior_sensitivity.ipynb`: reading and execution entry point, with a standalone HTML copy.

The supplied Lorimer fit is represented by its recorded medians, execution output and plots. A new fit writes complete posterior samples in pickle and NPZ format to `reproduced/lsr_sensitivity/lorimer2006/`. The median comparison applies to this model and sample; it is not an additional full/null evidence calculation.

## Commands

```bash
python scripts/verify_lsr_sensitivity.py
python scripts/summarize_lsr_sensitivity.py
python scripts/prepare_lorimer_data.py --execute-preprocess
python src/lsr_sensitivity/run_lorimer2006_77_LN_11.py --execute-fit --ncores 300 --steps 2000
```

The data and fit commands are optional and may take substantial time. The reading notebook uses the supplied files by default.

Reference: Lorimer et al. (2006), Model C spatial distribution. The specific prior parameters and Galactic geometry used here are defined in `scripts/lorimer_distance_velocity.py`.
