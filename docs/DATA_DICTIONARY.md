# Data definitions and ordering

The source workbook has 100 rows and 14 columns. `Dist`, `Dist_eplus`, and `Dist_eminus` are **parallax and parallax error widths in mas**, despite their names. `Period` is seconds; `Pmtot(mas/yr)`, `pmra`, `pmdec` are mas/year; `gl`, `gb` are Galactic degrees; `Age(10**7)` is characteristic age in units of 10 million years. References and observing techniques are preserved.

The sample selection is `Dist / Dist_eminus > 3`, exclude J1809-1943, and `Period > 0.1`. This gives 77 rows. Applying `Age(10**7) < 1` gives 47 young rows. The Binary column is included in the catalog.

`catalog_selected_77.csv` and `catalog_selected_47.csv` explicitly record identity and order. Each contains the original catalog fields plus `source_index_77`, period seconds, characteristic age years and subset flag.

| Files | Shape | Units / definition |
|---|---|---|
| `period_1_new_77.txt`, `period_77_data.txt` | 77 | period in seconds; equal numeric values, different text formatting |
| `period_2_new_47.txt` | 47 | young-sample periods in the matching order |
| `vpec_lsr_median_N.txt` | N | LSR-corrected transverse peculiar speed, km/s |
| `vpec_lsr_err_low_N.txt`, `vpec_lsr_err_up_N.txt` | N | positive lower/upper widths from percentiles 16/50/84, km/s |
| `vpec_lsr_samples_matrix_Nx100.txt` | N × 100 | row-aligned 100-draw velocity cloud, km/s |
| `vpec_lsr_summary_77.csv` | 77 rows | source identity and velocity percentiles |
| `period_87_data.txt`, `vtrans_87_data.txt`, `v_err_low_87_data.txt`, `v_err_up_87_data.txt` | 87 | heliocentric arrays used by Appendix plotting and loaded by original producers |

The population inference uses the 47/77 LSR arrays. The 87-source heliocentric arrays provide the input for the Appendix comparison. Source ordering is checked against the 77-source period array.

The LSR producer uses Astropy frame v4.0, R0=8.122 kpc, z_sun=20.8 pc and total Solar velocity (12.9,245.6,7.78) km/s, with an adopted Solar peculiar y velocity of 12.24 km/s and a flat circular speed 233.36 km/s. Only parallax uncertainties are sampled; proper motions are held fixed. The seed is 20260610 plus source index. Nonpositive parallax draws are rejected. The adopted Solar position is (-R0,0,z_sun).

MCMC pickles contain `columns`, `samples`, `model_name`; dimensions and units are explicit in `provenance/posterior_schema.json`. Dynesty equal-weight pickles contain `columns`, `samples_equal`, `logz`, `logzerr`, `nlive`, `n_sim`, `dlogz`. Native dynesty results are retained as well. Pickles are Python-oriented; neutral NPZ samples and column schemas are also provided, generated from these same frozen arrays.

## Lorimer Model C supplement

`data/lorimer2006/` uses the same 77-source row order as `catalog_selected_77.csv`. Distance files are in kpc; ordinary heliocentric and LSR-corrected velocity files are in km/s. Full matrices have shape 77×10,000. Quantile clouds have shape 77×100, obtained at probabilities `(j+0.5)/100`, j=0,...,99. The main and supplemental input arrays are stored separately.
