# Data and Bayesian population-synthesis codes for constraining the effective kick radius in a stochastic pulsar natal-kick model.
````markdown
# Pulsar Stochastic Natal-kick Radius

This repository contains the data table and Jupyter notebooks used for the Bayesian population-synthesis analysis in the paper:

**Joint Bayesian Constraints on the Effective Kick Radius in a Stochastic Natal-kick Model**

The goal of this project is to constrain the effective kick radius in a stochastic pulsar natal-kick framework. In this model, the intrinsic natal velocity and birth spin of a pulsar arise from the cumulative effect of multiple independent impulses, rather than from a single off-center kick. The analysis is performed in the observable space of pulsar spin period and transverse velocity.

## Repository contents

```text
.
├── data/
│   └── parallaxes_1.xlsx
├── notebooks/
│   ├── sample_selection.ipynb
│   ├── bayesian_analysis_young_47.ipynb
│   ├── bayesian_analysis_full_77.ipynb
│   └── bayesian_analysis_single_kick.ipynb
└── README.md
````

### Data table

The `data/` directory contains the curated pulsar sample used in this work. The table includes pulsar spin periods, parallax-based distance information, proper motions, transverse velocities, and sample labels used for the Bayesian analysis.

### Sample-selection notebook

`sample_selection.ipynb` applies the selection criteria used to construct the final isolated-pulsar samples. These include the parallax-quality cut, removal of binary systems and globular-cluster pulsars, and the definition of the full and young pulsar samples.

### Bayesian analysis notebooks

The Bayesian inference is carried out with separate notebooks for different samples and model assumptions:

* `bayesian_analysis_young_47.ipynb`: Bayesian population-synthesis analysis for the young pulsar sample.
* `bayesian_analysis_full_77.ipynb`: Bayesian population-synthesis analysis for the full isolated-pulsar sample.
* `bayesian_analysis_single_kick.ipynb`: Bayesian analysis under the single-kick framework, used for comparison with the stochastic-kick model.

## Method overview

The analysis follows a Bayesian population-synthesis approach. Synthetic pulsar populations are generated under different natal-velocity prescriptions and neutron-star parameter assumptions. The simulated populations are compared with the observed pulsar distribution in the joint period--transverse velocity space.

To preserve correlations between pulsar spin period and transverse velocity, the likelihood is constructed using multivariate kernel density estimation. The main inferred quantity is the effective kick radius, which is interpreted as a population-level root-mean-square angular-momentum lever arm of stochastic natal kicks.

## Samples

Two isolated-pulsar samples are analyzed:

* **Full sample:** 77 isolated pulsars with high-quality parallax-based distance estimates.
* **Young sample:** 47 isolated pulsars with characteristic ages below 10 Myr.

The young sample is used to reduce the impact of long-term spin evolution on the inferred birth-period distribution.

## Requirements

The notebooks were developed in Python and require the following common scientific packages:

```text
numpy
scipy
pandas
matplotlib
seaborn
emcee
corner
scikit-learn
astropy
```

Depending on the local Python environment, additional packages may be required.

## Usage

Clone the repository:

```bash
git clone https://github.com/your-username/pulsar-stochastic-kick-radius.git
cd pulsar-stochastic-kick-radius
```

Open the notebooks with Jupyter:

```bash
jupyter notebook
```

A typical workflow is:

1. Run `sample_selection.ipynb` to reproduce the selected pulsar samples.
2. Run `bayesian_analysis_young_47.ipynb` for the young pulsar sample.
3. Run `bayesian_analysis_full_77.ipynb` for the full isolated-pulsar sample.
4. Run `bayesian_analysis_single_kick.ipynb` for the comparison with the single-kick model.

Because the Bayesian analysis involves Monte Carlo sampling, the exact posterior samples may vary slightly with random seed and computational settings.

## Citation

If you use this repository, please cite the corresponding paper: 

Li Z, Liu X, You Z-Q, Zhu X-J. *Joint Bayesian Constraints on the Effective Kick Radius in a Stochastic Natal-kick Model*. Submitted, 2026.

## Data availability

The data table and analysis notebooks used in this work are provided in this repository. The original pulsar parameters are compiled from public pulsar catalogs and parallax measurements, as described in the paper.

## License

This repository is released for academic and research use. Please cite the corresponding paper if you use the data or code.

````

