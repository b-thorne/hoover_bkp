# hoover

Package to calculate the log probability of foregrounds.

## Description

Given a dataset $\mathbf{d}$, this package calculates, and samples from, the maximum 
likelihood set of foreground parameters, $P(\mathbf{\theta}}\mathbf{d})$.

This approach has been used in these papers:

1. [Simulated forecasts for primordial B-mode searches in ground-based experiments](https://arxiv.org/abs/1608.00551)
2. [The Simons Observatory: Science goals and forecasts](https://arxiv.org/abs/1808.07445)
3. [Removal of Galactic foregrounds for the Simons Observatory primordial gravitational wave search](https://arxiv.org/abs/1905.08888)

## Useage

[An example note book](notebooks/example.ipynb) is in the `notebooks` folder.

To add SEDs, edit [the SED functions file](src/hoover/seds.py). The likelihood is defined in [the likelihood file](src/hoover/likelihood.py).

## Note

This project has been set up using PyScaffold 3.2.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
