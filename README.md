# Spin and kick populations from arxiv:2107.09693
Constraints on the astrophysical distributions of the black hole spins and
recoil kicks using GWTC-2.

## Papers
This repo is based on the following papers:
* V. Varma, S. Biscoveanu, M. Isi, W. Farr, and S. Vitale.
[arxiv:2107.09693](https://arxiv.org/abs/2107.09693).
* V. Varma, M. Isi, S. Biscoveanu, W. Farr, and S. Vitale.
[arxiv:2107.09692](https://arxiv.org/abs/2107.09692).

If you use these results in your work, please cite these papers.

## Data products
* `data/result_Flatsigma.json`: Spin-model hyperposteriors from
  arxiv:2107.09693, for the Flat-sigma prior.
* `data/result_Jeffreyssigma.json`: Spin-model hyperposteriors from
  arxiv:2107.09693, for the Jeffreys-sigma prior.
* `data/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json`:
PowerLaw+Peak mass-model hyperposteriors from arxiv:2010.14533.

## Scripts and notebooks
* `generate_pop_draws.py`: How to load the above hyperposteriors, evaluate the
  corresponding spin and mass ratio PDFs, and derive the corresponding kick
  PDFs. This can take a few hours to run.
* `example.ipynb`: How to use the above population PDFs to plot 90% credible
  bounds. You can run this directly without running `generate_pop_draws.py`, in
  which case pre-evaluated PDF draws for the spin and kick populations are
  plotted. You can also use these pre-evaluated PDFs directly for your work.

## Dependencies
These are only needed for the `generate_pop_draws.py` script.
All of these can be installed through pip or conda.
* [surfinBH](https://pypi.org/project/surfinBH/)
* [lalsuite](https://pypi.org/project/lalsuite)
* [bilby](https://pypi.org/project/bilby/)
* [gwpopulation](https://pypi.org/project/gwpopulation/)
* [pesummary](https://pypi.org/project/pesummary/)

## Help
The code is maintained by [Vijay Varma](https://vijayvarma.com). Please report
bugs by raising an issue on this
[repo](https://github.com/vijayvarma392/spin_kick_pop_GWTC2/issues).
