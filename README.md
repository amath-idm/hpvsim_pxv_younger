# Prophylactic HPV vaccination for infants

This repository includes the code for analysing the impact of moving the HPV prophylactic vaccine to the infant series.

## Organization

The repository is organized as follows:

### Running scripts

#### `run_sim.py`
 - This script does not need to be run directly, although it can be run if you want to see the baseline epidemic profile for Nigeria. The purpose of this script is to create the baseline simulation, which is then used as the basis for constructing the different vaccination scenarios.

 Note: this repository does not contain the scripts to produce the calibrated parameters. Instead, the calibrated parameters are stored in the `results` folder and loaded directly by this file to create the baseline sim.

#### `run_scenarios.py`
- This script sets up all the scenarios that will be created as part of this analysis (see `make_vx_scenarios`). It will generally take <5min to run on HPCs.

#### `plot_fig1.py`
 - This script can be used to reproduce Figure 1.

#### `plot_fig23.py` 
- This script can be used to reproduce Figures 2 and 3.

### Additional utility scripts
- `utils.py` contains additional miscellaneous utilities for numerical calculations and creating plots.


## Installation

If HPVsim is already installed (`pip install hpvsim`), the only other required dependency is ``seaborn``.


## Usage

Run the desired analyses by running one of the scripts described above.


## Further information

Further information on HPVsim is available [here](http://docs.hpvsim.org). If you have further questions or would like technical assistance, please reach out to us at info@hpvsim.org.