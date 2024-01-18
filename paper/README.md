# Paper Materials
This part of the repository includes code used for all three experiments in the paper: ADD_LINK.
This folder is structured as follows:
- In `notebooks`, you can find notebooks for the following tasks:
    - `01-Baseline.ipynb` is the notebook used to estimate the stopping threshold for all simulations;
    - `02-Generate-Agents.ipynb` is used to generate agent populations of increasing diversity pairs;
    - `03-Pair-Level-Results.ipynb` includes all visualizations of pair-level results reported in the paper;
    - `04-Simulation-Level-Results.ipynb` includes all visualizations of simulation-level results reported in the paper
- In `scripts`, you will find scripts used to run and postprocess simulations. More specifically:
    - `run_individual.py` is used to run individual simulations (see `run_simulations.sh`);
    - `run_pairs.py` is used to run pairwise simulations (see `run_simulations.sh`);
    - `postprocess.py` is used to postprocess the outputs of all simulations (see also `utils.py`)
- In `models`, you can find semantic models and distance matrices used for simulations (i.e., the semantic memories of agents);
- In `metrics`, you can find the processed outputs of the simulations (i.e., aggregate estimates of performance across pairs, for all conditions).

Methodological details for all steps can be found in the paper.
