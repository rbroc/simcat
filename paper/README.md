# Paper Materials
This part of the repository includes code used for all three experiments in the paper.
This folder is structured as follows:
- You will find the paper and supplementary materials submitted to CogSci under `cogsci_paper.pdf` and `cogsci_supplementary_materials.pdf`
- In `notebooks`, you can find notebooks for the following tasks:
    - `01-Baseline.ipynb` is the notebook used to estimate the stopping threshold for all simulations;
    - `02-Generate-Agents.ipynb` is used to generate agent populations of increasing diversity pairs;
    - `03-Pair-Level-Results.ipynb` includes all visualizations of pair-level results reported in the paper;
    - `04-Simulation-Level-Results.ipynb` includes all visualizations of simulation-level results reported in the paper
- In `metrics`, you can find the processed outputs of the simulations (i.e., aggregate estimates of performance across pairs, for all conditions), which are used to run all the analyses.
- In `scripts`, you will find scripts used to run and postprocess simulations. More specifically:
    - `run_individual.py` is used to run individual simulations (see `run_simulations.sh`);
    - `run_pairs.py` is used to run pairwise simulations (see `run_simulations.sh`);
    - `postprocess.py` is used to postprocess the outputs of all simulations (see also `utils.py`)
- In `models`, you can find semantic models and distance matrices used for simulations (i.e., the semantic memories of agents). Note that we only share the baseline (pre-noising) semantic space, due to storage requirements of the files. Sharing semantic spaces and vectors for all agents would require ~2.5Gb, which would make cloning this repository extremely inefficient. We share, however, the notebook that we have used to generate agent popualtions, for reproducibility (see `02-Generate-Agents.ipynb`). We are happy to share our models upon request;
- In `logs` you can find outputs of specific simulations. We only share individual logs used in the analyses, for storage and efficiency reasons (the full set of logs occupies >250Gb). We are happy to share the remaining logs upon request, if needed. However, we share scripts used to run the simulations and postprocess the files.
