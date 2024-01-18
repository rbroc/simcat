import glob
from simcat.agents import Agent
import itertools
import pandas as pd
import numpy as np
from simcat import compute_thresholds
from simcat import Interaction
from multiprocessing import Pool
import argparse
import warnings

warnings.filterwarnings("ignore")

# ID of experiment (for logging mainly)
ID = "experiments"

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--n-back", type=int, default=0, help="n-back allowed")
parser.add_argument(
    "--threads", type=int, default=1, help="number of parallel processes"
)

# Compute thresholds, init_seeds, and get models
animals = pd.read_csv("../models/animal_list.csv")
models = ["../models/baseline/wiki_euclidean_distance.tsv"]  # baseline
thresholds = compute_thresholds(
    models, q=[round(n, 2) for n in np.arange(0.05, 1.0, 0.05)], round_at=5
)
matrices = glob.glob(f"../models/noised_agents/noised_distance_matrices/*")
map_id = f'{f.split("/")[-1].strip(".tsv")}.json'
dicts = [f"../models/noised_agents/mappings/{map_id}" for f in matrices]
vects = [f'../models/noised_agents/noised_vectors/{f.split("/")[-1]}' for f in matrices]

# Initialize the agents
agents = []
for i, m in enumerate(matrices):
    print(f"initializing agent {i}")
    agent = Agent(
        agent_name=m.split("/")[-1][:-4],
        matrix_filename=m,
        vector_filename=vects[i],
        dict_filename=dicts[i],
    )
    agents.append(agent)

# Set number of simulations
nr_sim = len(animals["Animals"].tolist())


# Main function running individual simulations
def run_individual(agent, outpath, n_back):
    print(f"Agent Name: {agent.name}")
    log_id = f"{agent.name}"
    i = Interaction(
        agents=agent,
        threshold=thresholds[0.15],
        save_folder=outpath,
        log_id=log_id,
        nr_sim=nr_sim,
        kvals=[1, 3, 5],
    )
    i.run_interaction(seeds=animals["Animals"].tolist(), n_back=n_back)


if __name__ == "__main__":
    args = parser.parse_args()
    pool = Pool(processes=args.threads)
    nback_str = f"{args.n_back}_back"
    outpath = f"../logs/{ID}/{nback_str}/individual"
    outpaths = [outpath] * len(agents)
    pool.starmap(run_individual, zip(agents, outpaths, [args.n_back] * len(agents)))
    pool.close()
