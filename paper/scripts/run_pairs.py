import numpy as np
import random
from pathlib import Path
import pandas as pd
from simcat import Agent, Interaction, compute_thresholds
from multiprocessing import Pool
import argparse
import warnings
import glob

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--interaction-type",
    type=str,
    default="strict",
    help="strict, flexible or shortest",
)
parser.add_argument("--n-back", type=int, default=0, help="n-back allowed")
parser.add_argument(
    "--threads", type=int, default=1, help="number of parallel processes"
)


# Experiment ID (relevant for logging)
ID = "experiments"

# Define init_seeds and number of simulations
models = ["../models/baseline/wiki_euclidean_distance.tsv"]
animals = pd.read_csv("../models/animal_list.csv")
thresholds = compute_thresholds(
    models, q=[round(n, 2) for n in np.arange(0.05, 1.0, 0.05)], round_at=5
)
nr_sim = len(animals["Animals"].tolist())

# Define relevant parameters for all pairs
pair_df = pd.read_csv(f"../models/noised_agents/sampled_pairs.tsv", sep="\t")
fnames_1 = [
    f"../models/noised_agents/noised_distance_matrices/" + f
    for f in pair_df.fname_1.tolist()
]
fnames_2 = [
    f"../models/noised_agents/noised_distance_matrices/" + f
    for f in pair_df.fname_2.tolist()
]
afiles_list = list(zip(fnames_1, fnames_2))
anames_list = [
    (af[0].split("/")[-1].strip(".tsv"), af[1].split("/")[-1].strip(".tsv"))
    for af in afiles_list
]
dicts_1 = [
    f'../models/noised_agents/mappings/{f.strip(".tsv")}.json'
    for f in pair_df.fname_1.tolist()
]
vects_1 = [
    f"../models/noised_agents/noised_vectors/{f}" for f in pair_df.fname_1.tolist()
]
dicts_2 = [
    f'../models/noised_agents/mappings/{f.strip(".tsv")}.json'
    for f in pair_df.fname_2.tolist()
]
vects_2 = [
    f"../models/noised_agents/noised_vectors/{f}" for f in pair_df.fname_2.tolist()
]
dicts_list = list(zip(dicts_1, dicts_2))
vects_list = list(zip(vects_1, vects_2))

# Instantiate pairs
agents = []
for i in range(len(afiles_list)):
    a0 = Agent(
        agent_name=anames_list[i][0],
        matrix_filename=afiles_list[i][0],
        dict_filename=dicts_list[i][0],
        vector_filename=vects_list[i][0],
    )
    a1 = Agent(
        agent_name=anames_list[i][1],
        matrix_filename=afiles_list[i][1],
        dict_filename=dicts_list[i][1],
        vector_filename=vects_list[i][1],
    )
    agents.append((a0, a1))
print(f"{len(agents)} pairs found")


# Main loop
def run_pair(a, opath, itype, n_back):
    existing = glob.glob(f"{opath}/*")
    log_id = f"{a[0].name}_{a[1].name}"
    fname = (
        str(Path(opath) / "_".join([log_id, str(len(a)), str(thresholds[0.15])]))
        + ".txt"
    )

    # makes sure the log does not exist or is incomplete, then run simulation
    if fname not in existing or pd.read_csv(fname).iter.max() < 239:
        print(f"Agent Names: {a[0].name}, {a[1].name}")
        log_id = f"{a[0].name}_{a[1].name}"
        i = Interaction(
            agents=a,
            threshold=thresholds[0.15],
            save_folder=opath,
            log_id=log_id,
            nr_sim=nr_sim,
            kvals=[1, 3, 5],
        )
        i.run_interaction(
            seeds=animals["Animals"].tolist(), interaction_type=itype, n_back=n_back
        )


# Run
if __name__ == "__main__":
    args = parser.parse_args()
    pool = Pool(processes=args.threads)
    nback_str = f"{args.n_back}_back"
    outpath = f"../logs/{ID}/{nback_str}/{args.interaction_type}"
    pool.starmap(
        run_pair,
        zip(
            agents,
            [outpath] * len(agents),
            [args.interaction_type] * len(agents),
            [args.n_back] * len(agents),
        ),
    )
    pool.close()
