import numpy as np
import random
import pandas as pd
from simcat.agents import Agent
from simcat.utils import compute_thresholds
from simcat.interaction import Interaction
from multiprocessing import Pool

# Define date
date = '21_08_20'

# Define key vars
models = ['animal_game/models/wiki_euclidean_distance.tsv']
animals = pd.read_csv('animal_game/models/animal_list.csv')
thresholds = compute_thresholds(models, 
                                q=[round(n,2) for n in np.arange(0.05, 1.0, 0.05)], 
                                round_at=5)

# Run params
nr_sim = 2
outpath = f'animal_game/logs/test_interactions/pairs'

# Create pairs
pair_df = pd.read_csv(f'animal_game/models/{date}/sampled_pairs.tsv', sep='\t')
fnames_1 = [f'animal_game/models/{date}/noised_distance_matrices/' + f 
            for f in pair_df.fname_1.tolist()]
fnames_2 = [f'animal_game/models/{date}/noised_distance_matrices/' + f 
            for f in pair_df.fname_2.tolist()]
afiles_list = list(zip(fnames_1, fnames_2))
anames_list = [(af[0].split('/')[-1].strip('.tsv'), 
                af[1].split('/')[-1].strip('.tsv')) for af in afiles_list]

# Create agents
agents = []
for i in range(len(afiles_list)):
    a0 = Agent(agent_name=anames_list[i][0], 
               matrix_filename=afiles_list[i][0])
    a1 = Agent(agent_name=anames_list[i][1], 
               matrix_filename=afiles_list[i][1])
    agents.append((a0,a1))
print(f'{len(agents)} pairs found')

# Main loop
def run_pair(a, opath):
    print(f'Agent Names: {a[0].name}, {a[1].name}')
    log_id = f'{a[0].name}_{a[1].name}'
    i = Interaction(agents=a,
                    threshold=thresholds[0.15],
                    save_folder=opath,
                    log_id=log_id,
                    nr_sim=nr_sim)
    int_log = i.run_interaction(seeds=['cat','dog'], 
                                interaction_type='shortest')
    return int_log

out_log = run_pair(agents[-1], outpath)

# Run
#if __name__=='__main__':
#    pool = Pool(processes=22)
#    pool.starmap(run_pair, zip(agents,
#                               [outpath] * len(agents)))
#    pool.close()
