<a href="https://github.com/rbroc/cosearch"><img src="https://github.com/rbroc/cosearch/raw/master/img/simcat-logo-title.png" width="200" align="right" /></a>

# simcat: a Python package to Simulate Multi-agent Cognitive Association Tasks
![tests](https://github.com/rbroc/simcat/actions/workflows/run-tests.yml/badge.svg?event=push)
[![code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![python version](https://img.shields.io/badge/Python-%3E=3.8-blue)](https://github.com/rbroc/simcat)
[![license: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a Python package to perform simulations of multi-agent cognitive association tasks (e.g., [the verbal fluency test](https://en.wikipedia.org/wiki/Verbal_fluency_test)).

The package makes it eary to instantiate one or more agents, endow them with a semantic memory, defined as a vector space that can be different across agents, and have agents perform association tasks in the form of a *verbal fluency tasks* either individually of following different interaction structures.
At present, the package includes two hard-coded rules:
1. Agents always name the item in their semantic space which has the lowest distance from the prompt (or has the strongest association, depending on options);
2. The simulation stops when all items in the semantic space have been named during a simulation, or when no items in the semantic space can be found, whose distance from the current prompt is lower than (or higher than, depending on options) a set threshold, passed as a parameter to the simulation.
These constraints could potentially be relaxed in future releases.

When multiple agents perform the association task together, they can interact following three possible turn-taking rules:
1. **Strict** turn-taking: agents alternate in naming the next item;
2. **Flexible** (or **collaborative**) turn-taking: agents are allowed to step in if the partner has no available associations from the current prompt;
3. **Shortest** (or **competitive**) turn-taking: the turn-holder is the agent with the shortest distance between the current prompt and the closest item in their semantic memory.
The package is tested for single-agent and two-agent interactions, but it supports interactions with more than two agents.
Furthermore, the package makes it possible to endow individual agents with basic forms of working memory, which make it possible to revert to previous prompts if no association from the current prompt is available.

For each simulation, a log of the entire association chain, including a range of metadata (e.g., the distance in semantic space between the prompt and the response item, or the density of the neighborhood in which named words are located), is produced and stored.

### :cherry_blossom:  Applications
Applications of this framework include simulation of verbal fluency data, as well as computational studies investigating emergent properties of parameters such as **cognitive diversity** between agents and **turn-taking structures**.

A first study using this framework to investigate the effect of cognitive diversity on search behavior and performance in the association task was published in the proceedings of the 2022 Annual Meeting of the Cognitive Science Society (available at: https://escholarship.org/uc/item/58v5d82w):
```
@inproceedings{rocca2022cognitive,
  title={Cognitive diversity promotes collective creativity: an agent-based simulation},
  author={Rocca, Roberta and Tylén, Kristian},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume={44},
  number={44},
  year={2022}
}
```

A follow-up involving comprehensive investigation of the effect of turn-taking structures and working memory flexibility is available as a preprint at: https://osf.io/preprints/psyarxiv/n3t6j, and currently under review.
```
@misc{rocca2024diversity,
  title={The effect of diversity and social interaction on cognitive search: An agent-based simulation},
  url={osf.io/preprints/psyarxiv/n3t6j},
  author={Rocca, Roberta and Tylén, Kristian},
  publishr={PsyArXiv},
  year={2024},
}
```
Materials and documentation concerning this study are available under [`paper`](https://github.com/rbroc/simcat/tree/master/paper). 

### :gear:  Installation
We strongly recommend that you install `simcat` from pip:

`pip install simcat`

You can also install `simcat` from source by running:

`pip install git+https://github.com/rbroc/simcat.git`

or

```
git clone https://github.com/rbroc/simcat.git
cd simcat
pip3 install -e .
```

### :robot: :speech_balloon:  Running a simulation
Once you have installed `simcat` can easily run a simulation with the following command:

```python
from simcat import Interaction, Agent

agent_1 = Agent(agent_name="agent_1", matrix_filename="sample/agent_1/matrix.tsv")
agent_2 = Agent(agent_name="agent_2", matrix_filename="sample/agent_2/matrix.tsv")

i = Interaction(
    agents=[agent_1, agent_2],
    save_folder="./sample/outs",
    log_id="my_interaction",
    nr_sim=2,
    map_locations=False,
)

i.run_interaction(seeds=["cat", "dog"], interaction_type="strict", n_back=0)
```

<div class="table-wrapper" markdown="block">

### :floppy_disk:  Simulation outputs
A csv file containing a log of the interaction will be saved, whose first rows will look like this:
&nbsp;

| agent   | turn | iter | seed        | response    | prob_a0 | prob_a1 | threshold | nr_sim | max_exchanges | init_seed | log_id        | nr_agents | resp_knnd_5_a0 | resp_knnd_5_a0_current | avg_dist_remain_a0 | avg_knnd_5_a0 | var_knnd_5_a0 | resp_neighbors_a0 | resp_neighbors_a0_current | resp_knnd_5_a1 | resp_knnd_5_a1_current | avg_dist_remain_a1 | avg_knnd_5_a1 | var_knnd_5_a1 | resp_neighbors_a1 | resp_neighbors_a1_current |
|---------|------|------|-------------|-------------|---------|---------|-----------|--------|---------------|-----------|---------------|-----------|----------------|------------------------|--------------------|---------------|---------------|-------------------|---------------------------|----------------|------------------------|--------------------|---------------|---------------|-------------------|---------------------------|
| agent_1 | 0    | 0    | cat         | dog         | 0.00782 | 0.0113  | 0.01179   | 2      | 240           | cat       | test_2_agents | 2         | 0.00982        | 0.0099                 | 0.01274            | 0.011         | 0.00071       | 83                | 82                        | 0.01053        | 0.01053                | 0.01273            | 0.01099       | 0.00072       | 70                | 69                        |
| agent_2 | 1    | 0    | dog         | lion        | 0.01136 | 0.00891 | 0.01179   | 2      | 240           | cat       | test_2_agents | 2         | 0.01066        | 0.01066                | 0.01274            | 0.011         | 0.00071       | 50                | 48                        | 0.00989        | 0.01023                | 0.01274            | 0.011         | 0.00072       | 98                | 96                        |
| agent_1 | 2    | 0    | lion        | elephant    | 0.00938 | 0.01264 | 0.01179   | 2      | 240           | cat       | test_2_agents | 2         | 0.00987        | 0.01007                | 0.01274            | 0.01101       | 0.00071       | 81                | 78                        | 0.01148        | 0.01148                | 0.01274            | 0.011         | 0.00072       | 10                | 10                        |
| agent_2 | 3    | 0    | elephant    | coral_snake | 0.01317 | 0.01026 | 0.01179   | 2      | 240           | cat       | test_2_agents | 2         | 0.01019        | 0.01019                | 0.01274            | 0.01101       | 0.00071       | 46                | 46                        | 0.01054        | 0.01076                | 0.01274            | 0.011         | 0.00072       | 44                | 42                        |
| agent_1 | 4    | 0    | coral_snake | sea_snake   | 0.00836 | 0.01368 | 0.01179   | 2      | 240           | cat       | test_2_agents | 2         | 0.01045        | 0.01053                | 0.01274            | 0.01102       | 0.00071       | 66                | 65                        | 0.01056        | 0.01056                | 0.01274            | 0.011         | 0.00072       | 27                | 26                        |
| agent_2 | 5    | 0    | sea_snake   | coyote      | 0.0135  | 0.00919 | 0.01179   | 2      | 240           | cat       | test_2_agents | 2         | 0.01071        | 0.01071                | 0.01274            | 0.01102       | 0.00071       | 69                | 67                        | 0.01027        | 0.01036                | 0.01274            | 0.01101       | 0.00072       | 40                | 38                        |
| agent_1 | 6    | 0    | coyote      | rabbit      | 0.00982 | 0.01201 | 0.01179   | 2      | 240           | cat       | test_2_agents | 2         | 0.00987        | 0.01008                | 0.01275            | 0.01103       | 0.00071       | 91                | 86                        | 0.01175        | 0.01175                | 0.01274            | 0.01101       | 0.00072       | 7                 | 7                         |
| agent_2 | 7    | 0    | rabbit      | squid       | 0.01257 | 0.01078 | 0.01179   | 2      | 240           | cat       | test_2_agents | 2         | 0.00955        | 0.00955                | 0.01275            | 0.01103       | 0.0007        | 54                | 53                        | 0.01083        | 0.01088                | 0.01274            | 0.01101       | 0.00073       | 54                | 50                        |
| agent_1 | 8    | 0    | squid       | cuttlefish  | 0.00845 | 0.01253 | 0.01179   | 2      | 240           | cat       | test_2_agents | 2         | 0.01011        | 0.01028                | 0.01275            | 0.01104       | 0.0007        | 53                | 50                        | 0.0108         | 0.0108                 | 0.01274            | 0.01101       | 0.00073       | 54                | 52                        |
| agent_2 | 9    | 0    | cuttlefish  | gull        | 0.01311 | 0.01027 | 0.01179   | 2      | 240           | cat       | test_2_agents | 2         | 0.01164        | 0.01166                | 0.01275            | 0.01104       | 0.0007        | 10                | 9                         | 0.00982        | 0.00982                | 0.01274            | 0.01102       | 0.00072       | 83                | 78                        |

</div>

The columns in the output refer to the following:
| **name** | **description** |
|------|-------------|
| `agent` | name of agent speaking |
| `turn` | turn number within the simulation |
| `iter` | simulation number |
| `seed` | seed at current turn |
| `response` | response to seed |
| `prob_a{n}` | Euclidean distance between seed and response in the n-th agent's space (Python indexing, therefore the index for the first agent is 0) |
| `threshold` | stopping threshold for the simulation |
| `nr_sim` | total number of simulations to run |
| `max_exchanges` | maximum number of turns within a simulation |
| `init_seed` | first seed in the simulation |
| `log_id` | id of the simulation, used in the logfile |
| `nr_agents` | number of agents |
| `resp_knnd_{k}_a{n}` | distance between the response and its k-th nearest neighbor (relative to the original semantic memory of the n-th agent) -- _k_ is passed as a parameter to the `Interaction` call. This is an index of neighborhood density. |
| `resp_knnd_{k}_a{n}_current` | same as above, but computed on the updated space (i.e., accounting only for items that have _not_ been named) |
| `avg_dist_remain_a{n}` | average Euclidean distance between each pair of items that have not _yet_ been named (relative to the semantic memory of the n-th agent). |
| `avg_knnd_{k}_a{n}` | Mean distance from k-th neighbor, for each item in the n-th agent's semantic space. Only items that have _not_ been named are considered. |
| `var_knnd_{k}_a{n}` | Std of the distance from k-th neighbor, for each item in the n-th agent's semantic space. Only items that have _not_ been named are considered. |
| `resp_neighbors_a{n}` | number of sub-threshold neighbors of the response item (relative to the semantic memory of the n-th agent) |
| `resp_neighbors_a{n}_current` | number of _remaining_ sub-threshold neighbors of the response item (relative to the semantic memory of the n-th agent). Items are dropped from agents' semantic space as they are named. |

### :hammer_and_wrench:  Maintenance and development
Please feel free to contribute to this project. You can contribute to bug fixes :bug: or implement new functionality :seedling: by [opening a PR](https://github.com/rbroc/simcat/pulls) or [opening an issue](https://github.com/rbroc/simcat/issues).
You are also very welcome to contribute `docs`, which are not yet available.

### :file_folder:  Project structure
```
simcat
├── LICENSE
├── pyproject.toml
├── README.md
├── img
├── paper      
│   ├── README.md            
│   ├── figures
│   ├── logs
│   ├── metrics
│   ├── models 
│   ├── notebooks 
│   └── scripts
├── sample                  
│   ├── agent_1
│   ├── agent_2
│   └── outs
├── src
│   └── simcat
└── tests
```

### :book:  Citation

If you use this work, please cite:
```
@misc{rocca2024diversity,
  title={The effect of diversity and social interaction on cognitive search: An agent-based simulation},
  url={osf.io/preprints/psyarxiv/n3t6j},
  author={Rocca, Roberta and Tylén, Kristian},
  publishr={PsyArXiv},
  year={2024},
}
```
