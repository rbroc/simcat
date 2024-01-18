import numpy as np
import pandas as pd
from .agents import Agent
from .utils import generate_turn_idx
from pathlib import Path
from datetime import datetime


class Interaction:
    """Interaction class
    Args:
        agents (Agent or list): single Agent or list taking part in the
            interaction. If not defined, nr_agents and matrix_filenames
            must be defined, and agents will be created within Interaction
            init call.
        threshold (float): Lowest possible association threshold. If no
            value above it is found, the game will stop.
        nr_sim (int): How many interactions to run
        max_exchanges (int): Max number of exchanges in the game for early
            stopping (optional)
        log_id (str): filename for logfile
        save_folder (str): relative path for logfile
        nr_agents (int): if agents is not defined, this parameter must be set.
            Indicates how many agents must be initialized
        matrix_filenames (str or list): matrix filename from which agents
            will be initialized (if agents is not passed).
            If a list, multiple files can be passed, and must be of
            length equal to nr_agents
        dict_filenames (str or list): dict_filename, specifying mapping from
            strings to positions in vector spaces
        vector_filenames (str or list): path to vector filename for each agent.
        map_locations (bool): defines where, for each seed and response,
            information on which location the seed occupies should be
            calculated. Remember that, in the original implementation
            of the game, agents' diversity is induced by shuffling the position
            of words in agents' semantic space
        kvals (list): k values used to compute knn-based metrics, such as
            average distance from neighbors
        stopping_rule (str): 'distance' or 'strength'. If 'distance', the simulation
            stops when there are no values in the agent matrix that are lower than
            the threshold. If 'strength', the simulation stops when there are no values in
            the agent matrix that are higher than the threshold.
        agent_kwargs: named arguments for Agent initialization
    """

    def __init__(
        self,
        agents=None,
        threshold=0.006,
        nr_sim=1,
        max_exchanges=None,
        log_id=None,
        save_folder=None,
        nr_agents=None,
        matrix_filenames=None,
        map_locations=True,
        dict_filenames=None,
        vector_filenames=None,
        kvals=[5],
        stopping_rule="distance",
        **agent_kwargs,
    ):
        self.nr_agents = nr_agents
        self.stopping_rule = stopping_rule
        if agents is None:
            agents = []
            if nr_agents is None:
                raise ValueError(
                    "Please pass Agents or specify number of "
                    "agents to be initialized via nr_agents"
                )
            # Get agent parameters in the right shape
            matrix_filenames = self._check_agents_parameters(
                matrix_filenames, "matrix_filenames"
            )
            vector_filenames = self._check_agents_parameters(
                vector_filenames, "vector_filenames"
            )
            dict_filenames = self._check_agents_parameters(
                dict_filenames, "dict_filenames"
            )
            # Initialize agents if nr_agents is defined
            for i in range(nr_agents):
                agent_name = "agent" + str(i + 1)
                if dict_filenames:
                    dict_filename = dict_filenames[i]
                else:
                    dict_filename = None
                if vector_filenames:
                    vector_filename = vector_filenames[i]
                else:
                    vector_filename = None
                agent = Agent(
                    agent_name=agent_name,
                    matrix_filename=matrix_filenames[i],
                    vector_filename=vector_filename,
                    dict_filename=dict_filename,
                    **agent_kwargs,
                )
                agents.append(agent)
        self.agents = [agents] if isinstance(agents, Agent) else agents
        if self.stopping_rule not in ["distance", "strength"]:
            raise ValueError('stopping_rule must be "distance" or "strength"')
        for a in self.agents:
            if not isinstance(a, Agent):
                raise ValueError("agents must be a list of Agent types")
            if map_locations:
                if a.position_map is None:
                    raise ValueError(
                        "dict_filenames must be passed " "if map_location is True"
                    )
        # Set additional parameters
        self.agent_names = [a.name for a in self.agents]
        self.threshold = threshold
        self.nr_sim = nr_sim
        self.max_exchanges = max_exchanges or self.agents[0].matrix.data.shape[0]
        self.log_id = log_id or "log_" + datetime.now().strftime("%Y%m%d-%H%M")
        self.save_folder = save_folder
        self.map_locations = map_locations
        self.kvals = kvals

    def _check_agents_parameters(self, par, parname):
        """Checks whether a given agent parameter has the right length
        Args:
            par (list or str): parameter
            parname (str): name of the parameter
        """
        if isinstance(par, list):
            if len(par) != self.nr_agents:
                raise ValueError(
                    f"Length of {parname} should " "match value of nr_agents"
                )
        else:
            par = [par] * self.nr_agents
        return par

    def _run_trial(self, speaker, seed, turn, itr, init_seed, log=None):
        """Run a single trial (one agent)
        Args:
            speaker (Agent): agent performing speaking act
            seed (str): Cue word
            turn (int): turn number
            itr (int): interaction number
            init_seed (str): initial seed word
            log (df): dataframe containing interaction log
        """
        prob_agent, resp = speaker.speak(seed=seed)
        prob = [
            a.listen(seed, resp) if a is not speaker else prob_agent
            for a in self.agents
        ]
        metrics = []
        for a in self.agents:
            metrics += a.get_metrics(resp, self.kvals, self.threshold)
        if self.map_locations is True:
            pos_init_seed = [a.position_map[seed] for a in self.agents]
            pos_response = [a.position_map[resp] for a in self.agents]
        else:
            pos_init_seed = [np.nan] * len(self.agents)
            pos_response = [np.nan] * len(self.agents)
        log = self._append_data(
            log,
            speaker,
            turn,
            itr,
            seed,
            init_seed,
            resp,
            prob,
            pos_init_seed,
            pos_response,
            metrics,
        )
        return log, resp

    def _run_agent_loop(
        self,
        agent,
        log,
        first_seed,
        current_seed,
        n_attempts,
        max_attempts,
        turn,
        itr,
        init_seed,
    ):
        """Dynamic method that runs full loop of n-back
            attempts (relevant if n_back>0) for a given agent
            or agent pair
        Args:
            agent (Agent): agent object
            log (pd.DataFrame): current log
            first_seed (str): latest seed
            current_seed (str): current seed (if reverting to previous trials)
            n_attempts (int): number of trials recalled so far
            max_attempts (int): how many trials it is allowed to go back (n_back)
            turn (int): turn number
            itr (int): iteration number
            init_seed (str): initial seed of the interaction
        """
        if self.stopping_rule == "distance":
            rule = (agent.matrix.data[current_seed] < self.threshold).any()
        else:
            rule = (agent.matrix.data[current_seed] > self.threshold).any()
        if rule:
            log, current_seed = self._run_trial(
                agent, current_seed, turn, itr, init_seed, log
            )
            success = 1
            n_attempts = max_attempts + 1
        else:
            if log.shape[0] > n_attempts:
                n_attempts = n_attempts + 1
                current_seed = log.seed.tolist()[-n_attempts]
            else:
                n_attempts = max_attempts + 1
                current_seed = first_seed
            success = 0
        return log, current_seed, n_attempts, success

    def _append_data(
        self,
        log,
        agent,
        turn,
        itr,
        seed,
        init_seed,
        resp,
        prob,
        pos_init_seed,
        pos_response,
        metrics,
    ):
        """Appends all trial-level data to the log dataframe,
            for a given agent
        Args:
            log (pd.DataFrame): dataframe with interaction data
            agent (Agent): agent object
            turn (int): turn number
            itr (int): iteration number (relevant if running many simulations)
            seed (str): seed word
            init_seed (str): initial seed
            resp (str): response to the seed
            prob (int): distance between seed and response
            pos_init_seed (int): position of the initial seed in the agent's space
            pos_response (int): position of the response in the agent's space
            metrics (list): knn-based metrics for the agent
        """
        turn_data = [agent.name, turn, itr, seed, resp, *prob]
        if self.map_locations is True:
            turn_data += [*pos_init_seed, *pos_response]
        int_data = [
            self.threshold,
            self.nr_sim,
            self.max_exchanges,
            init_seed,
            self.log_id,
            len(self.agents),
        ]
        metadata = pd.Series(turn_data + int_data + metrics)
        metadata.index = log.columns
        log = log.append(pd.Series(metadata), ignore_index=True)
        return log

    def _create_outpath(self):
        """Create path for whole interaction"""
        fname = (
            "_".join([self.log_id, str(len(self.agents)), str(self.threshold)]) + ".txt"
        )
        if self.save_folder:
            as_path = Path(self.save_folder)
        else:
            as_path = Path("logs")
        as_path.mkdir(parents=True, exist_ok=True)
        fpath = str(as_path / fname)
        return fpath

    def _get_next_agent(self, a_idx):
        """Get agent index for next turn.
            Relevant for interactions without strict
            alternation (for those, this is computed
            in advance).
        Args:
            a_idx (int): dynamic index indicating
                which agent should speak next
        """
        if len(self.agents) > a_idx + 1:
            return a_idx + 1
        else:
            return 0

    def run_interaction(
        self, n_back=0, interaction_type="strict", seeds=None, return_logs=False
    ):
        """Run a full interaction between agents
        Args:
            n_back (int): how many seeds back if no association
                is present
            interaction_type (str): one between 'strict',
                'flexible', and 'shortest'. Irrelevant if only
                passing one agent
            seeds (str or list): name(s) of initial seeds
            return_logs (bool): whether to return all logs.
                Returns empty list if this is False.
        """
        if seeds:
            if isinstance(seeds, list):
                if len(seeds) != self.nr_sim:
                    raise ValueError(
                        f"Length of init_seed should " "match value of nr_sim"
                    )
            else:
                seeds = [seeds] * self.nr_sim
        else:
            seeds = np.random.choice(
                a=self.agents[0].matrix.data.index, size=self.nr_sim
            )
        nr_turns = self.max_exchanges
        turn_idx = generate_turn_idx(nr_turns, self.agents)
        fpath = self._create_outpath()
        logs = []
        cols_metrics = []
        for i in range(len(self.agents)):
            cols_metrics += [
                *[f"resp_knnd_{k}_a{i}" for k in self.kvals],
                *[f"resp_knnd_{k}_a{i}_current" for k in self.kvals],
                f"avg_dist_remain_a{i}",
                *[f"avg_knnd_{k}_a{i}" for k in self.kvals],
                *[f"var_knnd_{k}_a{i}" for k in self.kvals],
                f"resp_neighbors_a{i}",
                f"resp_neighbors_a{i}_current",
            ]
        for itr in range(self.nr_sim):
            cols_turn = [
                "agent",
                "turn",
                "iter",
                "seed",
                "response",
                *["prob_a" + str(i) for i, _ in enumerate(self.agents)],
            ]
            if self.map_locations is True:
                cols_turn += [
                    *["pos_init_seed_a" + str(i) for i, _ in enumerate(self.agents)],
                    *["pos_response_a" + str(i) for i, _ in enumerate(self.agents)],
                ]
            cols_int = [
                "threshold",
                "nr_sim",
                "max_exchanges",
                "init_seed",
                "log_id",
                "nr_agents",
            ]
            log = pd.DataFrame(columns=cols_turn + cols_int + cols_metrics)
            init_seed = seeds[itr]
            for agent in self.agents:
                agent._pop_words(init_seed)

            if interaction_type == "strict":
                # run simulation with strict alternation
                for idx in turn_idx:
                    turn, agent = idx
                    if turn == 0:
                        seed = init_seed
                    n_attempts = 0
                    first_seed = seed
                    while n_attempts <= n_back:
                        log, seed, n_attempts, s = self._run_agent_loop(
                            agent,
                            log,
                            first_seed,
                            seed,
                            n_attempts,
                            n_back,
                            turn,
                            itr,
                            init_seed,
                        )
                    else:
                        if s == 0:
                            break
                        else:
                            continue

            elif interaction_type == "flexible":
                # include possibility of partner helping in case of fixation
                for turn in range(nr_turns):
                    if turn == 0:
                        a_idx = 0
                        seed = init_seed
                    else:
                        a_idx = self._get_next_agent(a_idx)
                    agent_counter = 0
                    n_attempts = 0
                    first_seed = seed
                    while agent_counter < len(self.agents):
                        agent = self.agents[a_idx]
                        while n_attempts <= n_back:
                            log, seed, n_attempts, s = self._run_agent_loop(
                                agent,
                                log,
                                first_seed,
                                seed,
                                n_attempts,
                                n_back,
                                turn,
                                itr,
                                init_seed,
                            )
                        else:
                            if s == 1:
                                break
                            else:
                                a_idx = self._get_next_agent(a_idx)
                                agent_counter += 1
                                n_attempts = 0
                                continue
                    else:
                        break

            elif interaction_type == "shortest":
                # agent with shortest distance to closest word is the speaker
                for turn in range(nr_turns):
                    if turn == 0:
                        seed = init_seed
                    n_attempts = 0
                    first_seed = seed
                    while n_attempts <= n_back:
                        a_idx = np.argmin(
                            [np.nanmin(a.matrix.data[seed]) for a in self.agents]
                        )
                        agent = self.agents[a_idx]
                        log, seed, n_attempts, s = self._run_agent_loop(
                            agent,
                            log,
                            first_seed,
                            seed,
                            n_attempts,
                            n_back,
                            turn,
                            itr,
                            init_seed,
                        )
                    else:
                        if s == 0:
                            break
                        else:
                            continue

            else:
                raise ValueError(
                    """interaction_type must be strict,
                                    flexible, or shortest"""
                )
            named_words = [init_seed] + log["response"].tolist()
            for n_agent, a in enumerate(self.agents):
                if a.vectors is not None:
                    vec_var = a.get_vector_var(named_words)
                    log[f"vecvar_a{n_agent}"] = vec_var
            if itr == 0:
                log.to_csv(fpath, index=False)
            else:
                log.to_csv(fpath, mode="a", index=False, header=False)
            for agent in self.agents:
                agent.matrix.data = agent.matrix_backup.data.copy()
            if return_logs:
                logs.append(log)
        print(f"{self.log_id} done!")
        return logs
