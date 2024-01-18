from simcat import Interaction, Agent
import shutil
import pandas as pd
import numpy as np
import os


THRESHOLD = 0.01179
SEEDS = ["dog", "cat"]


def _check_order_strict(o, i):
    check_nr = len(i.agents) + 1 if o.shape[0] >= len(i.agents) + 1 else o.shape[0]
    for idx in range(check_nr):
        seed = o["seed"].iloc[idx]
        resp = o["response"].iloc[idx]
        probs = [o[f"prob_a{i}"].iloc[idx] for i in range(len(i.agents))]
        for ix, p in enumerate(probs):
            assert p == i.agents[ix].matrix_backup.data.loc[seed][resp]


def test_individual():
    l_id = "test_individual"
    agent = Agent(
        agent_name="beta",
        matrix_filename="data/matrix.tsv",
        dict_filename="data/mapping.json",
        vector_filename="data/vectors.tsv",
    )
    i = Interaction(
        agents=agent,
        threshold=THRESHOLD,
        nr_sim=len(SEEDS),
        max_exchanges=3,
        kvals=[1, 3],
        log_id=l_id,
    )
    outs = i.run_interaction(n_back=0, seeds=SEEDS, return_logs=True)
    assert len(outs) == len(SEEDS)
    assert all([o.shape[0] == 3 for o in outs])
    assert all([o.shape[1] == 26 for o in outs])
    assert all([o["turn"].max() == len(SEEDS) for o in outs])
    for o in outs:
        assert all([v < THRESHOLD for v in o["prob_a0"].tolist()])
    assert outs[0].init_seed.iloc[0] == SEEDS[0]
    assert outs[1].init_seed.iloc[0] == SEEDS[1]
    assert all(
        [
            x == y
            for x, y in zip(
                outs[0]["seed"].tolist()[1:], outs[0]["response"].shift(1).tolist()[1:]
            )
        ]
    )
    assert outs[0].log_id.iloc[0] == l_id
    assert outs[0].threshold.iloc[0] == THRESHOLD
    assert outs[0].nr_agents.iloc[0] == 1
    assert f"{l_id}_1_{THRESHOLD}.txt" in os.listdir("logs")
    i_noname = Interaction(
        nr_agents=1,
        matrix_filenames="data/matrix.tsv",
        dict_filenames="data/mapping.json",
        vector_filenames="data/vectors.tsv",
        threshold=THRESHOLD,
        nr_sim=len(SEEDS),
        max_exchanges=3,
        kvals=[1, 3],
        log_id=l_id,
    )
    assert len(i_noname.agents) == 1
    assert i_noname.agents[0].name == "agent1"
    assert len(i_noname.agents) == 1
    a_data = i_noname.agents[0].matrix.data.values
    assert (a_data == agent.matrix_backup.data.values).sum() == 240 * 239
    shutil.rmtree("logs")


def test_two_agents():
    a_1 = Agent(agent_name="a1", matrix_filename="data/matrix.tsv")
    a_2 = Agent(agent_name="a2", matrix_filename="data/matrix_2.tsv")
    i = Interaction(
        agents=[a_1, a_2],
        threshold=THRESHOLD,
        kvals=[1, 3],
        nr_sim=240,
        max_exchanges=3,
        map_locations=False,
    )
    outs = i.run_interaction(n_back=0, return_logs=True)
    assert len(outs) == 240
    for o in outs:
        if o.shape[0] > 0:
            assert all(
                [
                    x != y
                    for x, y in zip(
                        o["agent"].tolist()[1:], o["agent"].shift(1).tolist()[1:]
                    )
                ]
            )
            assert all([p in o.columns for p in ["prob_a0", "prob_a1"]])
            assert o["turn"].max() < 3
            _check_order_strict(o, i)
    shutil.rmtree("logs")


def test_interaction_styles():
    a_1 = Agent(agent_name="a1", matrix_filename="data/matrix.tsv")
    a_2 = Agent(agent_name="a2", matrix_filename="data/matrix_2.tsv")
    i = Interaction(
        agents=[a_1, a_2],
        threshold=THRESHOLD,
        kvals=[1, 3],
        nr_sim=len(SEEDS),
        map_locations=False,
    )
    out_intstyles = {}
    for istyle in ["strict", "flexible", "shortest"]:
        out_concat = pd.concat(
            i.run_interaction(interaction_type=istyle, return_logs=True, seeds=SEEDS)
        )
        out_intstyles[istyle] = out_concat
    for s in SEEDS:
        m = np.min([v[v["init_seed"] == s].shape[0] for v in out_intstyles.values()])
        probs = {}
        for k, v in out_intstyles.items():
            probs[k] = [
                r["prob_a0"] if r["agent"] == "a1" else r["prob_a1"]
                for i, r in v[v["init_seed"] == s].iterrows()
            ]
        assert len(probs["flexible"]) > len(probs["strict"])
        assert np.mean(probs["shortest"][:m]) < np.mean(probs["strict"][:m])
        assert np.mean(probs["shortest"][:m]) < np.mean(probs["flexible"][:m])
    shutil.rmtree("logs")


def test_three_agents():
    a_1 = Agent(agent_name="a1", matrix_filename="data/matrix.tsv")
    a_2 = Agent(agent_name="a2", matrix_filename="data/matrix_2.tsv")
    a_3 = Agent(agent_name="a3", matrix_filename="data/matrix.tsv")
    i = Interaction(
        agents=[a_1, a_2, a_3],
        threshold=THRESHOLD,
        kvals=[1, 3],
        nr_sim=len(SEEDS),
        map_locations=False,
    )
    outs = i.run_interaction(seeds=SEEDS, return_logs=True)
    for o in outs:
        _check_order_strict(o, i)
    shutil.rmtree("logs")
