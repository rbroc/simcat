from simcat import Agent
import numpy as np
import math


def test_base():
    agent = Agent(agent_name="beta", matrix_filename="data/matrix.tsv")
    assert agent.matrix.data.shape[0] == 240
    assert agent.matrix.data.shape[1] == 240
    assert agent.model is agent.matrix.data
    n_same = (agent.matrix_backup.data.values == agent.matrix.data.values).sum()
    assert n_same == 240 * 239  # except diagonal
    assert agent.name == "beta"


def test_dict_filename():
    agent = Agent(
        agent_name="beta",
        matrix_filename="data/matrix.tsv",
        dict_filename="data/mapping.json",
    )
    assert agent.position_map is not None
    assert len(agent.position_map.keys()) == 240


def test_vec_filename():
    agent = Agent(
        agent_name="beta",
        matrix_filename="data/matrix.tsv",
        dict_filename="data/mapping.json",
        vector_filename="data/vectors.tsv",
    )
    assert agent.vectors.shape[0] == 240
    assert agent.vectors.shape[1] == 400


def test_methods():
    agent = Agent(
        agent_name="beta",
        matrix_filename="data/matrix.tsv",
        dict_filename="data/mapping.json",
        vector_filename="data/vectors.tsv",
    )
    agent._pop_words("dog")
    assert np.isnan(agent.matrix.data.values).sum() == 479
    assert np.isnan(agent.matrix_backup.data.values).sum() == 240
    assert agent._return_prob("cat", "rat") == 0.00982
    assert agent.speak("rat", pop=True) == (0.00912, "mouse")
    assert np.isnan(agent.model.loc["mouse"].values).sum() == 240
    prob, resp = agent.speak("eagle", pop=False)
    assert np.isnan(agent.model.loc[resp].values).sum() == 1
    prob_listen = agent.listen("eagle", resp, pop=False)
    assert prob == prob_listen
    assert np.isnan(agent.model.loc[resp].values).sum() == 1
    agent.listen("eagle", resp, pop=True)
    assert np.isnan(agent.model.loc[resp].values).sum() == 240
    v_1 = agent.vectors.loc[["rat", "cat"]].values.var(axis=0).mean().round(5)
    v_2 = agent.vectors.loc[["rat", "kiwi"]].values.var(axis=0).mean().round(5)
    assert v_1 < v_2


def test_get_metrics():
    agent = Agent(
        agent_name="beta",
        matrix_filename="data/matrix.tsv",
        dict_filename="data/mapping.json",
        vector_filename="data/vectors.tsv",
    )
    outs = agent.get_metrics(
        resp_wd="dog", kvals=[2, 3], t=np.quantile(agent.matrix.data.values, 0.15)
    )
    assert len(outs) == 11
    assert math.isnan(outs[4]) is False
