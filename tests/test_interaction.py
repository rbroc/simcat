from simcat import Interaction, Agent
import shutil
import os

def test_individual():
    threshold = 0.01179
    l_id = 'test_individual'
    seeds = ['dog','cat']
    agent = Agent(agent_name='beta', 
                  matrix_filename='data/matrix.tsv',
                  dict_filename='data/mapping.json',
                  vector_filename='data/vectors.tsv')
    i = Interaction(agents=agent, 
                    threshold=0.01179,
                    nr_sim=len(seeds),
                    max_exchanges=3,
                    kvals=[1,3],
                    log_id=l_id)
    outs = i.run_interaction(n_back=0, 
                             seeds=seeds,
                             return_logs=True)
    assert len(outs) == len(seeds)
    assert all([o.shape[0]==3 for o in outs])
    assert all([o.shape[1]==26 for o in outs])
    assert all([o['turn'].max()==len(seeds) for o in outs])
    assert outs[0].init_seed.iloc[0] == seeds[0]
    assert outs[1].init_seed.iloc[0] == seeds[1]
    assert all([x==y 
                for x,y in zip(outs[0]['seed'].tolist()[1:],
                               outs[0]['response'].shift(1).tolist()[1:])])
    assert outs[0].log_id.iloc[0] == l_id
    assert outs[0].threshold.iloc[0] == threshold
    assert outs[0].nr_agents.iloc[0] == 1
    assert f'{l_id}_1_{threshold}.txt' in os.listdir('logs')
    shutil.rmtree('logs')


def test_two_agents():
    # test nr agents
    # test interaction styles
    pass


def test_no_agent_passed():
    pass
