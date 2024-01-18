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
