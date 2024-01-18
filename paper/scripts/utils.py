import pandas as pd
import numpy as np
import glob
import warnings

warnings.filterwarnings("ignore")

animals = pd.read_csv(f"../models/animal_list.csv")
animals_idx_dict = dict(
    zip(animals["Animals"].tolist(), [str(i) for i in range(len(animals))])
)

# Dictionaries define how turn-level values should be aggregated (trial-level)
INDIVIDUAL_DICT = {
    "turn": "max",
    "threshold": "first",
    "agent": "first",
    "init_seed": "first",
    "prob_a0": np.nanmean,
    "response": "last",
}
PAIR_DICT = {
    "turn": "max",
    "log_id": "first",
    "agent_0": "first",
    "agent_1": "first",
    "threshold": "first",
    "init_seed": "first",
    "jump_a0": np.nanmean,
    "jump_a1": np.nanmean,
    "jump_speaker": np.nanmean,
    "jump_listener": np.nanmean,
    "response": "last",
    # Vector variance
    "vecvar_a0": "last",
    "vecvar_a1": "last",
    # Nr neighbors (measure of density)
    "resp_neighbors_a0": np.nanmean,
    "resp_neighbors_a1": np.nanmean,
    "resp_neighbors_diff": np.nanmean,  # |a0-a1|
    "resp_neighbors_speaker": np.nanmean,
    "resp_neighbors_listener": np.nanmean,
    "resp_neighbors_a0_current": np.nanmean,
    "resp_neighbors_a1_current": np.nanmean,
    "resp_neighbors_speaker_current": np.nanmean,
    "resp_neighbors_listener_current": np.nanmean,
    # Distance from kth neighbor (measure of density)
    "resp_knnd_5_a0": lambda x: np.nanmean(x),
    "resp_knnd_5_a1": lambda x: np.nanmean(x),
    "resp_knnd_5_diff": np.nanmean,
    "resp_knnd_5_speaker": lambda x: np.nanmean(x),
    "resp_knnd_5_listener": lambda x: np.nanmean(x),
    # global speed of neighborhood depletion
    "avg_knnd_5_a0": lambda x: x[:-1].diff(periods=5).mean(),
    "avg_knnd_5_a1": lambda x: x[:-1].diff(periods=5).mean(),
    # variance in speed of neighborhood depletion
    "var_knnd_5_a0": lambda x: x[:-1].diff(periods=5).mean(),
    "var_knnd_5_a1": lambda x: x[:-1].diff(periods=5).mean(),
    # speed of local neighborhood depletion
    "resp_depletion_a0": lambda x: x[:-1].diff(periods=5).mean(),
    "resp_depletion_a1": lambda x: x[:-1].diff(periods=5).mean(),
    # difference in depletion of a neighborhood across two agents
    "resp_depletion_diff": np.nanmean,
    # speed of neighborhood depletion
    "resp_depletion_speaker": lambda x: x[:-1].diff(periods=5).mean(),
    "resp_depletion_listener": lambda x: x[:-1].diff(periods=5).mean(),
}

# Define new names for outputs of trial->simulation-level aggregation
INDIVIDUAL_NAMES = [
    "iter",
    "performance",
    "threshold",
    "agent_name",
    "init_seed",
    "flexibility",
    "last_response",
]
PAIR_NAMES = [
    "iter",
    "performance",
    "pair",
    "agent_0",
    "agent_1",
    "threshold",
    "init_seed",
    "flexibility_a0_pair",
    "flexibility_a1_pair",
    "flexibility_speaker",
    "flexibility_listener",
    "last_response",
]
METRIC_NAMES = [
    "vecvar_a0",
    "vecvar_a1",
    "resp_neighbors_a0",
    "resp_neighbors_a1",
    "resp_neighbors_diff",
    "resp_neighbors_speaker",
    "resp_neighbors_listener",
    "resp_neighbors_a0_current",
    "resp_neighbors_a1_current",
    "resp_neighbors_speaker_current",
    "resp_neighbors_listener_current",
    "resp_knnd_5_a0",
    "resp_knnd_5_a1",
    "resp_knnd_5_diff",
    "resp_knnd_5_speaker",
    "resp_knnd_5_listener",
    "avg_knnd_5_a0_slope",
    "avg_knnd_5_a1_slope",
    "var_knnd_5_a0_slope",
    "var_knnd_5_a1_slope",
    "resp_depletion_a0_slope",
    "resp_depletion_a1_slope",
    "resp_depletion_diff",
    "resp_depletion_speaker_slope",
    "resp_depletion_listener_slope",
]
PAIR_NAMES = PAIR_NAMES + METRIC_NAMES

# Dicts to rename columns that have the same name across individual and pair
# Needed when merging
RENAME_0 = {
    "performance_x": "performance_pair",
    "performance_y": "performance_a0",
    "noise_level_y": "noise_level_a0",
    "flexibility": "flexibility_a0",
    "last_response_x": "last_response_pair",
    "last_response_y": "last_response_a0",
}
RENAME_1 = {
    "performance": "performance_a1",
    "flexibility": "flexibility_a1",
    "last_response": "last_response_a1",
    "noise_level": "noise_level_a1",
}

# Define dictionaries for aggregation trial -> pair-level
PAIR_LEVEL_DICT = {
    "diversity_level": np.nanmean,
    "fluency_a0": np.nanmean,
    "fluency_a1": np.nanmean,
    "fluency_pair": np.nanmean,
    "flexibility_speaker": np.nanmean,
    "flexibility_listener": np.nanmean,
    "flexibility_a0_pair": np.nanmean,
    "flexibility_a1_pair": np.nanmean,
    "flexibility_a0": np.nanmean,
    "flexibility_a1": np.nanmean,
    "orig_a0": np.nanmean,
    "orig_a1": np.nanmean,
    "orig_speaker": np.nanmean,
    "orig_listener": np.nanmean,
    "orig_a0_individual": np.nanmean,
    "orig_a1_individual": np.nanmean,
    "collective_inhibition": np.nanmean,
}
# Names for correlation variables
COR_NAMES = ["resp_neighbors_cor", "resp_neighbors_current_cor", "jump_cor"]
PAIR_LEVEL_DICT.update(
    dict(zip(METRIC_NAMES + COR_NAMES, [np.nanmean] * len(METRIC_NAMES + COR_NAMES)))
)

# Define names for outputs of simulation -> pair/ind-level aggregation
PAIR_LEVEL_NAMES = (
    [
        "pair",
        "diversity_level",
        "fluency_a0",
        "fluency_a1",
        "fluency_pair",
        "flexibility_speaker",
        "flexibility_listener",
        "flexibility_a0_pair",
        "flexibility_a1_pair",
        "flexibility_a0",
        "flexibility_a1",
        "orig_a0",
        "orig_a1",
        "orig_speaker",
        "orig_listener",
        "orig_a0_individual",
        "orig_a1_individual",
        "collective_inhibition",
    ]
    + METRIC_NAMES
    + COR_NAMES
)


def _get_speaker_listener_metric(df, metric, aid, affix):
    """Util function to compute metric relative to turn-holder"""
    df[f"{metric}_speaker{affix}"] = np.where(
        df["agent_speaking"] == "agent_0",
        df[f"{metric}_{aid}0{affix}"],
        df[f"{metric}_{aid}1{affix}"],
    )
    df[f"{metric}_listener{affix}"] = np.where(
        df["agent_speaking"] == "agent_0",
        df[f"{metric}_{aid}1{affix}"],
        df[f"{metric}_{aid}0{affix}"],
    )
    return df


def _get_diff_metric(df, metric, aid, affix):
    """Util function to compute difference in metric between agents"""
    mname_0 = f"{metric}_{aid}0{affix}"
    mname_1 = f"{metric}_{aid}1{affix}"
    df[f"{metric}_diff{affix}"] = abs(df[mname_0] - df[mname_1])
    return df


def get_individual_aggs(f):
    """Util function to compute aggregates in individual performance data"""
    log = pd.read_csv(f)
    ind_agg = log.groupby("iter").agg(INDIVIDUAL_DICT).reset_index()
    ind_agg.columns = INDIVIDUAL_NAMES
    try:
        ind_agg["noise_level"] = ind_agg["agent_name"].str.split("_")
        ind_agg["noise_level"] = ind_agg["noise_level"].str[1].astype(float)
    except:
        ind_agg["noise_level"] = -1.0
    return ind_agg


def _add_cors(pair_agg, log, v1, v2, o):
    """Util function to compute correlations between metrics"""
    c_agg = log.groupby("iter").apply(lambda x: x[v1].corr(x[v2]))
    c_agg = c_agg.reset_index()
    c_agg.columns = ["iter", o]
    return pair_agg.merge(c_agg)


def get_pair_aggs(f):
    """Function to compute aggregates in pair performance data (full pipeline).
    This function is applied to a single logfile from a pair.
    """
    log = pd.read_csv(f)
    # Code for speaker
    log["agent_0"] = log["log_id"].str.split("_").str[:3].str.join("_").iloc[0]
    log["agent_1"] = log["log_id"].str.split("_").str[3:].str.join("_").iloc[0]
    log["agent_speaking"] = np.where(
        log["agent"] == log["agent_0"], "agent_0", "agent_1"
    )
    log = log.rename({"prob_a0": "jump_a0", "prob_a1": "jump_a1"}, axis=1)
    # Compute response depletion
    vname = "resp_neighbors"
    o_vname = "resp_depletion"
    log[f"{o_vname}_a0"] = 1 - (log[f"{vname}_a0_current"] / log[f"{vname}_a0"])
    log[f"{o_vname}_a1"] = 1 - (log[f"{vname}_a1_current"] / log[f"{vname}_a1"])
    # Get speaker-listener size of jump (used to compute exploration)
    log = _get_speaker_listener_metric(log, "jump", aid="a", affix="")
    # Get a0-a1 differences for a number of metrics
    log = _get_diff_metric(log, "resp_knnd_5", aid="a", affix="")
    log = _get_diff_metric(log, "resp_knnd_5", aid="a", affix="_current")
    log = _get_diff_metric(log, "resp_depletion", aid="a", affix="")
    log = _get_diff_metric(log, "resp_neighbors", aid="a", affix="")
    # Get a range of measures for relative to turn holder
    log = _get_speaker_listener_metric(log, "resp_knnd_5", aid="a", affix="")
    log = _get_speaker_listener_metric(log, "resp_knnd_5", aid="a", affix="_current")
    log = _get_speaker_listener_metric(log, "resp_neighbors", aid="a", affix="")
    log = _get_speaker_listener_metric(log, "resp_neighbors", aid="a", affix="_current")
    # Compute response depletion for speaker and listener
    vname_s = "resp_neighbors_speaker"
    vname_l = "resp_neighbors_listener"
    log[f"{o_vname}_speaker"] = 1 - (log[f"{vname_s}_current"] / log[vname_s])
    log[f"{o_vname}_listener"] = 1 - (log[f"{vname_l}_current"] / log[vname_l])
    # Use the aggregation dictionary to compute trial-level estimates
    pair_agg = log.groupby("iter").agg(PAIR_DICT).reset_index()
    pair_agg.columns = PAIR_NAMES
    # Add correlations variables
    pair_agg = _add_cors(
        pair_agg, log, "resp_neighbors_a0", "resp_neighbors_a1", "resp_neighbors_cor"
    )
    pair_agg = _add_cors(
        pair_agg,
        log,
        "resp_neighbors_a0_current",
        "resp_neighbors_a1_current",
        "resp_neighbors_current_cor",
    )
    pair_agg = _add_cors(pair_agg, log, "jump_a0", "jump_a1", "jump_cor")
    # Add info on diversity levels
    pair_agg["noise_level"] = pair_agg["agent_0"].str.split("_").str[1]
    pair_agg["noise_level"] = pair_agg["noise_level"].astype(float)
    return pair_agg


def concat_dfs(result_list):
    """Concatenate dataframes in list as single df
    This function is applied to outputs from multiple pairs
    (see postprocess.py)
    """
    for idx, r in enumerate(result_list):
        if idx == 0:
            out = r
        else:
            out = pd.concat([out, r], ignore_index=True)
    return out


def merge_pairs_inds(pdf, idf, agent_nr):
    """Merge df of pair aggregate metrics with individual aggregates
    Needed because metrics for individual performance are also
    included in the final pairwise logs
    """
    pdf = pdf.merge(
        idf,
        right_on=["agent_name", "init_seed"],
        left_on=[f"agent_{agent_nr}", "init_seed"],
        how="outer",
    ).drop(["agent_name"], axis=1)
    if agent_nr == "0":
        pdf.drop("noise_level_x", axis=1, inplace=True)
        pdf = pdf.rename(RENAME_0, axis=1)
    else:
        pdf.drop("noise_level", axis=1, inplace=True)
        pdf = pdf.rename(RENAME_1, axis=1)
    return pdf


def get_unique_named(f, LOG_PATH, date, n_back, threshold=0.01179):
    """
    Get dataframe with number of unique words named for simulations
    with a given initial seed. For pairs, this is the same as
    performance. For individuals, this is the lenght of the concatenated
    lists of animals named by the two agents in individual simulations.
    This is used to get collective inhibition estimates.
    """
    log = pd.read_csv(f)
    pair_id = log.log_id.iloc[0]
    pair_counts = log.groupby("init_seed").response.count().reset_index()
    pair_counts = pair_counts.rename({"response": "unique_pair"}, axis=1)
    a0_name = log["log_id"].str.split("_").str[:3].str.join("_").iloc[0]
    a1_name = log["log_id"].str.split("_").str[3:].str.join("_").iloc[0]
    IND_PATH = f"{LOG_PATH}/{date}/{n_back}_back"
    a0_log = pd.read_csv(f"{IND_PATH}/individual/{a0_name}_1_{threshold}.txt")
    a1_log = pd.read_csv(f"{IND_PATH}/individual/{a1_name}_1_{threshold}.txt")
    a0_list = a0_log.groupby("init_seed").agg({"response": list}).reset_index()
    a1_list = a1_log.groupby("init_seed").agg({"response": list}).reset_index()
    counts = a0_list.merge(a1_list, on="init_seed")
    counts["unique_individual"] = (counts["response_x"] + counts["response_y"]).apply(
        lambda x: len(set(x))
    )
    counts = counts.drop(["response_x", "response_y"], axis=1)
    counts = pair_counts.merge(counts, on="init_seed")
    counts["pair"] = pair_id
    return counts


def get_wd_originality_scores(f):
    """Util function used to compute how many times a given
    agent names each word over the entire set of simulations
    that are run for that agent. This is applied to a single agents.
    In postprocess.py, the results are aggregated across multiple
    agents to get overall originality scores.
    """
    data = pd.read_csv(f)
    counts = data.groupby("pos_response_a0")["agent"].count().reset_index()
    counts.columns = ["word", "count"]
    return counts


def _get_trial_level_orig(log, merge_col, wd_orig, agent):
    """Get originality for a given agent in a pair (a0, a1)
    Or relative to the turn-holder.
    """
    if agent in ["a0", "a0_individual", "a1_individual"]:
        left_col = "pos_response_a0"
    elif agent == "a1":
        left_col = "pos_response_a1"
    elif agent == "listener":
        left_col = "pos_listener"
    elif agent == "speaker":
        left_col = "pos_speaker"
    merged = log.merge(wd_orig[merge_col], left_on=left_col, right_on="word").drop(
        "word", axis=1
    )
    merged = (
        merged.groupby("init_seed").agg({"originality_score": np.nanmean}).reset_index()
    )
    merged.columns = ["init_seed", f"orig_{agent}"]
    return merged


def get_originality(f, wd_orig, LOG_PATH, date, n_back, threshold=0.01179):
    """Compute average originality of each simulation for pairs/individuals,
    based on originality scores of the words the name in the simulation
    """
    # Add some metrics to the pair's dataframe
    merge_col = ["word", "originality_score"]
    log = pd.read_csv(f)
    log["agent_0"] = log["log_id"].str.split("_").str[:3].str.join("_").iloc[0]
    log["agent_1"] = log["log_id"].str.split("_").str[3:].str.join("_").iloc[0]
    log["agent_speaking"] = np.where(
        log["agent"] == log["agent_0"], "agent_0", "agent_1"
    )
    log["pos_speaker"] = np.where(
        log["agent_speaking"] == "agent_0",
        log["pos_response_a0"],
        log["pos_response_a1"],
    )
    log["pos_listener"] = np.where(
        log["agent_speaking"] == "agent_0",
        log["pos_response_a1"],
        log["pos_response_a0"],
    )
    pair_id = log.log_id.iloc[0]
    # Compute originality for agents & from the perspective of the turn-holder
    for agent in ["a0", "a1", "listener", "speaker"]:
        out = _get_trial_level_orig(log, merge_col, wd_orig, agent)
        if agent == "a0":
            merged_pair = out
        else:
            merged_pair = merged_pair.merge(out, on="init_seed")
    merged_pair["pair"] = pair_id
    # Get originality of individual agents
    a0_name = log["log_id"].str.split("_").str[:3].str.join("_").iloc[0]
    IND_PATH = f"{LOG_PATH}/{date}/{n_back}_back"
    a0_log = pd.read_csv(f"{IND_PATH}/individual/{a0_name}_1_{threshold}.txt")
    a1_name = log["log_id"].str.split("_").str[3:].str.join("_").iloc[0]
    a1_log = pd.read_csv(f"{IND_PATH}/individual/{a1_name}_1_{threshold}.txt")
    merged_a0 = _get_trial_level_orig(a0_log, merge_col, wd_orig, agent="a0_individual")
    merged_a1 = _get_trial_level_orig(a1_log, merge_col, wd_orig, agent="a1_individual")
    # Merge the individual originality dataframe with the pair dataframe
    merged_pair = merged_pair.merge(merged_a0, on="init_seed")
    merged_pair = merged_pair.merge(merged_a1, on="init_seed")
    return merged_pair


def rename_metrics(df):
    """Rename some metrics"""
    df.rename(
        {
            "noise_level_a0": "diversity_level",
            "performance_a0": "fluency_a0",
            "performance_a1": "fluency_a1",
            "performance_pair": "fluency_pair",
        },
        axis=1,
        inplace=True,
    )
    return df


def get_pair_level_aggregates(pdf):
    """Compute pair-level aggregates for all metrics"""
    aggs = pdf.groupby("pair").agg(PAIR_LEVEL_DICT).reset_index()
    aggs.columns = PAIR_LEVEL_NAMES
    return aggs
