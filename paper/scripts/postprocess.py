import glob
import itertools
import pandas as pd
import numpy as np
from utils import (
    get_individual_aggs,
    get_pair_aggs,
    concat_dfs,
    merge_pairs_inds,
    get_unique_named,
    get_wd_originality_scores,
    get_originality,
    rename_metrics,
    get_pair_level_aggregates,
)
from multiprocessing import Pool
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--interaction-type",
    type=str,
    default="strict",
    help="strict, flexible or shortest",
)
parser.add_argument("--n-back", type=int, default=0)
parser.add_argument("--threads", type=int, default=1)

ID = "experiments"
LOG_PATH = "../logs"
AN_PATH = "../metrics"


def _get_aggs(fn, flist, threads):
    """Util function to compute simulation-level aggregates in parallel"""
    pool = Pool(threads)
    results = pool.map(fn, flist)
    pool.close()
    aggs = concat_dfs(results)
    return aggs


def _merge_aggs(pdf, idf):
    """Util function to merge individual and pair aggregates
    (simulation-level)
    """
    if all([i in idf.columns for i in ["iter", "threshold"]]):
        idf.drop(["iter", "threshold"], axis=1, inplace=True)
    if all([p in pdf.columns for p in ["iter", "threshold"]]):
        pdf.drop(["iter", "threshold"], axis=1, inplace=True)
    pdf = merge_pairs_inds(pdf, idf, "0")
    pdf = merge_pairs_inds(pdf, idf, "1")
    return pdf


def _get_unique_named(flist, pdf, n_back, threads):
    """Util function to compute collective inhibition"""
    pool = Pool(threads)
    results = pool.starmap(
        get_unique_named,
        zip(flist, [LOG_PATH] * len(flist), [ID] * len(flist), [n_back] * len(flist)),
    )
    pool.close()
    unique_named = concat_dfs(results)
    pdf = pdf.merge(unique_named, on=["init_seed", "pair"], how="outer")
    outname = "collective_inhibition"
    pdf[outname] = pdf["unique_individual"] - pdf["unique_pair"]
    pdf[outname] = (pdf[outname]) / 240 * 100
    return pdf


def _get_originality(iflist, pflist, pdf, n_back, threads):
    """Util function to compute originality"""
    pool = Pool(threads)
    results = pool.map(get_wd_originality_scores, iflist)
    pool.close()
    wd_orig = concat_dfs(results)
    wd_orig = wd_orig.groupby("word")["count"].sum().reset_index()
    norm = 240 * 100 * 21  # words * agents * nr_populations
    wd_orig["originality_score"] = (norm - wd_orig["count"]) / norm
    pool = Pool(threads)
    results = pool.starmap(
        get_originality,
        zip(
            pflist,
            [wd_orig] * len(pflist),
            [LOG_PATH] * len(pflist),
            [ID] * len(pflist),
            [n_back] * len(pflist),
        ),
    )
    pool.close()
    orig_df = concat_dfs(results)
    pdf = pdf.merge(orig_df, on=["init_seed", "pair"], how="outer")
    return pdf


def postprocess(int_str, n_back, threads):
    """Function to run full post-processing pipeline
    Args:
        int_str (str): type of interaction (strict, shortest, flexible)
        n_back (int): n-back condition
        threads (int): number of parallel processes
    """
    # Get log fnames for both pairs and individual
    fs = glob.glob(f"{LOG_PATH}/{ID}/{n_back}_back/individual/*")
    pair_fs = glob.glob(f"{LOG_PATH}/{ID}/{n_back}_back/{int_str}/*")

    # Get simulation-level aggregates
    print("*** Computing simulation-level aggregates ***")
    ia = _get_aggs(get_individual_aggs, fs, threads)
    pa = _get_aggs(get_pair_aggs, pair_fs, threads)

    # Add missing trials (e.g., trials where you don't go beyond a single association)
    animals = pa.init_seed.unique().tolist()
    pairs = pa.pair.unique().tolist()
    agents = ia.agent_name.unique().tolist()
    full_df = pd.DataFrame(list(product(pairs, animals)), columns=["pair", "init_seed"])
    full_df["agent_0"] = full_df["pair"].str.split("_").str[:3].str.join("_")
    full_df["agent_1"] = full_df["pair"].str.split("_").str[3:].str.join("_")
    pa = pa.merge(full_df, on=["pair", "init_seed", "agent_0", "agent_1"], how="outer")
    full_df_ind = pd.DataFrame(
        list(product(agents, animals)), columns=["agent_name", "init_seed"]
    )
    ia = ia.merge(full_df_ind, on=["agent_name", "init_seed"], how="outer")
    # Merge individual and pair simulation-level aggregates
    pa = _merge_aggs(pa, ia)
    for c in ["a1", "a0", "pair"]:
        pa[f"performance_{c}"] = pa[f"performance_{c}"].fillna(0)

    # Compute collective inhibition
    print("*** Computing collective inhibition metrics ***")
    pa = _get_unique_named(pair_fs, pa, n_back, threads)

    # Compute originality
    print("*** Computing originality ***")
    pa = _get_originality(fs, pair_fs, pa, n_back, threads)

    # Compute pair-level aggregates
    print("*** Summarizing ***")
    pa = rename_metrics(pa)
    aggs = get_pair_level_aggregates(pa)

    # Add metadata & save
    print("*** Saving ***")
    aggs["n_back"] = n_back
    aggs["interaction_type"] = int_str
    aggs.to_csv(
        f"{AN_PATH}/{n_back}_back/aggregates_{int_str}.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    args = parser.parse_args()
    postprocess(args.interaction_type, args.n_back, args.threads)
