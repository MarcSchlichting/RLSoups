from packaging import version

import pandas as pd
import tensorboard as tb
import os
from tqdm import tqdm
import argparse

def main(experiment_id):
    major_ver, minor_ver, _ = version.parse(tb.__version__).release
    assert major_ver >= 2 and minor_ver >= 3, \
        "This notebook requires TensorBoard 2.3 or later."
    print("TensorBoard version: ", tb.__version__)

    # experiment_id = "HA2dimEtQKSZxb7h1ROoWw"
    try:
        os.mkdir("./csv_files/{}".format(experiment_id))
    except:
        pass

    print("Downloading...")
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df_mean_rew = df[df["tag"]=="rollout/ep_rew_mean"]
    # df_rew_non_smooth = df[df["tag"]=="non_smoothed_reward"]
    unique_run_ids = list(df_mean_rew.run.unique())
    print("Converting...")
    for r_id in tqdm(unique_run_ids):
        file_name = "".join(r_id.split("_")[:-1])
        df_mean_rew.loc[df_mean_rew["run"]==r_id,"run"] = file_name
        # df_rew_non_smooth.loc[df_rew_non_smooth["run"]==r_id,"run"] = file_name

    # df_mean_rew = df_mean_rew[["run","step","value"]]  
    # df_mean_rew.insert(4,"value_non_smoothed",list(df_rew_non_smooth["value"]))
    df_mean_rew = df_mean_rew[["run","step","value"]]
    df_mean_rew.to_csv("./csv_files/{}/{}.csv".format(experiment_id,experiment_id),index=False)

    print("Finished Converting")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='exp_id', help="Tensorboard Dev Experiment ID")

    # Parse and print the results
    args = parser.parse_args()
    print("Recognized Experiment ID: ",args.exp_id)
    main(args.exp_id)