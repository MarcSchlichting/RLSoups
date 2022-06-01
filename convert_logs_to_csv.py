from packaging import version

import pandas as pd
import tensorboard as tb
import os
from tqdm import tqdm

major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
    "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)

experiment_id = "dIlAtRS4SG2UY6nOrvlfgw"
try:
    os.mkdir("./csv_files/{}".format(experiment_id))
except:
    pass

print("Downloading...")
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
df = df[df["tag"]=="rollout/ep_rew_mean"]
unique_run_ids = list(df.run.unique())
print("Converting...")
for r_id in tqdm(unique_run_ids):
    file_name = "".join(r_id.split("_")[:-1])
    df.loc[df["run"]==r_id,"run"] = file_name

df = df[["run","step","value"]]    
df.to_csv("./csv_files/{}/{}.csv".format(experiment_id,experiment_id),index=False)

print("Finished Converting")