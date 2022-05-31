from cv2 import resizeWindow
import gym
import numpy as np
import torch

from stable_baselines3 import TD3
from evaluation_methods import evaluate_policies_ensemble, evaluate_policies_individually, top_n_mean, top_n_median
# from stable_baselines3.common.evaluation import evaluate_policy

# env = gym.make("Hopper-v3")
# models = [TD3.load("./models/hopper/Hopper_{}.zip".format(str(i).zfill(2))) for i in range(10)]

env = gym.make("Walker2d-v3")
models = [TD3.load("./models/walker2d/Walker2d_{}.zip".format(str(i).zfill(2))) for i in range(10)]

# env = gym.make("HalfCheetah-v3")
# models = [TD3.load("./models/halfcheetah/HalfCheetah_{}.zip".format(str(i).zfill(2))) for i in range(10)]

mean_list,std_list = evaluate_policies_individually(models,env)

print("mean",np.mean(mean_list))
print("top 3 mean",torch.mean(torch.topk(torch.Tensor(mean_list),3)[0]))

m,s = evaluate_policies_ensemble(models,env,top_n_mean,ensemble_voting_function_args=[mean_list,10])

print("emsemble mean", m)

m,s = evaluate_policies_ensemble(models,env,top_n_median,ensemble_voting_function_args=[mean_list,10])

print("ensemble median", m)

m,s = evaluate_policies_ensemble(models,env,top_n_mean,ensemble_voting_function_args=[mean_list,3])

print("top 3 emsemble mean", m)

m,s = evaluate_policies_ensemble(models,env,top_n_median,ensemble_voting_function_args=[mean_list,3])

print("top 3 ensemble median", m)
