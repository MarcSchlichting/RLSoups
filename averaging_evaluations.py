from re import M
from cv2 import resizeWindow
import gym
import numpy as np
import torch

from stable_baselines3 import TD3
from evaluation_methods import evaluate_policies_individually
from average_models import create_mean_model_parameters,create_top_n_mean_model_parameters, create_weighted_mean_model_parameters,create_median_model_parameters, create_top_n_median_model_parameters
# from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("Hopper-v3")
models = [TD3.load("./models/hopper/Hopper_{}.zip".format(str(i).zfill(2))) for i in range(10)]

# env = gym.make("Walker2d-v3")
# models = [TD3.load("./models/walker2d/Walker2d_{}.zip".format(str(i).zfill(2))) for i in range(10)]

# env = gym.make("HalfCheetah-v3")
# models = [TD3.load("./models/halfcheetah/HalfCheetah_{}.zip".format(str(i).zfill(2))) for i in range(10)]

mean_list,std_list = evaluate_policies_individually(models,env)
print("mean",np.mean(mean_list))

#averaging everything
m_avg = create_mean_model_parameters(models)
performance,_ = evaluate_policies_individually([m_avg],env)
print("all weights averaged mean",performance)

m_avg = create_top_n_mean_model_parameters(models,mean_list,3)
performance,_ = evaluate_policies_individually([m_avg],env)
print("top 3 weights averaged mean",performance)

m_avg = create_median_model_parameters(models)
performance,_ = evaluate_policies_individually([m_avg],env)
print("all weights averaged median",performance)

m_avg = create_top_n_median_model_parameters(models,mean_list,3)
performance,_ = evaluate_policies_individually([m_avg],env)
print("top 3 weights averaged median",performance)

weights = torch.nn.functional.softmax(torch.Tensor(mean_list))
m_avg = create_weighted_mean_model_parameters(models,weights)
performance,_ = evaluate_policies_individually([m_avg],env)
print("all weights softmax-averages",performance)

print("stop")


