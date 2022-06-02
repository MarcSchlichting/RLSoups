import gym
import numpy as np
import copy

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.td3.policies import TD3Policy
from average_models import create_weighted_mean_model_parameters
from evaluation_methods import evaluate_policies_individually
from helpers import constant_schedule
import torch

NO_POLICIES = 10
AVERAGE_EVERY = 10000
GYM_NAME = "Hopper-v3"
NO_TOTAL_STEPS = 100000

model = []
for i in range(NO_POLICIES):
    env = gym.make(GYM_NAME)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model.append(TD3("MlpPolicy", env, action_noise=action_noise, verbose=0, tensorboard_log="./logs/hopper_averaged_training/",seed=i))

#trainings loop
for j in range(int(NO_TOTAL_STEPS/AVERAGE_EVERY)):

    for i in range(NO_POLICIES):
        print(j,i)
        env = gym.make(GYM_NAME)
        model[i].set_env(env)
        model[i].learn(total_timesteps=AVERAGE_EVERY)
        model[i].save("./models/hopper_averaged_training/Hopper_{}".format(str(i*j+i).zfill(3)))
    
    
    del model
    env = gym.make(GYM_NAME)
    model = [TD3.load("./models/hopper_averaged_training/Hopper_{}".format(str(i*j+i).zfill(3))) for i in range(NO_POLICIES)]
    #average models
    env = gym.make(GYM_NAME)
    mean_list,std_list = evaluate_policies_individually(model,env)

    weights = torch.nn.functional.softmax(torch.Tensor(mean_list))
    m_avg = create_weighted_mean_model_parameters(model,weights)
    performance,_ = evaluate_policies_individually([m_avg],env)

    model = [copy.deepcopy(m_avg) for i in range(NO_POLICIES)]

    print("all weights softmax-averages",performance)
    
#save final model  
model[0].save("./models/hopper_averaged_training/Hopper_final")