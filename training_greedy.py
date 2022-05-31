import gym
import numpy as np
import copy

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.td3.policies import TD3Policy
# from wandb.integration.sb3 import WandbCallback
from average_models import create_top_model
from evaluation_methods import evaluate_policies_individually
import torch

NO_POLICIES = 10
AVERAGE_EVERY = 50000
GYM_NAME = "Hopper-v3"
NO_TOTAL_STEPS = 1000000
SAVE_ID = "hopper_greedy_50"

model = []
for i in range(NO_POLICIES):
    env = gym.make(GYM_NAME)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model.append(TD3("MlpPolicy", env, action_noise=action_noise, verbose=0, tensorboard_log="./logs/{}/".format(SAVE_ID)))
counter = 0
#trainings loop
for j in range(int(NO_TOTAL_STEPS/AVERAGE_EVERY)):

    for i in range(NO_POLICIES):
        counter = counter+1
        print(j,i)
        env = gym.make(GYM_NAME)
        model[i].set_env(env)
        if j>0:
            model[i].learn(total_timesteps=AVERAGE_EVERY,reset_num_timesteps=False,tb_log_name=str(counter).zfill(3))
        else:
            model[i].learn(total_timesteps=AVERAGE_EVERY,tb_log_name=str(counter).zfill(3))
        model[i].save("./models/{}/Hopper_{}".format(SAVE_ID,str(counter).zfill(3)))

    
    del model
    env = gym.make(GYM_NAME)
    model = [TD3.load("./models/{}/Hopper_{}".format(SAVE_ID,str(counter).zfill(3))) for i in range(NO_POLICIES)]
    #average models
    env = gym.make(GYM_NAME)
    mean_list,std_list = evaluate_policies_individually(model,env)

    m_top = create_top_model(model,mean_list)
    performance,_ = evaluate_policies_individually([m_top],env)

    model = [copy.deepcopy(m_top) for i in range(NO_POLICIES)]

    print("all weights top-performance",performance)
    print("mean",mean_list)
    
#save final model  
model[0].save("./models/{}/Hopper_final".format(SAVE_ID))

