# import gym
# import numpy as np

# from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# env = gym.make("Hopper-v3")

# # The noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1,tensorboard_log="./hopper/")
# model.learn(total_timesteps=500000, log_interval=10,eval_freq=1000)
# model.save("hopper_1")

import gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import torch

NO_RUNS = 10

# #Hopper
# model = []

# for i in range(NO_RUNS):
#     env = gym.make("Hopper-v3")

#     # The noise objects for TD3
#     n_actions = env.action_space.shape[-1]
#     action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#     model.append(TD3("MlpPolicy", env, action_noise=action_noise, verbose=0, tensorboard_log="./hopper/",seed=i))
#     model[-1].learn(total_timesteps=100000)
#     model[-1].save("Hopper_{}".format(str(i).zfill(2)))

# #Swimmer
# model = []

# for i in range(NO_RUNS):
#     env = gym.make("Swimmer-v3")

#     # The noise objects for TD3
#     n_actions = env.action_space.shape[-1]
#     action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#     model.append(TD3("MlpPolicy", env, action_noise=action_noise, verbose=0, tensorboard_log="./swimmer/",seed=i))
#     model[-1].learn(total_timesteps=100000)
#     model[-1].save("Swimmer_{}".format(str(i).zfill(2)))

#Walker
model = []

for i in range(NO_RUNS):
    env = gym.make("Walker2d-v3")

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model.append(TD3("MlpPolicy", env, action_noise=action_noise, verbose=0, tensorboard_log="./walker2d/",seed=i))
    model[-1].learn(total_timesteps=100000)
    model[-1].save("Walker2d_{}".format(str(i).zfill(2)))



# model_mean = model.append(TD3("MlpPolicy", env))

# for k in model[0].get_parameters()["policy"]:
#     diff = torch.sum(torch.abs(model[0].get_parameters()["policy"][k]-model[1].get_parameters()["policy"][k]))
#     print(diff)
