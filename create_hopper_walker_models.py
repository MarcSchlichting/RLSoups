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
from stable_baselines3.td3.policies import TD3Policy
from helpers import constant_schedule
import torch

NO_RUNS = 10

#Hopper
model = []

for i in range(NO_RUNS):
    env = gym.make("Hopper-v3")

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    policy_kw = {"net_arch":[64,64]}
    model.append(TD3("MlpPolicy", env, action_noise=action_noise, verbose=0, policy_kwargs=policy_kw, tensorboard_log="./logs/hopper_small_policy/",seed=i))
    model[-1].learn(total_timesteps=100000)
    model[-1].save("./models/hopper_small_policy/Hopper_{}".format(str(i).zfill(2)))

# #Half Cheetah
# model = []

# for i in range(NO_RUNS):
#     env = gym.make("HalfCheetah-v3")

#     # The noise objects for TD3
#     n_actions = env.action_space.shape[-1]
#     action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#     policy_kw = {"net_arch":[64,64]}
#     model.append(TD3("MlpPolicy", env, action_noise=action_noise, verbose=0, tensorboard_log="./logs/halfcheetah_small_policy/",seed=i))
#     model[-1].learn(total_timesteps=100000)
#     model[-1].save("./models/halfcheetah_small_policy/HalfCheetah_{}".format(str(i).zfill(2)))


# #Walker
# model = []

# for i in range(NO_RUNS):
#     env = gym.make("Walker2d-v3")

#     # The noise objects for TD3
#     n_actions = env.action_space.shape[-1]
#     action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#     policy = MlpPolicy(env.observation_space,env.action_space,constant_schedule(0.001),net_arch=[64,64])
#     model.append(TD3(policy, env, action_noise=action_noise, verbose=0, tensorboard_log="./logs/walker2d_small_policy/",seed=i))
#     model[-1].learn(total_timesteps=100000)
#     model[-1].save("./models/walker2d_small_policy/Walker2d_{}".format(str(i).zfill(2)))



# model_mean = model.append(TD3("MlpPolicy", env))

# for k in model[0].get_parameters()["policy"]:
#     diff = torch.sum(torch.abs(model[0].get_parameters()["policy"][k]-model[1].get_parameters()["policy"][k]))
#     print(diff)
