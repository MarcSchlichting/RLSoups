import gym
import numpy as np
import copy

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.td3.policies import TD3Policy
# from wandb.integration.sb3 import WandbCallback
from average_models import create_top_model,create_top_n_mean_model_parameters
from evaluation_methods import evaluate_policies_individually
import torch
from custom_callback import CurrentTrainReward
from stable_baselines3.common.monitor import Monitor


NO_POLICIES = 5
AVERAGE_EVERY = 5000
GYM_NAME = "Hopper-v3"
NO_TOTAL_STEPS = 100000
SAVE_ID = "hopper_test_1"

model = []
for i in range(NO_POLICIES):
    env = gym.make(GYM_NAME)
    env = Monitor(env,"./logs/{}".format(SAVE_ID))
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model.append(TD3("MlpPolicy", env, action_noise=action_noise, verbose=0, tensorboard_log="./logs/{}/".format(SAVE_ID)))
counter = 0
#trainings loop
for j in range(int(NO_TOTAL_STEPS/AVERAGE_EVERY)):

    current_train_reward = [CurrentTrainReward(log_dir="./logs/{}".format(SAVE_ID)) for _ in range(NO_POLICIES)]

    for i in range(NO_POLICIES):
        counter = counter+1
        print(j,i)
        env = gym.make(GYM_NAME)
        env = Monitor(env,"./logs/{}".format(SAVE_ID))
        model[i].set_env(env)
        if j>0:
            model[i].learn(total_timesteps=AVERAGE_EVERY,reset_num_timesteps=False,tb_log_name=str(counter).zfill(3),callback=current_train_reward[i])
        else:
            model[i].learn(total_timesteps=AVERAGE_EVERY,tb_log_name=str(counter).zfill(3),callback=current_train_reward[i])
        model[i].save("./models/{}/Hopper_{}".format(SAVE_ID,str(counter).zfill(3)))

    #get mean performance
    performance = [m.current_train_reward for m in current_train_reward]

    del model
    env = gym.make(GYM_NAME)
    env = Monitor(env,"./logs/{}".format(SAVE_ID))
    model = [TD3.load("./models/{}/Hopper_{}".format(SAVE_ID,str(counter).zfill(3))) for i in range(NO_POLICIES)]
    #average models
    env = gym.make(GYM_NAME)
    env = Monitor(env,"./logs/{}".format(SAVE_ID))
    mean_list,std_list = evaluate_policies_individually(model,env)
    print(mean_list)
    print(performance)

    m_top = create_top_model(model,mean_list)
    # m_top = create_top_n_mean_model_parameters(model,mean_list,2)
    performance,_ = evaluate_policies_individually([m_top],env)

    model = [copy.deepcopy(m_top) for i in range(NO_POLICIES)]

    print("all weights top-performance",performance)
    print("mean",mean_list)
    
#save final model  
model[0].save("./models/{}/Hopper_final".format(SAVE_ID))

