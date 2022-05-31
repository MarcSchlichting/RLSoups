import numpy as np
from tqdm import tqdm
import torch

def top_n_mean(actions,performance,n):
    actions = torch.Tensor(actions)
    performance = torch.Tensor(performance)
    _, idx = torch.topk(performance,n)
    reduced_actions = actions[idx]
    return torch.mean(reduced_actions,dim=0).numpy()

def top_n_median(actions,performance,n):
    actions = torch.Tensor(actions)
    performance = torch.Tensor(performance)
    _, idx = torch.topk(performance,n)
    reduced_actions = actions[idx]
    median, _ = torch.median(reduced_actions,dim=0)
    median = median.numpy()
    return median
    

def evaluate_policies_individually(models:list,env,n_eval=10):
    mean_list = []
    std_list = []
    for m in tqdm(models):
        rewards_list = []
        for i in range(n_eval):
            obs = env.reset()
            done = False
            running_reward = 0
            while not done:
                action, _states = m.predict(obs)
                obs, rewards, done, info = env.step(action)
                running_reward += rewards
            rewards_list.append(running_reward)
        mean_list.append(np.mean(rewards_list))
        std_list.append(np.std(rewards_list))
    
    return mean_list,std_list


def evaluate_policies_ensemble(models:list,env,ensemble_voting_fcn,n_eval=10,ensemble_voting_function_args=[]):
    rewards_list = []
    for i in range(n_eval):
        obs = env.reset()
        done = False
        running_reward = 0
        while not done:
            action_list = []
            for m in models:
                action, _states = m.predict(obs)
                action_list.append(action)
            a = ensemble_voting_fcn(action_list,*ensemble_voting_function_args)
            obs, rewards, done, info = env.step(a)
            running_reward += rewards
        rewards_list.append(running_reward)

    return np.mean(rewards_list), np.std(rewards_list)
