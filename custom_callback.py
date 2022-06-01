import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class CurrentTrainReward(BaseCallback):

    def __init__(self, log_dir: str, verbose: int = 1):
        super(CurrentTrainReward, self).__init__(verbose)
        self.log_dir = log_dir
        self.current_train_reward = -np.inf
    
    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:

        # Retrieve training reward
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
            # Mean training reward over the last 100 episodes
            self.current_train_reward = np.mean(y[-10:])
            print("Current Training Reward", self.current_train_reward)
