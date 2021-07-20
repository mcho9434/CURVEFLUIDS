import gym 
import gym_fluids
import numpy as np
import matplotlib.pyplot as plt
import os

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env

from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

#env = make_vec_env("fluids-v2",n_envs=1)
env = gym.make("fluids-v2")
env = Monitor(env, log_dir)

model =PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./ppo2_fluids_1e5ts_tensorboard") #use cnn training?

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# # # Train the agent
# time_steps = 1e6
# print(str(int(time_steps)))
# print("starting training")
# model.learn(total_timesteps=int(time_steps), callback=callback) #deprecated?
# model.save("ppo2_fluids_1e6ts_mcho_test")
# print(type(log_dir))
# results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO2 Fluids")
# #plt.show()
# plt.savefig('mcho1e6.png', bbox_inches='tight')
# print("done showing")
#rewards.py is fluids/utils/reward.py
#del model

model=PPO2.load("ppo2_fluids_1e5ts_mcho_test")
obs = env.reset()

#action=[steering, acc]

while True:
    action, _states = model.predict(obs)
    #action=[-1.0,1.0]

    obs, reward, done, info = env.step(action)
    env.render()
    print(obs)

#center on waypoint
#edit reward fn
#try using cnn models
#computer vision instead?
# 1 state space representation
# 2 visualization
# look at rendering code, explicit view
# 2 weeks 