#!/usr/bin/env python3

import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env=gym.make("CartPole-v1", render_mode="human")
model=DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(1e5), progress_bar=True)
model.save("DQN_cartpole")
del model

model=DQN.load("dqn_lunar", env=env)

mean_reward, std_reward=evaluate_policy(model, model.get_env(), n_eval_episodes=10)

vec_env=model.get_env()
obs=vec_env.reset()
for i in range(1000):
    action, _states=model.predict(obs, deterministic=True)
    obs, rewards, dones, info=vec_env.step(action)
    vec_env.render("human")

