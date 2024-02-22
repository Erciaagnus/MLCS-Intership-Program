#!/usr/bin/env python3
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode='human')

def policy_evaluation(policy, env):
    V = np.zeros(env.observation_space.n)
    theta = 0.0001
    norm_track = []  # 가치 함수 벡터의 놈(norm)을 추적하기 위한 리스트
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a in range(env.action_space.n):
                for prob, nextState, reward, done in env.P[s][a]:
                    v += policy[s][a] * prob * (reward + 0.90 * V[nextState])
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v
        norm_track.append(np.linalg.norm(V, 2))  # 가치 함수 벡터의 놈(norm)을 리스트에 추가
        if delta < theta:
            break
    return V, norm_track  # 가치 함수와 가치 함수 벡터의 놈(norm) 리스트 반환

def policy_improvement(env):
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    number_iterations = 0
    optimal = False
    while not optimal:
        optimal_policy = True
        number_iterations += 1
        V, _ = policy_evaluation(policy, env)
        for s in range(env.observation_space.n):
            current_action = np.argmax(policy[s])
            A = np.zeros(env.action_space.n) # number of actions
            for a in range(env.action_space.n):
                for prob, nextState, reward, done in env.P[s][a]:
                    A[a] += prob * (reward + 0.90 * V[nextState])
            new_action = np.argmax(A)
            if current_action != new_action:
                optimal_policy = False
            policy[s] = np.zeros(env.action_space.n)
            policy[s][new_action] = 1
        if optimal_policy:
            optimal = True        
    print("Number of iterations: {}".format(number_iterations))
    return policy, V

env.reset()  # Reset the environment
env.render()

# 정책 평가 및 개선 수행
policy, value = policy_improvement(env)

# 에피소드 실행하여 결과 출력
steps = 0
while True:
    env.render()
    action = np.argmax(policy[env.env.s])
    observation, reward, done, _, info = env.step(action)
    print('reward is ', reward)
    steps += 1
    if reward == 20:
        print("Episode finished after {} timesteps".format(steps + 1))
        break

# 가치 함수 벡터의 놈(norm) 추적을 통한 수렴 그래프 표시
plt.plot(reward)
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.title('Convergence of Value Function')
plt.show()
