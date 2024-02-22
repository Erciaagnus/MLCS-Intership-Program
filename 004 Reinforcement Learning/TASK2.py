#!/usr/bin/env python3
import numpy as np
import gym
import matplotlib.pyplot as plt

env=gym.make("Taxi-v3", render_mode='human')

def value_iteration(env, theta=0.00001, discount_factor=0.99):
    def one_step_lookahead(state, V):
        A=np.zeros(env.env.action_space.n)
        for a in range(env.action_space.n):
            # 출력값 4개 맞는지 확인
            for prob, nextState, reward, done in env.P[state][a]:
                A[a] += prob*(reward+discount_factor*V[nextState])
        return A
    V=np.zeros(env.env.observation_space.n)
    numIterations=0
    
    while True:
        numIterations +=1
        delta=0
        for s in range(env.observation_space.n):
            qValues=one_step_lookahead(s, V)
            newValue=np.max(qValues)
            
            delta=max(delta, np.abs(newValue-V[s]))
            V[s]=newValue
        if delta < theta:
            break
    policy=np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        qValues=one_step_lookahead(s,V)
        newAction=np.argmax(qValues)
        policy[s][newAction]=1
    print(numIterations)
    return policy, V
env.reset()
policyVI, valueVI=value_iteration(env, discount_factor=0.99)
print(policyVI)

env.render()
steps=0
reward_list=[]
while True:
    env.render()
# Use the following function to see the rendering of the final policy output in the environment
    action=np.argmax(policyVI[env.env.s])
    state, reward, done, _, info = env.step(action)
    print('reward is ', reward)
    curr_state = state
    steps += 1
    reward_list.append(reward)
    if reward==20:
        print("Episode finished after {} timesteps".format(steps+1))
        break
plt.plot(reward_list)
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.title('Reward Graph')
plt.show()
