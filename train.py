import gym
import numpy as np

from gymclass import Notebook

env = gym.make("LunarLander-v2")
env = env.unwrapped
#env.seed(2)
from collections import deque
from dqn_agent import DQNAgent
import time

# eps stands for Epsilon which is the amount of the Agent will randomly take actions 
# its starts being quite random and this causes it to 'explore' its options and potentially
# try things that might not work as well.  As time passes in the training we allow less 
# random actions. 

eps_start=1.0
eps_end=0.001
eps_decay=0.995
eps = eps_start  # initialize epsilon

episode_rewards = [] # List of all rewards
episode_rewards_window = deque(maxlen=100)  # last 100 scores


# Replace MYNAME with your name below. 
save_path = "output/LunarLander-Robin-v3.ckpt"
agent = DQNAgent(state_size=8, action_size=4, seed=0, hidden_layer1=64, hidden_layer2=108)

episodes = 2000
for t in Notebook.log_progress(range(episodes)):
    observation = env.reset()
    episode_reward = 0
    tic = time.perf_counter()
    steps = 0
    while True:
        # 1. Choose an action based on observation
        action = agent.act(observation, eps)
        # 2. Take action in the environment
        observation_next, reward, done, info = env.step(action)

        # 3. Now tell the agent about the action and reward so it can learn
        agent.step(observation, action, reward, observation_next, done)
        steps = steps +1
        # Taking too long (10seconds is too long)
        if steps > 1000:
            done = True

        # Oops Crashed or flew away, stops early
        if episode_reward<-400:
            done = True

        # After initial training quit early when things go wrong
        # try to amplify good experience, remove random
        #if t>100 and episode_reward<-350:
        #    done = True
        #if t>200 and episode_reward<-250:
        #    done = True
        if t>500 and steps > 350:
            done = True

        observation = observation_next
        episode_reward += reward
        if done:
            break
    # save scores and update epsilon which sets the amount of random exploration
    episode_rewards_window.append(episode_reward)
    episode_rewards.append(episode_reward)
    eps = max(eps_end, eps_decay*eps)
    raw = np.mean(episode_rewards_window)
    print("\r Episodes ", t, " Current Rolling Avg Reward ", raw, end="")

    if raw > 350:
        break;

agent.save(save_path)
agent.save_bin(save_path+'.bin')
print("")
print("Done! Average Reward =", np.mean(episode_rewards_window))
print("Average Fitness Score =", agent.fitness(np.mean(episode_rewards_window)))




