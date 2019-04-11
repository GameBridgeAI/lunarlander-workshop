import tensorflow as tf
import numpy as np
from policy_gradient import PolicyGradient

class AgentPolicyGradient:
    def __init__(self, 
                 n_x,
                 n_y,
                 learning_rate = 0.02,
                 reward_decay=0.99,
                 load_path=None, 
                 save_path=None):
        self.PG = PolicyGradient(n_x, n_y,
                learning_rate=learning_rate,
                reward_decay=reward_decay,
                load_path=load_path,
                save_path=save_path
                )
        
        
    def choose_action(self, observation):
        return self.PG.choose_action(observation)
    
    def store_transition(self, s, a, r):
        return self.PG.store_transition(s,a,r)
    
    
    def learn(self):
        return self.PG.learn()
    
    
    def plot_cost(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.PG.cost_history)), self.PG.cost_history)
        plt.ylabel('Cost Ex')
        plt.xlabel('Training Steps Ex')
        plt.show()    
        
    def crashed(self):
        episode_rewards_sum = sum(self.PG.episode_rewards)
        return episode_rewards_sum < -250
    
    def episode_reward(self):
        episode_rewards_sum = sum(self.PG.episode_rewards)
        return episode_rewards_sum
    

    def costs(self):
        return self.PG.costs()
    
