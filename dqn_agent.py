"""
MIT License

This work is derived from 
https://github.com/udacity/deep-reinforcement-learning
Copyright (c) 2018 Udacity

Adapted by Robin Grosset 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import random
from collections import namedtuple, deque

from dqn_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

import os
import struct 
from math import trunc
from collections import OrderedDict
from array import array

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, hidden_layer1=64, hidden_layer2=108):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_layer1, hidden_layer2).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_layer1, hidden_layer2).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def save(self, file_name):
        torch.save(self.qnetwork_local.state_dict(), file_name)  

    def load(self, file_name):
        self.qnetwork_local.load_state_dict(torch.load(file_name))

    def save_bin(self, file_name):
        torch.save(self.qnetwork_local.state_dict(), file_name)  
        keymap = self.qnetwork_local.state_dict()
        outputfloats = []

        for key, value in keymap.items():
            # ignore keys order is all that matteers
            if isinstance(value, torch.Tensor):
                tup = value.size()
                lv = value.tolist()
                if isinstance(lv, float):
                    outputfloats.append(lv)
                else:
                    for row in lv:
                        if isinstance(row, float):
                            outputfloats.append(row)
                        else:
                            for item in row:
                                outputfloats.append(item)                                                
        while (len(outputfloats) < 8192):
            outputfloats.append(0.0)
            
        output_file = open(file_name, 'wb')
        float_array = array('f', outputfloats)
        float_array.tofile(output_file)
        output_file.close()    

    def load_bin(self, file_name):
        keymap = self.qnetwork_local.state_dict()
        new_keymap = OrderedDict()
        sz = os.path.getsize(file_name)
        input_file = open(file_name, 'rb')
        n = sz / 4
        buff = input_file.read(sz)
        fmtstr = '{:d}f'.format(trunc(n))
        inputfloats = struct.unpack(fmtstr,buff )
        input_file.close()    
        index = 0
        for key, value in keymap.items():
            # ignore keys order is all that matteers
            if isinstance(value, torch.Tensor):
                tup = value.size()
                if len(tup) ==2:
                    dtensor = []
                    for row in range(tup[0]):
                        trow = []
                        for col in range(tup[1]):
                            trow.append(inputfloats[index])
                            index+=1
                        dtensor.append(trow)
                    tensor_from_list = torch.FloatTensor(dtensor)
                    new_keymap[key] = tensor_from_list
                else:
                    dtensor = []
                    for row in range(tup[0]):
                        dtensor.append(inputfloats[index])
                        index+=1                
                    tensor_from_list = torch.FloatTensor(dtensor)
                    new_keymap[key] = tensor_from_list
        self.qnetwork_local.load_state_dict(new_keymap)            

    def weights(self):
        return self.qnetwork_local.state_dict()


    def fitness(self, episode_reward):
        # How we calculate the % fitness
        # Episide reward is a value from about -500 to +300
        # < 0 is very bad crash
        # < 100 is bad crash   0%
        # < 150 is failure
        # < 200 is poor
        # > 200 is okay, pass mark  50%
        # > 220 is good
        # > 240 is very good
        # > 270 is excellent 
        # > 300 is perfect, 100%

        # Assuming 100% = 300 and 100 is 0
        fitness = episode_reward - 100
        if fitness < 0:
            fitness = 0 # Bad lowest floor of 0%, anything below is consisdered 0%

        # Now divide score by 2 to get %
        fitness = fitness / 2
        if fitness>100:
            fitness = 100 # Highest cap to 100%, anything above is considered 100%

        # Note that 50% is now considered 'okay' its a successful landing, passing mark
        return fitness        

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)