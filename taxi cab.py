#!/usr/bin/env python
# coding: utf-8

# In[126]:


import numpy as np

import random


# In[127]:


import gym
env = gym.make("Taxi-v2").env

env.render()


# In[128]:


env.reset() # reset environment to a new, random state
env.render()

action_size = env.action_space.n
print("Action size ", action_size)

state_size = env.observation_space.n
print("State size ", state_size)


# In[129]:


q_table = np.zeros([env.observation_space.n, env.action_space.n])
print (q_table)


# In[130]:


get_ipython().run_cell_magic('time', '', '"""Training the agent"""\n\nimport random\nfrom IPython.display import clear_output\n\n# Hyperparameters\nalpha = 0.1\ngamma = 0.6\nepsilon = 0.1\nmax_steps = 199  \ntotal_test_episodes = 200   \n\n# For plotting metrics\nall_epochs = []\nall_penalties = []\n\nfor i in range(1, 100001):\n    state = env.reset()\n\n    epochs, penalties, reward, = 0, 0, 0\n    done = False\n    \n    while not done:\n        if random.uniform(0, 1) < epsilon:\n            action = env.action_space.sample() # Explore action space\n        else:\n            action = np.argmax(q_table[state]) # Exploit learned values\n\n        next_state, reward, done, info = env.step(action) \n        \n        old_value = q_table[state, action]\n        next_max = np.max(q_table[next_state])\n        \n        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)\n        q_table[state, action] = new_value\n\n        if reward == -10:\n            penalties += 1\n\n        state = next_state\n        epochs += 1\n        \n    if i % 100 == 0:\n        clear_output(wait=True)\n        print("Episode:",{i})\n\nprint("Training finished.\\n")')


# In[141]:


#evaluate the agent's performance

from IPython.display import clear_output
from time import sleep


env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    #print("****************************************************")
    #print("EPISODE ", episode)

    for step in range(max_steps):
        # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
        env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(q_table[state])
        
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        if done:
            rewards.append(total_rewards)
            print ("Score", total_rewards)
            break
        state = new_state
env.close()
print ("Score over time: " +  str(sum(rewards)/total_test_episodes))


# In[ ]:





# In[ ]:




