import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, goal_dim, n_agents, max_size=1e6):
        self.memory_pointer = 0
        self.memory_size = 0
        self.n_agents = n_agents
        self.max_size = int(max_size)
        self.state_memory = np.zeros((self.max_size, n_agents, state_dim))
        self.action_memory = np.zeros((self.max_size, n_agents, action_dim))
        self.reward_memory = np.zeros((self.max_size, n_agents))
        self.next_state_memory = np.zeros((self.max_size, n_agents, state_dim))
        self.goal_memory = np.zeros((self.max_size, n_agents, goal_dim))
        self.gamma_memory = np.zeros((self.max_size, n_agents))
        self.terminal_memory = np.zeros(self.max_size, dtype=bool)

    def add(self, transition):
        state, action, reward, next_state, goal, gamma, done = transition
        self.memory_pointer = self.memory_pointer % self.max_size

        self.state_memory[self.memory_pointer] = state
        self.action_memory[self.memory_pointer] = action
        self.reward_memory[self.memory_pointer] = reward
        self.next_state_memory[self.memory_pointer] = next_state
        self.goal_memory[self.memory_pointer] = goal
        self.gamma_memory[self.memory_pointer] = gamma
        self.terminal_memory[self.memory_pointer] = done

        self.memory_pointer += 1
        self.memory_size = min(self.memory_size + 1, self.max_size)

    def sample(self, batch_size):
        max_mem = min(self.memory_size, self.max_size)
        batch_ids = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch_ids]
        actions = self.action_memory[batch_ids]
        rewards = self.reward_memory[batch_ids]
        next_states = self.next_state_memory[batch_ids]
        goals = self.goal_memory[batch_ids]
        gamma = self.gamma_memory[batch_ids]
        done = self.terminal_memory[batch_ids]

        return states, actions, rewards, next_states, goals, gamma, done

    def ready(self, batch_size):
        return self.memory_size >= batch_size
