import random
import torch
import numpy as np
from MADDPG import MADDPG
from utils import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HAC:
    def __init__(self, k_level, n_agents, H, obs_dim, goal_state_dim, action_dim,
                 threshold, lr, lamda, gamma, action_clip_low,
                 action_clip_high, state_clip_low, state_clip_high):
        # adding lowest level (primitive actions)
        self.HAC = [MADDPG(0, n_agents, obs_dim, goal_state_dim, action_dim, lr, H, "softmax")]
        self.replay_buffer = [ReplayBuffer(obs_dim, action_dim, goal_state_dim, n_agents)]

        # adding remaining levels
        for level in range(1, k_level):
            self.HAC.append(MADDPG(level, n_agents, obs_dim, goal_state_dim, goal_state_dim, lr, H, "tanh"))
            self.replay_buffer.append(ReplayBuffer(obs_dim, goal_state_dim, goal_state_dim, n_agents))

        # set some parameters
        self.k_level = k_level
        self.n_agents = n_agents
        self.H = H
        self.action_dim = action_dim
        self.state_dim = obs_dim
        self.threshold = threshold
        self.noise_rate_goal = None
        self.noise_rate_action = None
        self.lamda = lamda
        self.gamma = gamma
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high

        # logging parameters
        self.reward = 0
        self.timestep = 0
        self.goal_rewards = {k: [] for k in range(k_level)}

    def check_goal(self, state, goal, threshold):
        return np.all(np.abs(state - goal) < threshold, axis=1)

    def goal_distance(self, state, goal):
        # MSE between state and goal as reward
        return -(np.abs(state - goal) ** 0.5).mean(axis=1)

    def run_HAC(self, env, i_level, obs, state, goal, is_subgoal_test, state_bits):
        next_obs = None
        next_state = None
        done = False
        goal_transitions = []
        goal_reward = []

        # H attempts
        for _ in range(self.H):
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test

            # actions = []
            # for agent_idx in range(self.n_agents):
            #     actions.append(self.HAC[i_level].select_action(obs[agent_idx], goal[agent_idx]))
            # action = np.array(actions)

            action = self.HAC[i_level].select_action(obs, goal)

            #   <================ high level policy ================>
            if i_level > 0:
                # add noise or take random action if not subgoal testing
                if not is_subgoal_test:
                    # noise = np.random.normal(size=action.shape)
                    # action = action + noise * self.noise_rate_goal
                    # action = action.clip(self.state_clip_low, self.state_clip_high)

                    noise = np.random.normal(0, self.noise_rate_goal *
                                             (self.state_clip_high - self.state_clip_low))
                    action += noise
                    action = action.clip(self.state_clip_low, self.state_clip_high)

                # Determine whether to test subgoal (action)
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True

                # Pass subgoal to lower level
                next_obs, next_state, done = self.run_HAC(env, i_level - 1, obs, state, action,
                                                          is_next_subgoal_test, state_bits)

                # if subgoal was tested but not achieved, add subgoal testing transition
                # if is_next_subgoal_test and not np.all(self.check_goal(next_state, action, self.threshold)):
                #     self.replay_buffer[i_level].add((obs, action, self.goal_distance(next_state, goal) * self.H,
                #                                      next_obs, goal, np.array([0.0] * self.n_agents), float(done)))

                if is_next_subgoal_test and not np.all(self.check_goal(next_state, action, self.threshold)):
                    self.replay_buffer[i_level].add((obs, action, np.array([-self.H] * self.n_agents),
                                                     next_obs, goal, np.array([0.0] * self.n_agents), float(done)))

                # for hindsight action transition
                action = next_state

            #   <================ low level policy ================>
            else:
                # add noise or take random action if not subgoal testing
                if not is_subgoal_test:
                    noise = np.random.rand(*action.shape)
                    action = action + noise * self.noise_rate_action
                    action = action.clip(self.action_clip_low, self.action_clip_high)

                # take primitive action
                available_actions = np.array(env.get_avail_actions())
                act_range = np.tile(np.arange(env.n_actions), env.n_agents).reshape(env.n_agents, -1)
                act_probs = action * available_actions
                act_probs /= act_probs.sum(axis=1).reshape(env.n_agents, -1)
                actions = [random.choices(act, weights=p)[0] if not a_act[0] else 0 for act, a_act, p in
                           zip(act_range, available_actions, act_probs)]
                action = act_probs

                try:
                    rew, done, info = env.step(actions)
                except Exception as e:
                    print(f"Exception while env.step(actions): {e}")
                    print(f"actions: {actions}, avail_acts: {available_actions}, act_probs: {act_probs}")

                    new_actions = []
                    ag_ids, act_ids = available_actions.nonzero()
                    for ag_idx in range(self.n_agents):
                        new_actions.append(np.random.choice(act_ids[ag_ids == ag_idx]))

                    print(f"new_actions: {new_actions}")
                    if len(new_actions) == self.n_agents:
                        rew, done, info = env.step(new_actions)
                    else:
                        print("continue without env.step")
                        continue

                next_obs = env.get_obs()
                next_state = env.get_state()
                next_state = next_state[state_bits]

                # this is for logging
                self.reward += rew
                self.timestep += 1

            #   <================ finish one step/transition ================>

            # check if goal is achieved
            goal_achieved = self.check_goal(next_state, goal, self.threshold)

            # hindsight action transition
            rewards = np.float32(goal_achieved) - 1  # 0 or -1
            # rewards = self.goal_distance(next_state, goal)  # MSE rewards
            gammas = np.float32(np.logical_not(goal_achieved)) * self.gamma  # 0 or gamma
            self.replay_buffer[i_level].add((obs, action, rewards, next_obs, goal, gammas, float(done)))

            # copy for goal transition
            # goal_transitions.append([obs, action, next_state, next_obs, None,
            #                          np.array([self.gamma] * self.n_agents), float(done)])
            goal_transitions.append([obs, action, -1.0, next_obs, None,
                                     np.array([self.gamma] * self.n_agents), float(done)])

            obs = next_obs
            state = next_state
            # goal_reward = rewards
            goal_reward = self.goal_distance(next_state, goal)

            if done or np.all(goal_achieved):
                done = True
                break

        #   <================ finish H attempts ================>

        self.goal_rewards[i_level].append(goal_reward)

        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1][2] = np.zeros(self.n_agents)
        goal_transitions[-1][5] = np.zeros(self.n_agents)
        for transition in goal_transitions:
            # last state is goal for all transitions
            # transition[2] = self.goal_distance(transition[2], np.repeat(next_state[None, ...], self.n_agents, axis=0))
            transition[4] = next_state
            self.replay_buffer[i_level].add(tuple(transition))

        return next_obs, next_state, done

    def update(self, batch_size):
        losses = {}
        for i in range(self.k_level):
            if self.replay_buffer[i].ready(batch_size):
                losses[i] = self.HAC[i].update(self.replay_buffer[i], batch_size)
        return losses

    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name + '_level_{}'.format(i))

    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name + '_level_{}'.format(i))
