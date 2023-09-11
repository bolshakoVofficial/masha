import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, activation, lr):
        super(ActorNetwork, self).__init__()
        activations = {
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=1),
            'tanh': nn.Tanh()
        }
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            activations[activation]
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, goal):
        return self.actor(torch.hstack([state, goal]))


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, H, lr):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.H = H
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action, goal):
        # rewards are in range [-H, 0]
        return -self.critic(torch.hstack([state, action, goal])) * self.H


class Agent:
    def __init__(self, actor_dim, critic_dim, action_dim, H, activation, lr, tau=0.01):
        self.actor = ActorNetwork(actor_dim, action_dim, activation, lr).to(device)
        self.target_actor = ActorNetwork(actor_dim, action_dim, activation, lr).to(device)

        self.critic = CriticNetwork(critic_dim, H, lr).to(device)
        self.target_critic = CriticNetwork(critic_dim, H, lr).to(device)

        self.mseLoss = torch.nn.MSELoss()
        self.tau = tau
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save(self, directory, name, level, agent_idx):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), '%s/%s_level%s_agent%s_actor.pth' % (directory, name,
                                                                                 str(level), str(agent_idx)))
        torch.save(self.target_actor.state_dict(), '%s/%s_level%s_agent%s_target_actor.pth' % (directory, name,
                                                                                               str(level),
                                                                                               str(agent_idx)))
        torch.save(self.critic.state_dict(), '%s/%s_level%s_agent%s_critic.pth' % (directory, name,
                                                                                   str(level), str(agent_idx)))
        torch.save(self.target_critic.state_dict(), '%s/%s_level%s_agent%s_target_critic.pth' % (directory, name,
                                                                                                 str(level),
                                                                                                 str(agent_idx)))

    def load(self, directory, name, level, agent_idx):
        self.actor.load_state_dict(torch.load('%s/%s_level%s_agent%s_actor.pth' % (directory, name,
                                                                                   str(level), str(agent_idx)),
                                              map_location='cpu'))
        self.target_actor.load_state_dict(torch.load('%s/%s_level%s_agent%s_target_actor.pth' % (directory, name,
                                                                                                 str(level),
                                                                                                 str(agent_idx)),
                                                     map_location='cpu'))
        self.critic.load_state_dict(torch.load('%s/%s_level%s_agent%s_critic.pth' % (directory, name,
                                                                                     str(level), str(agent_idx)),
                                               map_location='cpu'))
        self.target_critic.load_state_dict(torch.load('%s/%s_level%s_agent%s_target_critic.pth' % (directory, name,
                                                                                                   str(level),
                                                                                                   str(agent_idx)),
                                                      map_location='cpu'))


class MADDPG:
    def __init__(self, level, n_agents, obs_dim, goal_state_dim, action_dim, lr, H, activation=None):
        self.level = level
        self.agents = []
        self.n_agents = n_agents
        for idx in range(n_agents):
            self.agents.append(Agent(obs_dim + goal_state_dim,
                                     (obs_dim + goal_state_dim + action_dim) * n_agents,
                                     action_dim, H, activation, lr))

    def select_action(self, obs, goal):
        actions = []
        for idx, agent in enumerate(self.agents):
            obs_t = torch.FloatTensor(obs[idx].reshape(1, -1)).to(device)
            goal_t = torch.FloatTensor(goal[idx].reshape(1, -1)).to(device)
            actions.append(agent.actor.forward(obs_t, goal_t).detach().cpu().data.numpy().flatten())

        return np.array(actions)

    def update(self, buffer, batch_size):
        losses = {k: {} for k in range(self.n_agents)}
        state, action, reward, next_state, goal, gamma, done = buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        goal = torch.FloatTensor(goal).to(device)
        gamma = torch.FloatTensor(gamma).to(device)
        done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = next_state[:, agent_idx, :]
            new_pi = agent.target_actor.forward(new_states, goal[:, agent_idx, :])
            all_agents_new_actions.append(new_pi)

            mu_states = state[:, agent_idx, :]
            pi = agent.actor.forward(mu_states, goal[:, agent_idx, :])
            all_agents_new_mu_actions.append(pi)

            old_agents_actions.append(action[:, agent_idx, :])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(next_state.reshape((batch_size, -1)),
                                                        new_actions,
                                                        goal.reshape((batch_size, -1))).flatten().detach()
            critic_value = agent.critic.forward(state.reshape((batch_size, -1)),
                                                old_actions,
                                                goal.reshape((batch_size, -1))).flatten()
            target = reward[:, agent_idx] + ((1 - done[:, 0]) * gamma[:, agent_idx] * critic_value_)

            agent.critic.optimizer.zero_grad()
            critic_loss = agent.mseLoss(target, critic_value)
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            agent.actor.optimizer.zero_grad()
            actor_loss = -agent.critic.forward(state.reshape((batch_size, -1)), mu,
                                               goal.reshape((batch_size, -1))).flatten()
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward(retain_graph=True, inputs=list(agent.actor.parameters()))
            agent.actor.optimizer.step()

            losses[agent_idx]["critic"] = critic_loss.cpu().detach().numpy().tolist()
            losses[agent_idx]["actor"] = actor_loss.cpu().detach().numpy().tolist()

        for agent_idx, agent in enumerate(self.agents):
            agent.update_network_parameters()

        return losses

    def save(self, directory, name):
        for idx, agent in enumerate(self.agents):
            agent.save(directory, name, self.level, idx)

    def load(self, directory, name):
        for idx, agent in enumerate(self.agents):
            agent.load(directory, name, self.level, idx)
