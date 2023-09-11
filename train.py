import torch
import os
import time
import datetime
import numpy as np
from HAC import HAC
from smac.env import StarCraft2Env
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train():
    env_name = "2m_vs_2zg_IM"
    env = StarCraft2Env(map_name=env_name, step_mul=4, state_last_action=False,
                        obs_pathing_grid=True, obs_terrain_height=True)
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]
    obs_dim = env_info["obs_shape"]
    # state_dim = env_info["state_shape"]
    action_dim = env_info["n_actions"]

    # 2v2 attack -- full sized state
    # setup_name = "2v2_attack_fullState"
    # state_bits = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]
    # goal_state = np.array([1., -0.12053572, -0.10940988,
    #                        1., -0.12053572, -0.08482143,
    #                        0., 0., 0.,
    #                        0., 0., 0.], dtype=np.float32)
    # threshold = np.array([0.2, 0.1, 0.1,
    #                       0.2, 0.1, 0.1,
    #                       0.005, 0.005, 0.005,
    #                       0.005, 0.005, 0.005], dtype=np.float32)
    # state_clip_low = np.array([0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5], dtype=np.float32)
    # state_clip_high = np.array([1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5], dtype=np.float32)

    # 2v2 hide -- full sized state
    setup_name = "2v2_hide_fullState"
    state_bits = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]
    goal_state = np.array([1., -0.35055977, -0.4497768,
                           1., -0.3506208, -0.40596226,
                           0., 0., 0.,
                           0., 0., 0.], dtype=np.float32)
    threshold = np.array([0.2, 0.1, 0.1,
                          0.2, 0.1, 0.1,
                          0.005, 0.005, 0.005,
                          0.005, 0.005, 0.005], dtype=np.float32)
    state_clip_low = np.array([0., -0.5, -0.5,
                               0., -0.5, -0.5,
                               0., -0.5, -0.5,
                               0., -0.5, -0.5], dtype=np.float32)
    state_clip_high = np.array([1., 0.5, 0.5,
                                1., 0.5, 0.5,
                                1., 0.5, 0.5,
                                1., 0.5, 0.5], dtype=np.float32)

    # 2v10 -- coord and hp state
    # setup_name = "2v10_hide_coord"
    # state_bits = [0, 2, 3, 4, 6, 7]
    # goal_state = np.array([1., -0.35055977, -0.4497768,
    #                        1., -0.3506208, -0.40596226], dtype=np.float32)
    # threshold = np.array([0.2, 0.1, 0.1,
    #                       0.2, 0.1, 0.1], dtype=np.float32)
    # state_clip_low = np.array([0., -0.5, -0.5,
    #                            0., -0.5, -0.5], dtype=np.float32)
    # state_clip_high = np.array([1., 0.5, 0.5,
    #                             1., 0.5, 0.5], dtype=np.float32)

    # 2v10 -- full sized state
    # setup_name = "2v10_hide_fullState"
    # state_bits = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    #               17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #               34, 35, 36, 37]
    # goal_state = np.array([1., -0.35055977, -0.4497768,
    #                        1., -0.3506208, -0.40596226,
    #                        0., 0., 0.,
    #                        0., 0., 0.,
    #                        0., 0., 0.,
    #                        0., 0., 0.,
    #                        0., 0., 0.,
    #                        0., 0., 0.,
    #                        0., 0., 0.,
    #                        0., 0., 0.,
    #                        0., 0., 0.,
    #                        0., 0., 0.], dtype=np.float32)
    # threshold = np.array([0.2, 0.1, 0.1,
    #                       0.2, 0.1, 0.1,
    #                       0.005, 0.005, 0.005,
    #                       0.005, 0.005, 0.005,
    #                       0.005, 0.005, 0.005,
    #                       0.005, 0.005, 0.005,
    #                       0.005, 0.005, 0.005,
    #                       0.005, 0.005, 0.005,
    #                       0.005, 0.005, 0.005,
    #                       0.005, 0.005, 0.005,
    #                       0.005, 0.005, 0.005,
    #                       0.005, 0.005, 0.005], dtype=np.float32)
    # state_clip_low = np.array([0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5,
    #                            0., -0.5, -0.5
    #                            ], dtype=np.float32)
    # state_clip_high = np.array([1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5,
    #                             1., 0.5, 0.5], dtype=np.float32)

    goal_state_dim = len(state_bits)
    action_clip_low = np.zeros((n_agents, action_dim))
    action_clip_high = np.ones((n_agents, action_dim))

    save_episode = 100  # keep saving every n episodes
    max_episodes = 10_000  # max num of training episodes
    random_seed = 42

    # HAC parameters:
    k_level = 2  # num of levels in hierarchy
    H = 20  # time horizon to achieve subgoal, default 20
    lamda = 0.3  # subgoal testing parameter, default 0.3

    # DDPG parameters:
    gamma = 0.99  # discount factor for future rewards
    batch_size = 1024  # num of transitions sampled from replay buffer
    lr = 0.0005

    learn_every = 20
    show_log_every = 100
    agent_rewards = []
    agent_timesteps = []

    noise_rate_goal = 0.2
    noise_rate_action = 0.8
    noise_rate_min_goal = 0.001
    noise_rate_min_action = 0.01
    noise_decay_rate_goal = noise_rate_goal / max_episodes
    noise_decay_rate_action = noise_rate_action / max_episodes

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        # env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # creating HAC agent and setting parameters
    agent = HAC(k_level, n_agents, H, obs_dim, goal_state_dim, action_dim, threshold, lr,
                lamda, gamma, action_clip_low, action_clip_high, state_clip_low, state_clip_high)
    agent.noise_rate_goal = noise_rate_goal
    agent.noise_rate_action = noise_rate_action

    # logging file:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = env_name + timestamp
    log_folder = 'logs'
    log_filename = experiment_name + "_log.txt"
    os.makedirs(log_folder, exist_ok=True)
    log_f = open(os.path.join(log_folder, log_filename), "w+")

    # save trained models
    directory = "./preTrained/{}/{}level/".format(env_name, k_level)
    filename = "HAC_{}".format(experiment_name)

    # tensorboard
    tb_log_name_parts = [
        setup_name,
        f"goalStateDim_{goal_state_dim}",
        f"kLevel_{k_level}",
        f"H_{H}",
        f"gamma_{gamma}".replace(".", ""),
        f"bs_{batch_size}",
        f"lr_{lr}".replace(".", ""),
        f"lamda_{lamda}".replace(".", ""),
        f"learnEvery_{learn_every}",
        env_name,
        timestamp
    ]
    tb_log_dir = "tb_logs/" + "_".join(tb_log_name_parts)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)

    # training procedure
    train_start = time.time()
    n_eps_start = time.time()
    for i_episode in range(1, max_episodes + 1):
        agent.reward = 0
        agent.timestep = 0
        agent.goal_rewards = {k: [] for k in range(k_level)}
        agent.noise_rate_goal = max(noise_rate_min_goal, agent.noise_rate_goal - noise_decay_rate_goal)
        agent.noise_rate_action = max(noise_rate_min_action, agent.noise_rate_action - noise_decay_rate_action)

        env.reset()
        obs = env.get_obs()
        state = env.get_state()[state_bits]
        done = False

        while not done:
            # collecting experience in environment
            last_obs, last_state, done = agent.run_HAC(env, k_level - 1, obs, state,
                                                       np.repeat(goal_state[None, ...], n_agents, axis=0),
                                                       False, state_bits)

            if np.all(agent.check_goal(last_state, np.repeat(goal_state[None, ...], n_agents, axis=0), threshold)):
                print("################ Solved! ################ ")
                name = filename + '_solved'
                agent.save(directory, name)
                done = True

            obs = last_obs
            state = last_state

        # update all levels
        if not i_episode % learn_every:
            losses = agent.update(batch_size)

            for level, l_vals in losses.items():
                for ag_idx, agent_losses in l_vals.items():
                    for loss_type, loss in agent_losses.items():
                        tb_tag = f"0_losses/L{level}_{loss_type}_ag{ag_idx}"
                        writer.add_scalar(tb_tag, loss, i_episode)

        # logging updates:
        log_f.write('{},{}\n'.format(i_episode, agent.reward))
        log_f.flush()

        # tb logs update
        for level, values in agent.goal_rewards.items():
            goal_rewards_mean = np.mean(values, axis=0)
            goal_rewards_std = np.std(values, axis=0)
            for ag_idx in range(n_agents):
                writer.add_scalar(f"1_goal_rewards/mean_R_goal_L{level}_ag{ag_idx}", goal_rewards_mean[ag_idx],
                                  i_episode)
                writer.add_scalar(f"1_goal_rewards/std_R_goal_L{level}_ag{ag_idx}", goal_rewards_std[ag_idx], i_episode)

        writer.add_scalar("2_env_rewards/R_ex", agent.reward, i_episode)
        writer.add_scalar("stats/noise_rate_goal", agent.noise_rate_goal, i_episode)
        writer.add_scalar("stats/noise_rate_action", agent.noise_rate_action, i_episode)
        writer.add_scalar("stats/episode_steps", env._episode_steps, i_episode)

        if i_episode % save_episode == 0:
            agent.save(directory, filename)

        agent_rewards.append(agent.reward)
        agent_timesteps.append(env._episode_steps)
        if not i_episode % show_log_every:
            n_eps_time = time.time() - n_eps_start
            n_eps_start = time.time()
            print(
                "Ep: {}\t Steps: {:.2f}\t Reward: {:.4f}\t Noise: {:.4f}/{:.4f}\t Time: {:.1f} / {:.1f} ({:.1f}m)".format(
                    i_episode,
                    np.mean(agent_timesteps[-show_log_every:]),
                    np.mean(agent_rewards[-show_log_every:]),
                    agent.noise_rate_goal,
                    agent.noise_rate_action,
                    n_eps_time,
                    time.time() - train_start,
                    (time.time() - train_start) / 60
                ))
            writer.add_scalar("stats/n_eps_time", n_eps_time, i_episode)
            writer.add_scalar("stats/total_time_min", (time.time() - train_start) / 60, i_episode)


if __name__ == '__main__':
    train()
