from __future__ import division, print_function
import scipy.io
import numpy as np
import torch
from RLEnvironment import RLHighWayEnvironment
import os
from DRL.DQNAgent import DQNAgent
from tqdm import tqdm

# RL method applied in simulation
RL_method = "DoubleDQN"
# ################## SETTINGS ######################
config_environment = {
    "powerV2VdB": [23, 10, 5, -100],
    "powerV2I": 23,
    "stdV2V": 3,
    "stdV2I": 8,
    "freq": 2,
    "radius": 500,
    "bsHgt": 25,
    "disBstoHwy": 35,
    "bsAntGaindB": 8,
    "bsNoisedB": 5,
    "vehHgt": 1.5,
    "vehAntGaindB": 3,
    "vehNoisedB": 9,
    "numLane": 6,
    "laneWidth": 4,
    "dB_gamma0": 5,
    "backgroundNoisedB": -114,
    "numDUE": 4,
    "numCUE": 2,
    "time_fast_fading": 0.001,
    "time_slow_fading": 0.1,
    "bandwidth": int(1e6),
    "demand": int((4 * 190 + 300) * 8 * 2),
    "speed": 70
    }
config_agent = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "discount": 0.5,
    "target_net_update_freq": 400,
    "experience_replay_size": 1000000,
    "batch_size": 2000,
    "lr": 0.001
}
# Initialize environment simulator
env = RLHighWayEnvironment(config_environment)
env.init_simulation()  # initialize parameters in env
n_episode = 4000
n_step_per_episode = int(env.time_slow_fading/env.time_fast_fading) # 100
epsilon_final = 0.02
epsilon_anneal_length = 3000
mini_batch_step = 100

numRB = config_environment["numCUE"]
numDUE = config_environment["numDUE"]
n_feature = len(env.get_state(0))
n_action = numRB * len(config_environment["powerV2VdB"])
agents = []
# initialize agents
for ind_agent in range(config_environment["numDUE"]):
    print("Initializing {:s} agent {:d}".format(RL_method, ind_agent))
    agent = DQNAgent(config_agent, n_Feature=n_feature, n_Action=n_action)
    agents.append(agent)
#------------------------- Training -----------------------------
record_reward = np.zeros(n_episode)
record_loss = np.zeros(n_episode)

pbar = tqdm(range(n_episode))
for i_episode in pbar:
    if i_episode < epsilon_anneal_length:
        epsilon = 1 - i_episode * (1 - epsilon_final) / (epsilon_anneal_length - 1)  # epsilon decreases over each episode
    else:
        epsilon = epsilon_final

    # update the vehicle position after each episode
    env.update_vehicle_position() # update vehicle position
    env.update_V2VReceiver() # update the receiver for each V2V link
    env.update_channels_slow() # update channel slow fading
    env.update_channels_fast() # update channel fast fading

    # reset the task buffer for each agent after each episode
    env.demand = env.demand_size * np.ones(numDUE)
    env.individual_time_limit = env.time_slow_fading * np.ones(numDUE)
    env.active_links = np.ones(numDUE, dtype='bool')

    episode_reward = np.zeros(n_step_per_episode)
    episode_loss = []
    # A episode is a transmission task in 100 ms, V2V agents need to make decision every 1 ms
    for i_step in range(n_step_per_episode):
        time_step = i_episode*n_step_per_episode + i_step
        state_old_all = []
        action_all = []
        action_all_training = np.zeros([numDUE, 2], dtype='int32')
        for i in range(numDUE):
            state = env.get_state(i, epsilon, i_episode/(n_episode-1))
            state_old_all.append(state)
            action = agents[i].get_action(state, epsilon, is_static_policy=False)
            action_all.append(action)
            action_all_training[i, 0] = action % numRB  # chosen RB
            action_all_training[i, 1] = action // numRB # power level
        # All agents take actions simultaneously, obtain shared reward, and update the environment.
        action_temp = action_all_training.copy()
        train_reward = env.compute_reward(action_temp)
        episode_reward[i_step] = train_reward
        env.update_channels_fast()
        env.compute_V2V_interference(action_temp)
        for i in range(numDUE):
            state_old = state_old_all[i]
            action = action_all[i]
            state_new = env.get_state(i, epsilon, i_episode/(n_episode-1))
            agents[i].memory.add(state_old, state_new, train_reward, action)  # add entry to this agent's memory

    # training agents after finishing one episode
    for i in range(numDUE):
        loss_val_batch = agents[i].update_dqn(time_step)
        episode_loss.append(loss_val_batch)

    record_reward[i_episode] = np.mean(episode_reward)
    record_loss[i_episode] = np.mean(episode_loss)
    pbar.set_description("loss = {:f}, reward = {:f}".format(record_loss[i_episode], record_reward[i_episode]))

print('Training Done. Saving models and training statistics')
model_dir = os.path.join(os.getcwd(), "model", "marl", RL_method)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
for i in range(env.numDUE):
    agent = agents[i]
    torch.save(agent.predict_net.state_dict(), os.path.join(model_dir, "V2V_{:d}.pth".format(i)))

reward_dir = os.path.join(os.getcwd(), "reward", "marl", RL_method)
if not os.path.isdir(reward_dir):
    os.makedirs(reward_dir)
reward_path = os.path.join(reward_dir, "V2V_{:d}.mat".format(env.numDUE))
scipy.io.savemat(reward_path, {'reward': record_reward})

loss_dir = os.path.join(os.getcwd(), "loss", "marl", RL_method)
if not os.path.isdir(loss_dir):
    os.makedirs(loss_dir)
record_loss = np.asarray(record_loss)
loss_path = os.path.join(loss_dir, "V2V_{:d}.mat".format(env.numDUE))
scipy.io.savemat(loss_path, {'train_loss': record_loss})





