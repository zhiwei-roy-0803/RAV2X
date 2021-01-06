from __future__ import division, print_function
import numpy as np
import torch
from RLEnvironment import RLHighWayEnvironment
from DRL.DQNAgent import DQNAgent
from torchtracer import Tracer
from torchtracer.data import Config
from torchtracer.data import Model
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
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
    "numCUE": 4,
    "time_fast_fading": 0.001,
    "time_slow_fading": 0.1,
    "bandwidth": int(1e6),
    "demand": int((4 * 190 + 300) * 8 * 2),
    "speed": 70,
    "lambdda": 0.5
    }
config_agent = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "gamma": 0.9,
    "target_net_update_freq": 1,
    "experience_replay_size": 1000000,
    "batch_size": 2048,
    "lr": 1e-3,
    "lr_decay_step": 100,
    "lr_decay_gamma": 0.95,
    "lr_last_epoch": -1,
    "n_episode": 4000,
    "n_step_per_episode": 100,
    "epsilon_final": 0.02,
    "epsilon_anneal_length": 3000
}
# build a torch experiment tracer
experiment_name = "Discount_0.9"
if os.path.isdir(os.path.join(os.getcwd(), "checkpoints", experiment_name)):
    shutil.rmtree(os.path.join(os.getcwd(), "checkpoints", experiment_name))
tracer = Tracer('checkpoints').attach(experiment_name)

# Initialize environment simulator
env = RLHighWayEnvironment(config_environment)
env.init_simulation()  # initialize parameters in env
n_episode = config_agent["n_episode"]
n_step_per_episode = config_agent["n_step_per_episode"]
epsilon_final = config_agent["epsilon_final"]
epsilon_anneal_length = config_agent["epsilon_anneal_length"]

# initialize agents
numRB = config_environment["numCUE"]
numDUE = config_environment["numDUE"]
n_feature = len(env.get_state(0))
n_action = numRB * len(config_environment["powerV2VdB"])
agents = []
for ind_agent in range(config_environment["numDUE"]):
    print("Initializing {:s} agent {:d}".format(RL_method, ind_agent))
    agent = DQNAgent(config_agent, n_Feature=n_feature, n_Action=n_action)
    agents.append(agent)

# all optimizer to the global configuration, obtain global configuration and save it in the config.json
config_agent["optimizer"] = agents[0].optimizer
config = dict(config_environment, **config_agent)
tracer.store(Config(config))
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
    if i_episode % 1 == 0:
        env.update_vehicle_position() # update vehicle position
        env.update_V2VReceiver() # update the receiver for each V2V link
        env.update_channels_slow() # update channel slow fading
        env.update_channels_fast() # update channel fast fading
    # env.init_simulation()
    # reset the task buffer for each agent after each episode
    env.demand = env.demand_size * np.ones(numDUE)
    env.individual_time_limit = env.time_slow_fading * np.ones(numDUE)
    env.active_links = np.ones(numDUE, dtype='bool')
    episode_reward = np.zeros(n_step_per_episode)
    episode_loss = []
    # A episode is a transmission task in 100 ms, V2V agents need to make decision every 1 ms
    for i_step in range(n_step_per_episode):
        state_old_all = []
        action_all = []
        action_all_training = np.zeros([numDUE, 2], dtype='int32')
        for i in range(numDUE):
            state = env.get_state(i, epsilon, i_episode/(n_episode-1))
            state_old_all.append(state)
            action = agents[i].get_action(state, epsilon, learned_policy=True)
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
        loss_val_batch = agents[i].update_double_dqn(i_episode)
        episode_loss.append(loss_val_batch)

    record_reward[i_episode] = np.mean(episode_reward)
    record_loss[i_episode] = np.mean(episode_loss)
    tracer.log(msg="Episode #{:04d}, TD Loss : {:.3f}".format(i_episode, record_loss[i_episode]), file="loss")
    tracer.log(msg="Episode #{:04d}, Reward : {:.3f}".format(i_episode, record_reward[i_episode]), file="reward")
    pbar.set_description("Loss = {:.3f}, Reward = {:.3f}".format(record_loss[i_episode], record_reward[i_episode]))

print('Training Done. Saving models and training statistics')
for i in range(env.numDUE):
    agent = agents[i]
    tracer.store(Model(agent.predict_net), file="V2V_{:d}".format(i))

# Plot training loss and reward curve
plt.figure(dpi=300)
plt.plot(np.arange(1, n_episode + 1), record_loss, color='r', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("TD Error (Loss)")
plt.grid()
tracer.store(plt.gcf(), "loss.png")

plt.figure(dpi=300)
plt.plot(np.arange(1, n_episode + 1), record_reward, color='r', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
tracer.store(plt.gcf(), "reward.png")