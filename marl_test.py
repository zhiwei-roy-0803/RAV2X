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
    "lambdda": 1
    }
config_agent = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "gamma": 0.9,
    "target_net_update_freq": 1,
    "experience_replay_size": 1000000,
    "batch_size": 512,
    "lr": 1e-3,
    "lr_decay_step": 500,
    "lr_decay_gamma": 0.95,
    "lr_last_epoch": -1,
    "n_episode": 1000,
    "n_step_per_episode": 100,
    "epsilon_final": 0.02,
    "epsilon_anneal_length": int(5000*0.75)
}
# build a torch experiment tracer
experiment_name = "Discount_0.9"
experiment_dir = os.path.join(os.getcwd(), "checkpoints", experiment_name)

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
    print("Initializing {:s} Agent {:d}".format(RL_method, ind_agent))
    agent = DQNAgent(config_agent, n_Feature=n_feature, n_Action=n_action)
    agent.predict_net.load_state_dict(torch.load(os.path.join(experiment_dir, "V2V_{:d}.pth".format(ind_agent)), map_location=config_agent["device"])) # load trained agent
    agents.append(agent)

# all optimizer to the global configuration, obtain global configuration and save it in the config.json
config_agent["optimizer"] = agents[0].optimizer
config = dict(config_environment, **config_agent)
#------------------------- Evaluation -----------------------------
record_V2IRate = np.zeros(n_episode)
record_V2VRate = np.zeros(n_episode)
record_V2VSuccessRate = np.zeros(n_episode)
pbar = tqdm(range(n_episode))
for i_episode in pbar:
    # update the vehicle position after each episode
    if i_episode % 1 == 0:
        env.update_vehicle_position() # update vehicle position
        env.update_V2VReceiver() # update the receiver for each V2V link
        env.update_channels_slow() # update channel slow fading
        env.update_channels_fast() # update channel fast fading
    # reset the task buffer for each agent after each episode
    env.demand = env.demand_size * np.ones(numDUE)
    env.individual_time_limit = env.time_slow_fading * np.ones(numDUE)
    env.active_links = np.ones(numDUE, dtype='bool')
    episode_V2I_rate = np.zeros(n_step_per_episode)
    episode_V2V_rate = np.zeros(n_step_per_episode)
    # A episode is a transmission task in 100 ms, V2V agents need to make decision every 1 ms
    for i_step in range(n_step_per_episode):
        state_old_all = []
        action_all = []
        action_all_training = np.zeros([numDUE, 2], dtype='int32')
        for i in range(numDUE):
            state = env.get_state(i, epsilon_final, n_episode/(n_episode-1))
            state_old_all.append(state)
            action = agents[i].get_action(state, epsilon=0.0, learned_policy=False) # use learned policy
            action_all.append(action)
            action_all_training[i, 0] = action % numRB  # chosen RB
            action_all_training[i, 1] = action // numRB # power level
        # All agents take actions simultaneously, obtain shared reward, and update the environment.
        action_temp = action_all_training.copy()
        V2I_rate, V2V_rate = env.compute_rate(action_temp)
        episode_V2I_rate[i_step] = V2I_rate
        episode_V2V_rate[i_step] = V2V_rate
        env.update_channels_fast()
        env.compute_V2V_interference(action_temp)
    # summary the mean V2I rate and the V2V success rate in this 100 ms transmission task
    record_V2IRate[i_episode] = np.mean(episode_V2I_rate)
    record_V2VRate[i_episode] = np.mean(episode_V2V_rate)
    record_V2VSuccessRate[i_episode] = np.sum(env.active_links == False) / numDUE
    pbar.set_description("V2I Rate = {:.3f}, V2V Rate = {:.3f}, V2V Success Rate = {:.3f}".format(record_V2IRate[i_episode],
                                                                                                  record_V2VRate[i_episode],
                                                                                                  record_V2VSuccessRate[i_episode]))
print('Evaluation Done. Evaluation Statistics')
avg_SumV2IRate = np.mean(record_V2IRate)
avg_SumV2VRate = np.mean(record_V2VRate)
avg_V2VSuccessRate = np.mean(record_V2VSuccessRate)
print("Avg V2I Rate = {:.3f} Mbps, Avg Sum V2V Rate = {:.3f} Mbps, Avg V2V Success Rate = {:.6f}".format(
    avg_SumV2IRate,
    avg_SumV2VRate,
    avg_V2VSuccessRate
))


