import numpy as np
from numpy.random import normal
from GenCUEandDUE import VehicleGenerator
from HighwayChannel import HwyChannelLargeScaleFadingGenerator

class RLHighWayEnvironment():
    '''
    This class implement the simulator environment for reinforcement learning in the Highway scenario
    '''
    def __init__(self, config):
        '''
        Construction methods for class RLHighWayEnvironment
        :param config: dict containing key parameters for simulation
        '''
        # transceiver configuration
        self.power_list_V2V_dB = config["powerV2VdB"]
        self.power_V2I_dB = config["powerV2I"]
        self.sig2_dB = config["backgroundNoisedB"]
        self.bsAntGaindB = config["bsAntGaindB"]
        self.bsNoisedB = config["bsNoisedB"]
        self.vehAntGaindB = config["vehAntGaindB"]
        self.vehNoisedB = config["vehNoisedB"]
        self.sig2 = 10**(self.sig2_dB/10)
        self.stdV2V = config["stdV2V"]
        self.stdV2I = config["stdV2I"]
        # agent configuration
        self.numCUE = config["numCUE"]
        self.numDUE = config["numDUE"]
        self.numChannel = self.numCUE
        # protocol configuration
        self.freq = config["freq"]
        self.radius = config["radius"]
        self.numLane = config["numLane"]
        self.laneWidth = config["laneWidth"]
        self.disBstoHwy = config["disBstoHwy"]
        self.bsHgt = config["bsHgt"]
        self.vehHgt = config["vehHgt"]
        self.d0 = np.sqrt(self.radius**2 - self.disBstoHwy**2) # half length for the highway
        self.time_fast_fading = config["time_fast_fading"]
        self.time_slow_fading = config["time_slow_fading"]
        self.bandwidth = config["bandwidth"]
        self.demand_size = config["demand"]
        self.speed = config["speed"]
        # initialize vehicle and channel sampler
        self.vehicle_generator = VehicleGenerator(self.d0, self.laneWidth, self.numLane, self.disBstoHwy)
        self.hwy_large_scale_channel_sampler = HwyChannelLargeScaleFadingGenerator(self.stdV2I, self.stdV2V, self.vehHgt,
                                                                                   self.bsHgt, self.freq, self.vehAntGaindB,
                                                                                   self.bsAntGaindB,self.bsNoisedB,
                                                                                   self.vehNoisedB)
        # internal simulation parameters
        self.active_links = np.ones(self.numDUE, dtype="bool")
        self.individual_time_limit = self.time_slow_fading * np.ones(self.numDUE)
        self.demand = self.demand_size * np.ones(self.numDUE)
        self.lambdda = config["lambdda"]
        self.V2V_interference = np.zeros((self.numDUE, self.numDUE, self.numCUE))


    def init_simulation(self):
        '''
        Initialize Highway Environment simulator
        :return:
        '''
        self.generate_vehicles()
        self.update_channels_slow()
        self.update_channels_fast()
        self.active_links = np.ones(self.numDUE, dtype="bool")
        self.individual_time_limit = self.time_slow_fading * np.ones(self.numDUE)
        self.demand = self.demand_size * np.ones(self.numDUE)

    def generate_vehicles(self):
        d_avg  = 2.5 * self.speed / 3.6 # vehicle density
        while 1:
            vehPos = []
            vehDir = []
            # 每个车道产生根据车流密度利用柏松分布产生一个随机数确定这个车道有多少车，然后利用均匀分布在这个车道上进行实际的撒点
            for ilane in range(self.numLane):
                nveh = np.random.poisson(lam=2*self.d0/d_avg)
                posi_ilane = np.zeros((nveh, 2))
                posi_ilane[:, 0] = (2*np.random.rand(nveh) - 1) * self.d0
                posi_ilane[:, 1] = (self.disBstoHwy + ilane * self.laneWidth) * np.ones(nveh)
                vehPos.append(posi_ilane)
                if ilane < self.numLane//2:
                    vehDir.append([0]*nveh) # left -> right
                else:
                    vehDir.append([1]*nveh) # right -> left
            vehPos = np.concatenate(vehPos)
            vehDir = np.concatenate(vehDir)
            numVeh = vehPos.shape[0]
            if numVeh > self.numCUE + self.numDUE * 2:
                break
        # 从所有的汽车中随机选择一些车辆出来作为D2D通信的发起者和接收方，每个D2D通信的发起者的接收机都是距离它最近的那个
        indPerm = np.random.permutation(numVeh)
        indDUETransmitter = indPerm[:self.numDUE]
        indDUEReceiver = -np.ones(self.numDUE, dtype=np.int)
        for i in range(self.numDUE):
            minDist = np.inf
            tmpInd = 0
            for j in range(numVeh):
                if j in indDUETransmitter or j in indDUEReceiver:
                    continue
                newDist = np.sqrt(np.sum((vehPos[indDUETransmitter[i]] - vehPos[j])**2))
                if newDist < minDist:
                    tmpInd = j
                    minDist = newDist
            indDUEReceiver[i] = tmpInd
        # 从剩下的车里面随机选择一些作为CUE
        cntCUE = self.numDUE + 1
        indCUE = []
        while cntCUE <= numVeh:
            if indPerm[cntCUE] not in indDUEReceiver:
                indCUE.append(indPerm[cntCUE])
            cntCUE += 1
            if len(indCUE) >= self.numCUE:
                break
        self.V2I = np.array(indCUE)
        self.V2VTransmitter = indDUETransmitter
        self.V2VReceiver = indDUEReceiver
        self.vehPos = vehPos
        self.vehDir = vehDir

    def update_vehicle_position(self):
        '''
        Update the position of each vehicle according to their current position, direction and speed
        :return:
        '''
        factor = 1000/3600
        for veh_idx in range(len(self.vehPos)):
            cur_posi = self.vehPos[veh_idx]
            if self.vehDir[veh_idx] == 0: # left -> right
                tmp = cur_posi[0] + self.speed * factor * self.time_slow_fading
                if tmp > self.d0:
                    self.vehPos[veh_idx][0] = tmp - 2*self.d0
                else:
                    self.vehPos[veh_idx][0] = tmp
            else:
                tmp = cur_posi[0] - self.speed * factor * self.time_slow_fading
                if tmp < -self.d0:
                    self.vehPos[veh_idx][0] = tmp + 2 * self.d0
                else:
                    self.vehPos[veh_idx][0] = tmp


    def update_V2VReceiver(self):
        '''
        Update the V2V receiver according the updated position of each vehicle
        :return:
        '''
        numVeh = len(self.vehPos)
        self.V2VReceiver = -np.ones(self.numDUE, dtype=np.int)
        for i in range(self.numDUE):
            minDist = np.inf
            tmpInd = 0
            for j in range(numVeh):
                if j in self.V2VTransmitter or j in self.V2VReceiver:
                    continue
                newDist = np.sqrt(np.sum((self.vehPos[self.V2VTransmitter[i]] - self.vehPos[j])**2))
                if newDist < minDist:
                    tmpInd = j
                    minDist = newDist
            self.V2VReceiver[i] = tmpInd


    def update_channels_slow(self):
        '''
        Compute larger scale fading for five channels:
        V2I_channel_dB: channel for V2I->BS (M * 1)
        V2V_channel_dB: channel between V2V transceiver (K * 1)
        V2V_V2I_interference_channel_dB: channel between V2V transmitter and BS receiver
        V2I_V2V_interference_channel_dB: channel between V2I transmitter and V2V receiver
        V2V_V2V_interference_channel_dB: channel of V2V transmitter and receiver between different V2V pairs
        M : number of CUE, K: number of DUE
        :return:
        '''
        self.V2I_channel_dB = np.zeros(self.numCUE)
        self.V2V_channel_dB = np.zeros(self.numDUE)
        self.V2V_V2I_interference_channel_dB = np.zeros(self.numDUE)
        self.V2I_V2V_interference_channel_dB = np.zeros((self.numCUE, self.numDUE))
        self.V2V_V2V_interference_channel_dB = np.zeros((self.numDUE, self.numDUE))
        for m in range(self.numCUE):
            # 计算第m个CUE对基站的距离和路损，假设基站的坐标是（0，0）
            dist_mB = np.sqrt(self.vehPos[self.V2I[m], 0] ** 2 + self.vehPos[self.V2I[m], 1] ** 2)  # m-th CUE到基站的距离
            self.V2I_channel_dB[m] = self.hwy_large_scale_channel_sampler.generate_fading_V2I(dist_mB)
            # 计算第m个CUE和第k个DUE之间的距离、路损
            for k in range(self.numDUE):
                dist_mk = np.sqrt(np.sum((self.vehPos[self.V2I[m]] - self.vehPos[self.V2VReceiver[k]]) ** 2))
                self.V2I_V2V_interference_channel_dB[m, k] = self.hwy_large_scale_channel_sampler.generate_fading_V2V(dist_mk)

        for k in range(self.numDUE):
            # 计算第K对DUE之间的距离和路损
            dist_k = np.sqrt(np.sum((self.vehPos[self.V2VTransmitter[k]] - self.vehPos[self.V2VReceiver[k]]) ** 2))
            self.V2V_channel_dB[k] = self.hwy_large_scale_channel_sampler.generate_fading_V2V(dist_k)
            # 计算第k对DUE的发射机对基站的干扰
            dist_kB = np.sqrt(self.vehPos[self.V2VTransmitter[k], 0]**2 + self.vehPos[self.V2VTransmitter[k], 1]**2)  # k-th DUE发射机到基站的距离
            self.V2V_V2I_interference_channel_dB[k] = self.hwy_large_scale_channel_sampler.generate_fading_V2I(dist_kB)
            for j in range(self.numDUE):
                if j == k:
                    continue
                dist_kj = np.sqrt(np.sum((self.vehPos[self.V2VTransmitter[k]] - self.vehPos[self.V2VReceiver[j]])**2))
                self.V2V_V2V_interference_channel_dB[k, j]  =  self.hwy_large_scale_channel_sampler.generate_fading_V2V(dist_kj)


    def update_channels_fast(self):
        '''
        Update fasting fading component for four kinds of channels
        :return:
        '''
        V2I_channel_fast_dB = np.repeat(self.V2I_channel_dB[:, np.newaxis], self.numChannel, axis=1)
        fast_componet = np.abs(normal(0, 1, V2I_channel_fast_dB.shape) + 1j * normal(0, 1, V2I_channel_fast_dB.shape)) / np.sqrt(2)
        self.V2I_channel_with_fast_dB = V2I_channel_fast_dB + 20*np.log10(fast_componet)

        V2V_channel_fast_dB = np.repeat(self.V2V_channel_dB[:, np.newaxis], self.numChannel, axis=1)
        fast_componet = np.abs(normal(0, 1, V2V_channel_fast_dB.shape) + 1j * normal(0, 1, V2V_channel_fast_dB.shape)) / np.sqrt(2)
        self.V2V_channel_with_fast_dB = V2V_channel_fast_dB + 20 * np.log10(fast_componet)

        V2I_V2V_interference_channel_with_fast_dB = np.repeat(self.V2I_V2V_interference_channel_dB[:, :, np.newaxis], self.numChannel, axis=2)
        fast_componet = np.abs(normal(0, 1, V2I_V2V_interference_channel_with_fast_dB.shape) + 1j * normal(0, 1, V2I_V2V_interference_channel_with_fast_dB.shape)) / np.sqrt(2)
        self.V2I_V2V_interference_channel_with_fast_dB = V2I_V2V_interference_channel_with_fast_dB + 20*np.log10(fast_componet)

        V2V_V2V_interference_channel_with_fast_dB = np.repeat(self.V2V_V2V_interference_channel_dB[:, :, np.newaxis],self.numChannel, axis=2)
        fast_componet = np.abs(normal(0, 1, V2V_V2V_interference_channel_with_fast_dB.shape) + 1j * normal(0, 1, V2V_V2V_interference_channel_with_fast_dB.shape)) / np.sqrt(2)
        self.V2V_V2V_interference_channel_with_fast_dB = V2V_V2V_interference_channel_with_fast_dB + 20 * np.log10(fast_componet)

        V2V_V2I_interference_channel_with_fast_dB = np.repeat(self.V2V_V2I_interference_channel_dB[:, np.newaxis], self.numChannel, axis=1)
        fast_componet = np.abs(normal(0, 1, V2V_V2I_interference_channel_with_fast_dB.shape) + 1j * normal(0, 1, V2V_V2I_interference_channel_with_fast_dB.shape)) / np.sqrt(2)
        self.V2V_V2I_interference_channel_with_fast_dB = V2V_V2I_interference_channel_with_fast_dB + 20 * np.log10(fast_componet)


    def get_state(self, idx, epsilon=0.02, ind_episode=1.):
        # mu_fast_fading = -2.5126976697433587
        # std_fast_fading = 5.591597454173695
        V2I_fast = self.V2I_channel_with_fast_dB - self.V2I_channel_dB
        V2V_fast = self.V2V_V2V_interference_channel_with_fast_dB[:, idx, :] - self.V2V_V2V_interference_channel_dB[:, idx].reshape((self.numDUE, 1))
        V2I_fast = (V2I_fast + 10) / 35 # normalize the fast fading component of V2I links
        V2V_fast = (V2V_fast + 10) / 35 # normalize the fast fading component of V2V links

        V2V_interference = self.V2V_interference[idx, idx, :]
        V2V_interference = (V2V_interference - 60)/60

        V2I_slow = self.V2I_channel_dB
        V2V_slow = self.V2V_V2V_interference_channel_dB[:, idx]
        V2I_slow = (V2I_slow - 60) / 60
        V2V_slow = (V2V_slow - 60) / 60

        load_remaining = [self.demand[idx]/self.demand_size]
        time_remaining = [self.individual_time_limit[idx]/self.time_slow_fading]
        # assemble the state vector
        state = np.concatenate((V2I_slow, V2V_slow, np.reshape(V2I_fast, -1), np.reshape(V2V_fast, -1), V2V_interference,
                                time_remaining, load_remaining, np.asarray([epsilon, ind_episode])))

        return state

    def compute_V2V_interference(self, action):
        '''
        Compute interference power for each V2V link
        :param action:
        :return:
        '''
        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # interference from V2I link
        V2V_interference = np.zeros((self.numDUE, self.numDUE, self.numCUE)) + self.sig2
        for i in range(self.numDUE):
            # if current V2V link has finished its transmission, it then turns to silent mode
            if not self.active_links[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            V2I_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[RB_i, i, RB_i] + 2*self.vehAntGaindB + self.vehNoisedB
            V2V_interference[i, i, RB_i] += 10**(V2I_power_dB/10)
            # interference from other V2V link with the same RB
            for j in range(i + 1, self.numDUE):
                RB_j = RB_selection[j]
                power_dB_j = power_selection[j]
                if RB_j == RB_i:
                    power_i2j = self.power_list_V2V_dB[power_dB_i] - self.V2V_V2V_interference_channel_with_fast_dB[i, j, RB_i] + 2*self.vehAntGaindB + self.vehNoisedB
                    V2V_interference[i, j, RB_i] += 10 ** (power_i2j / 10)
                    power_j2i = self.power_list_V2V_dB[power_dB_j] - self.V2V_V2V_interference_channel_with_fast_dB[j, i, RB_j] + 2*self.vehAntGaindB + self.vehNoisedB
                    V2V_interference[j, i, RB_j] += 10 ** (power_j2i / 10)
        # transform inference from linear scale to dB
        self.V2V_interference = 10 * np.log10(V2V_interference)

    def compute_reward(self, action):
        '''
        Compute reward given current channel realization and the action taken by the agents
        :param action:
        :param V2I_channel_with_fast_dB:
        :param V2V_channel_with_fast_dB:
        :param V2I_inter_channel_with_fast_dB:
        :param V2V_inter_channel_with_fast_dB:
        :return:
        '''
        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        reward_scaling_factor = 0.1  # RL trick for scaling reward signal to accelerate convergence

        # compute V2I rate
        V2I_interference = np.zeros(self.numCUE)
        for i in range(self.numDUE):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links[i]:
                continue
            interference = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2I_interference_channel_with_fast_dB[i, RB_i] + self.vehAntGaindB  + self.bsAntGaindB  + self.bsNoisedB
            V2I_interference[RB_i] += 10**(interference/10)
        V2I_interference += self.sig2
        V2I_power_dB = self.power_V2I_dB + self.V2I_channel_with_fast_dB.diagonal() + self.vehAntGaindB + self.bsAntGaindB + self.bsNoisedB
        V2I_power = 10**(V2I_power_dB/10)
        V2I_rate = np.log2(1 + np.divide(V2I_power, V2I_interference))

        # compute V2V rate
        V2V_interference = np.zeros(self.numDUE)
        V2V_signal = np.zeros(self.numDUE)
        for i in range(self.numDUE):
            if not self.active_links[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            # compute receiver signal strength for current V2V link
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.V2V_channel_with_fast_dB[i, RB_i] + 2*self.vehAntGaindB + self.vehNoisedB
            V2V_signal[i] = 10**(receiver_power_dB/10)
            # compute V2I link interference to current V2V link
            V2I_interference_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[RB_i, i, RB_i] + 2*self.vehAntGaindB + self.vehNoisedB
            V2V_interference[i] = 10**(V2I_interference_power_dB/10)
            for k in range(i + 1, self.numDUE):
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_i2k = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2V_interference_channel_with_fast_dB[i, k, RB_i] + 2*self.vehAntGaindB + self.vehNoisedB
                    V2V_interference[i] += 10**(power_i2k/10)
                    power_k2i = self.power_list_V2V_dB[power_dB_k] + self.V2V_V2V_interference_channel_with_fast_dB[k, i, RB_k] + 2*self.vehAntGaindB + self.vehNoisedB
                    V2V_interference[k] += 10**(power_k2i/10)
        V2V_interference += self.sig2
        V2V_rate = np.log2(1 + np.divide(V2V_signal, V2V_interference))
        V2V_rate[self.active_links==0] = 0 # if a link is in silent mode, it does not transmit any data

        # update demand and time left for each V2V link
        self.demand -= V2V_rate * self.time_fast_fading * self.bandwidth
        self.demand[self.demand < 0] = 0
        self.individual_time_limit -= self.individual_time_limit
        self.active_links[self.demand <= 0] = 0

        # compute reward signal for V2I and V2V links
        reward_V2V = V2V_rate
        reward_V2V[self.demand <= 0] = 1
        reward_V2V = np.sum(reward_V2V)
        reward_V2I = np.sum(V2I_rate)

        # compute combined reward
        reward = (self.lambdda * reward_V2I + (1 - self.lambdda) * np.sum(reward_V2V)) * reward_scaling_factor
        return reward

    def compute_rate(self, action):
        RB_selection = action[:, 0]
        power_selection = action[:, 1]
        # compute V2I rate
        V2I_interference = np.zeros(self.numCUE)
        for i in range(self.numDUE):
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            if not self.active_links[i]:
                continue
            interference = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2I_interference_channel_with_fast_dB[
                i, RB_i] + self.vehAntGaindB + self.bsAntGaindB + self.bsNoisedB
            V2I_interference[RB_i] += 10 ** (interference / 10)
        V2I_interference += self.sig2
        V2I_power_dB = self.power_V2I_dB + self.V2I_channel_with_fast_dB.diagonal() + self.vehAntGaindB + self.bsAntGaindB + self.bsNoisedB
        V2I_power = 10 ** (V2I_power_dB / 10)
        V2I_rate = np.log2(1 + np.divide(V2I_power, V2I_interference))

        # compute V2V rate
        V2V_interference = np.zeros(self.numDUE)
        V2V_signal = np.zeros(self.numDUE)
        for i in range(self.numDUE):
            if not self.active_links[i]:
                continue
            RB_i = RB_selection[i]
            power_dB_i = power_selection[i]
            # compute receiver signal strength for current V2V link
            receiver_power_dB = self.power_list_V2V_dB[power_dB_i] + self.V2V_channel_with_fast_dB[i, RB_i] + 2 * self.vehAntGaindB + self.vehNoisedB
            V2V_signal[i] = 10 ** (receiver_power_dB / 10)
            # compute V2I link interference to current V2V link
            V2I_interference_power_dB = self.power_V2I_dB + self.V2I_V2V_interference_channel_with_fast_dB[
                RB_i, i, RB_i] + 2 * self.vehAntGaindB + self.vehNoisedB
            V2V_interference[i] = 10 ** (V2I_interference_power_dB / 10)
            for k in range(i + 1, self.numDUE):
                RB_k = RB_selection[k]
                power_dB_k = power_selection[k]
                if RB_k == RB_i:
                    power_i2k = self.power_list_V2V_dB[power_dB_i] + self.V2V_V2V_interference_channel_with_fast_dB[
                        i, k, RB_i] + 2 * self.vehAntGaindB + self.vehNoisedB
                    V2V_interference[i] += 10 ** (power_i2k / 10)
                    power_k2i = self.power_list_V2V_dB[power_dB_k] + self.V2V_V2V_interference_channel_with_fast_dB[
                        k, i, RB_k] + 2 * self.vehAntGaindB + self.vehNoisedB
                    V2V_interference[k] += 10 ** (power_k2i / 10)
        V2V_interference += self.sig2
        V2V_rate = np.log2(1 + np.divide(V2V_signal, V2V_interference))
        V2V_rate[self.active_links == 0] = 0  # if a link is in silent mode, it does not transmit any data

        # update demand and time left for each V2V link
        self.demand -= V2V_rate * self.time_fast_fading * self.bandwidth
        self.demand[self.demand < 0] = 0
        self.individual_time_limit -= self.individual_time_limit
        self.active_links[self.demand <= 0] = 0
        return np.sum(V2I_rate), np.sum(V2V_rate)

