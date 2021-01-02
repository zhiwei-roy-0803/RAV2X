import numpy as np
from GenCUEandDUE import VehicleGenerator
from HighwayChannel import HwyChannelLargeScaleFadingGenerator
from utils import computeCapacity
from munkres import Munkres
from utils import GetCost
from copy import deepcopy
from scipy import special

class LargeScaleFadingOnlyAllocator():
    def __init__(self, config):
        # max DUE/CUE transmit power in dBm
        self.dB_Pd_max = config["dB_Pd_max"]
        self.dB_Pc_max = config["dB_Pc_max"]
        # large scale fading parameters
        self.stdV2V = config["stdV2V"]
        self.stdV2I = config["stdV2I"]
        # cell parameter setup
        self.freq = config["freq"]
        self.radius = config["radius"]
        self.bsHgt = config["bsHgt"]
        self.disBstoHwy = config["disBstoHwy"]
        self.bsAntGain = config["bsAntGain"]
        self.bsNoiseFigure = config["bsNoiseFigure"]
        # vehicle parameter setup
        self.vehHgt = config["vehHgt"]
        self.vehAntGain = config["vehAntGain"]
        self.vehNoiseFigure = config["vehNoiseFigure"]
        # Highway parameter setup
        self.numLane = config["numLane"]
        self.laneWidth = config["laneWidth"]
        # QoS parameters for CUE and DUE
        self.r0 = config["r0"]
        self.dB_gamma0 = config["dB_gamma0"]
        self.p0 = config["p0"]
        self.dB_sigma2 = config["dB_sigma2"]
        # dB to linear scale conversion
        self.sig2 = 10**(self.dB_sigma2/10)
        self.gamma0 = 10**(self.dB_gamma0/10)
        self.Pd_max = 10**(self.dB_Pd_max/10)
        self.Pc_max = 10**(self.dB_Pc_max/10)
        # CUE/DUE number setup
        self.numDUE = config["numDUE"]
        self.numCUE = config["numCUE"]
        # initialize vehicle sampler
        d0 = np.sqrt(self.radius**2 - self.disBstoHwy**2)
        self.vehicle_generator = VehicleGenerator(d0, self.laneWidth, self.numLane, self.disBstoHwy)
        # initialize channel large scale fading generator
        self.fading_generator = HwyChannelLargeScaleFadingGenerator(self.stdV2I, self.stdV2V, self.vehHgt, self.bsHgt,
                                                                    self.freq, self.vehAntGain, self.bsAntGain,
                                                                    self.bsNoiseFigure, self.vehNoiseFigure)
        # initialize Hungrian solver
        self.munkres = Munkres()

    def sample_channel(self, indCUE, indDUETransmitter, indDUEReceiver, vehPos):
        alpha_mB = np.zeros(self.numCUE)
        alpha_k = np.zeros(self.numDUE)
        alpha_kB = np.zeros(self.numDUE)
        alpha_mk = np.zeros((self.numCUE, self.numDUE))
        for m in range(self.numCUE):
            # 计算第m个CUE对基站的距离和路损，假设基站的坐标是（0，0）
            dist_mB = np.sqrt(vehPos[indCUE[m], 0]**2 + vehPos[indCUE[m], 1]**2) # m-th CUE到基站的距离
            dB_alpha_mB = self.fading_generator.generate_fading_V2I(dist_mB)
            alpha_mB[m] = 10**(dB_alpha_mB/10)
            # 计算第m个CUE和第k个DUE之间的距离、路损
            for k in range(self.numDUE):
                dist_mk = np.sqrt(np.sum((vehPos[indCUE[m]]-vehPos[indDUEReceiver[k]])**2))
                dB_alpha_mk = self.fading_generator.generate_fading_V2V(dist_mk)
                alpha_mk[m][k] = 10**(dB_alpha_mk/10)

        for k in range(self.numDUE):
            # 计算第K对DUE之间的距离和路损
            dist_k = np.sqrt(np.sum((vehPos[indDUETransmitter[k]]-vehPos[indDUEReceiver[k]])**2))
            dB_alpha_k = self.fading_generator.generate_fading_V2V(dist_k)
            alpha_k[k] = 10**(dB_alpha_k/10)
            # 计算第k对DUE的发射机对基站的干扰
            dist_k = np.sqrt(vehPos[indDUETransmitter[k], 0]**2 + vehPos[indDUETransmitter[k], 1]**2) # k-th DUE发射机到基站的距离
            dB_alpha_kB = self.fading_generator.generate_fading_V2I(dist_k)
            alpha_kB[k] = 10**(dB_alpha_kB/10)

        return alpha_mB, alpha_k, alpha_kB, alpha_mk


    def run_allocation(self, d_avg, obj="MaxSum"):
        # sample vehicles
        indCUE, indDUETransmitter, indDUEReceiver, vehPos = self.vehicle_generator.generate_CUE_and_DUE(2.5*d_avg/3.6, self.numCUE, self.numDUE)

        # random large-scale fading generation
        alpha_mB, alpha_k, alpha_kB, alpha_mk = self.sample_channel(indCUE, indDUETransmitter, indDUEReceiver, vehPos)

        # run resource allocation for this channel and vehicle realization

        # single pair power allocation...
        C_mk = np.zeros((self.numCUE, self.numCUE))
        for m in range(self.numCUE):
            alpha_mB_ = alpha_mB[m]
            for k in range(self.numCUE):
                if k > self.numDUE:
                    a = self.Pc_max * alpha_mB / self.sig2
                    C_mk[m][k] = computeCapacity(a, 0)
                    continue
                else:
                    alpha_k_ = alpha_k[k]
                    alpha_kB_ = alpha_kB[k]
                    alpha_mk_ = alpha_mk[m][k]

                    Pc_dmax = alpha_k_ * self.Pd_max / (self.gamma0*alpha_mk_) * (np.exp(-self.gamma0*self.sig2/(self.Pd_max*alpha_k_))/(1-self.p0)-1)
                    if Pc_dmax <= self.Pc_max:
                        Pd_opt = self.Pd_max
                        Pc_opt = Pc_dmax
                    else:
                        eps = 1e-5
                        Pd_left = -self.gamma0*self.sig2/(alpha_k_*np.log(1-self.p0))
                        Pd_right = self.Pd_max
                        tmp = 0
                        while Pd_right - Pd_left > eps:
                            tmp = (Pd_left + Pd_right) / 2
                            if alpha_k_*tmp/(self.gamma0*alpha_mk_)*(np.exp(-self.gamma0*self.sig2/(tmp*alpha_k_))/(1-self.p0) -1) > self.Pc_max:
                                Pd_right = tmp
                            else:
                                Pd_left = tmp
                        Pd_cmax = tmp
                        Pd_opt = Pd_cmax
                        Pc_opt = self.Pc_max

                a = Pc_opt*alpha_mB_/self.sig2
                b = Pd_opt*alpha_kB_/self.sig2
                C_mk[m, k] = computeCapacity(a, b)
                if C_mk[m, k] < self.r0:
                    C_mk[m, k] = -2000

        # bipartite graph matching...
        if obj == "MaxSum":
            assignments = self.munkres.compute(-C_mk)
            sum_capacity, min_capacity = GetCost(C_mk, assignments)
        if obj == "MaxMin":
            C_mk_1d = np.reshape(C_mk, (-1,))
            sortedCapacity = np.sort(C_mk_1d)
            minInd = 0
            maxInd = self.numCUE*self.numDUE - 1
            while maxInd - minInd > 1:
                midInd = (minInd + maxInd) // 2
                midValue = sortedCapacity[midInd]
                F = np.zeros_like(C_mk)
                F[C_mk < midValue] = 1
                F_in = deepcopy(F)
                asgn = self.munkres.compute(F_in)
                cost, _ = GetCost(F, asgn)
                if cost > 0:
                    maxInd = midInd
                else:
                    minInd = midInd
                    assignments = asgn
            sum_capacity, min_capacity = GetCost(C_mk, assignments)
        return sum_capacity, min_capacity


class DelayedCSIAllocator():
    def __init__(self, config):
        # max DUE/CUE transmit power in dBm
        self.dB_Pd_max = config["dB_Pd_max"]
        self.dB_Pc_max = config["dB_Pc_max"]
        # large scale fading parameters
        self.stdV2V = config["stdV2V"]
        self.stdV2I = config["stdV2I"]
        # cell parameter setup
        self.freq = config["freq"]
        self.radius = config["radius"]
        self.bsHgt = config["bsHgt"]
        self.disBstoHwy = config["disBstoHwy"]
        self.bsAntGain = config["bsAntGain"]
        self.bsNoiseFigure = config["bsNoiseFigure"]
        # vehicle parameter setup
        self.vehHgt = config["vehHgt"]
        self.vehAntGain = config["vehAntGain"]
        self.vehNoiseFigure = config["vehNoiseFigure"]
        # Highway parameter setup
        self.numLane = config["numLane"]
        self.laneWidth = config["laneWidth"]
        # QoS parameters for CUE and DUE
        self.r0 = config["r0"]
        self.dB_gamma0 = config["dB_gamma0"]
        self.p0 = config["p0"]
        self.dB_sigma2 = config["dB_sigma2"]
        # dB to linear scale conversion
        self.sig2 = 10**(self.dB_sigma2/10)
        self.gamma0 = 10**(self.dB_gamma0/10)
        self.Pd_max = 10**(self.dB_Pd_max/10)
        self.Pc_max = 10**(self.dB_Pc_max/10)
        # CUE/DUE number setup
        self.numDUE = config["numDUE"]
        self.numCUE = config["numCUE"]
        # initialize vehicle sampler
        d0 = np.sqrt(self.radius**2 - self.disBstoHwy**2)
        self.vehicle_generator = VehicleGenerator(d0, self.laneWidth, self.numLane, self.disBstoHwy)
        # initialize channel large scale fading generator
        self.fading_generator = HwyChannelLargeScaleFadingGenerator(self.stdV2I, self.stdV2V, self.vehHgt, self.bsHgt,
                                                                    self.freq, self.vehAntGain, self.bsAntGain,
                                                                    self.bsNoiseFigure, self.vehNoiseFigure)
        # initialize Hungrian solver
        self.munkres = Munkres()

    def sample_channel(self, indCUE, indDUETransmitter, indDUEReceiver, vehPos):
        alpha_mB = np.zeros(self.numCUE)
        alpha_k = np.zeros(self.numDUE)
        alpha_kB = np.zeros(self.numDUE)
        alpha_mk = np.zeros((self.numCUE, self.numDUE))
        for m in range(self.numCUE):
            # 计算第m个CUE对基站的距离和路损，假设基站的坐标是（0，0）
            dist_mB = np.sqrt(vehPos[indCUE[m], 0]**2 + vehPos[indCUE[m], 1]**2) # m-th CUE到基站的距离
            dB_alpha_mB = self.fading_generator.generate_fading_V2I(dist_mB)
            alpha_mB[m] = 10**(dB_alpha_mB/10)
            # 计算第m个CUE和第k个DUE之间的距离、路损
            for k in range(self.numDUE):
                dist_mk = np.sqrt(np.sum((vehPos[indCUE[m]]-vehPos[indDUEReceiver[k]])**2))
                dB_alpha_mk = self.fading_generator.generate_fading_V2V(dist_mk)
                alpha_mk[m][k] = 10**(dB_alpha_mk/10)

        for k in range(self.numDUE):
            # 计算第K对DUE之间的距离和路损
            dist_k = np.sqrt(np.sum((vehPos[indDUETransmitter[k]]-vehPos[indDUEReceiver[k]])**2))
            dB_alpha_k = self.fading_generator.generate_fading_V2V(dist_k)
            alpha_k[k] = 10**(dB_alpha_k/10)
            # 计算第k对DUE的发射机对基站的干扰
            dist_k = np.sqrt(vehPos[indDUETransmitter[k], 0]**2 + vehPos[indDUETransmitter[k], 1]**2) # k-th DUE发射机到基站的距离
            dB_alpha_kB = self.fading_generator.generate_fading_V2I(dist_k)
            alpha_kB[k] = 10**(dB_alpha_kB/10)
        return alpha_mB, alpha_k, alpha_kB, alpha_mk


    def compute_opt_power(self, tol, alpha_k, alpha_mk, h_k, h_mk, epsilon_k, epsilon_mk):
        alpha_k = 1.9564e-5
        alpha_mk = 2.9124e-11
        h_k = np.array([-1.2357, 0.4734])
        h_mk = np.array([-0.5180, -0.7021])
        norm_h_k = np.linalg.norm(h_k)**2
        norm_h_mk = np.linalg.norm(h_mk)**2
        den0 = alpha_mk * (1 - epsilon_mk**2) * (1/self.p0 - 1) * (epsilon_k**2) * norm_h_k -\
               (1 - epsilon_k**2) * alpha_mk * (epsilon_mk**2) * norm_h_mk
        Pc0 = (1-epsilon_k**2)*self.sig2/den0
        Pd0 = Pc0*self.gamma0*alpha_mk*(1-self.epsilon_mk**2)*(1-self.p0)/(alpha_k*(1-self.epsilon_k**2)*self.p0)

        if self.Pd_max <= Pd0:
            # Case I：点（Pd0, Pc0）在（Pd_max, Pc_max）为对角线点矩阵外部，同时满足Pd0 > Pd_max，此时有2种情况
            B = self.Pd_max * alpha_k * (1 - epsilon_k**2)
            C = self.sig2 + self.Pc_max * (epsilon_mk**2) * alpha_mk * norm_h_mk
            D = self.Pc_max * alpha_mk * (1 - epsilon_mk**2)
            tmp = 1/(1-self.p0) * np.exp((epsilon_k**2) * norm_h_k/(1 - epsilon_k**2))
            if np.exp(C*self.gamma0/B)*(1+D/B*self.gamma0) - tmp > 0:
                # 此时Pd_max<=Pd0，同时Pc_d1_max<=Pc_max：最优点为（Pd_max, Pc_d1_max）
                Pd_opt = self.Pd_max
                P_left = 0
                P_right = self.Pc_max
                B = Pd_opt * alpha_k * (1 - epsilon_k**2)
                while abs(P_right - P_left) > tol:
                    P_mid = (P_left + P_right) / 2
                    C = self.sig2 + P_mid * (epsilon_mk**2) * alpha_mk * norm_h_mk
                    D = P_mid * alpha_mk * (1 - epsilon_mk**2)
                    if np.exp(C*self.gamma0/B)*(1+D/B*self.gamma0) - tmp > 0:
                        P_right = P_mid
                    else:
                        P_left = P_mid
                Pc_opt = P_mid
            else:
                # 此时Pd_max<=Pd0，但是Pc_d1_max > Pc_max，发生了削顶，最优点为（Pd_c1_max, Pc_max）
                Pc_opt = self.Pc_max
                P_left = 0
                P_right = self.Pd_max
                C = self.sig2 + Pc_opt * (epsilon_mk**2) * alpha_mk * norm_h_mk
                D = Pc_opt * alpha_mk * (1 - epsilon_mk**2)
                while abs(P_right - P_left) > tol:
                    P_mid = (P_left + P_right) / 2
                    B = P_mid * alpha_k * (1 - self.epsilon_k**2)
                    if np.exp(C*self.gamma0/B) * (1+D/B*self.gamma0) - tmp < 0:
                        P_right = P_mid
                    else:
                        P_left = P_mid
                Pd_opt = P_mid

        elif self.Pc_max > Pc0:
            # Case II：点（Pd0, Pc0）在（Pd_max, Pc_max）为对角线点矩阵内部，此时根据F2与Pd_max和Pc_max的关系有2种情况
            num = ((epsilon_mk**2)*norm_h_mk)/(1 - epsilon_mk**2)
            A = self.Pd_max * alpha_k * (epsilon_k**2) * norm_h_k
            B = self.Pd_max * alpha_k * (1 - epsilon_k**2)
            D = self.Pc_max * alpha_mk * (1 - epsilon_mk**2)
            den1 = np.log(1 + B/(self.gamma0*D))
            den2 = (A - self.sig2*self.gamma0) / (self.gamma0*D)
            if num - (den1 + den2) - np.log(self.p0) > 0:
                # 此时Pc_d2_max < Pc_max，不存在削顶现象，最优点为（Pd_max, Pc_d2_max），下面通过二分法求解Pc_d2_max
                Pd_opt = self.Pd_max
                P_left = 0
                P_right = self.Pc_max
                A = Pd_opt * alpha_k * (epsilon_k**2) * norm_h_k
                B = Pd_opt * alpha_k * (1 - epsilon_k**2)
                while abs(P_left - P_right) > tol:
                    P_mid = (P_left + P_right) / 2
                    D = P_mid * alpha_mk * (1 - epsilon_mk**2)
                    den1 = np.log(1 + B/(self.gamma0*D))
                    den2 = (A - self.sig2 * self.gamma0) / (self.gamma0 * D)
                    if num - (den1 + den2) - np.log(self.p0) > 0:
                        P_right = P_mid
                    else:
                        P_left = P_mid
                Pc_opt = P_mid
            else:
                Pc_opt = self.Pc_max
                P_left = 0
                P_right = self.Pd_max
                D = Pc_opt * alpha_mk * (1 - epsilon_mk**2)
                while abs(P_left - P_right) > tol:
                    P_mid = (P_left + P_right) / 2
                    A = P_mid * alpha_k * (epsilon_k**2) * norm_h_k
                    B = P_mid * alpha_k * (1 - epsilon_k**2)
                    den1 = np.log(1 + B/(self.gamma0*D))
                    den2 = (A - self.sig2 * self.gamma0) / (self.gamma0*D)
                    if num - (den1 + den2) - np.log(self.p0) < 0:
                        P_right = P_mid
                    else:
                        P_left = P_mid
                Pd_opt = P_mid
        else:
            # Case III
            tmp = 1/(1-self.p0)*np.exp((epsilon_k**2)*norm_h_k/(1-epsilon_k**2))
            Pc_opt = self.Pc_max
            P_left = 0
            P_right = self.Pd_max
            C = self.sig2 + Pc_opt*alpha_mk*(epsilon_mk**2)*norm_h_mk
            D = Pc_opt*alpha_mk*(1-epsilon_mk**2)
            while abs(P_right - P_left) > tol:
                P_mid = (P_right + P_left) / 2
                B = P_mid * alpha_k * (1 - epsilon_k**2)
                if np.exp(C*self.gamma0/B) * (1+D/B*self.gamma0) - tmp < 0:
                    P_right = P_mid
                else:
                    P_left = P_mid
            Pd_opt = P_mid
        return Pd_opt, Pc_opt


    def run_allocation(self, v, T):
        self.epsilon_k = special.j0(2 * np.pi * (T * 0.001) * (self.freq * (10**9)) * (v/3.6) / (3 * (10**8)))
        self.epsilon_mk = self.epsilon_k
        # sample vehicles
        indCUE, indDUETransmitter, indDUEReceiver, vehPos = self.vehicle_generator.generate_CUE_and_DUE(2.5*v/3.6, self.numCUE, self.numDUE)
        # random large-scale fading generation
        alpha_mB, alpha_k, alpha_kB, alpha_mk = self.sample_channel(indCUE, indDUETransmitter, indDUEReceiver, vehPos)
        # sample fast fading coefficient
        h_mB = np.random.randn(self.numCUE, 2)/np.sqrt(2)
        h_k = np.random.randn(self.numCUE, self.numDUE, 2)/np.sqrt(2)
        h_kB = np.random.randn(self.numCUE, self.numDUE, 2)/np.sqrt(2)
        h_mk = np.random.randn(self.numCUE, self.numDUE, 2)/np.sqrt(2)

        Pc_opts = np.zeros((self.numCUE, self.numDUE))
        Pd_opts = np.zeros((self.numCUE, self.numDUE))

        # single pair power allocation
        C_mk = np.zeros((self.numCUE, self.numCUE))
        for m in range(self.numCUE):
            alpha_mB_ = alpha_mB[m]
            h_mB_ = h_mB[m, :]
            g_mB_ = alpha_mB_ * np.linalg.norm(h_mB_)**2
            for k in range(self.numCUE):
                alpha_k_ = alpha_k[k]
                alpha_mk_ = alpha_mk[m, k]
                alpha_kB_ = alpha_kB[k]
                h_k_ = h_k[m, k]
                h_kB_ = h_kB[m, k]
                h_mk_ = h_mk[m, k]
                g_kB_ = alpha_kB_ * np.linalg.norm(h_kB_)**2

                # compute optimal power allocation scheme for the i-th CUE and the k-th VUE pair
                Pd_opt, Pc_opt = self.compute_opt_power(1e-6, alpha_k_, alpha_mk_, h_k_, h_mk_, self.epsilon_k, self.epsilon_mk)

                Pc_opts[m, k] = Pc_opt
                Pd_opts[m, k] = Pd_opt
                C_mk[m, k] = np.log2(1 + (Pc_opt*g_mB_)/(self.sig2 + Pd_opt*g_kB_))
                if C_mk[m, k] < self.r0:
                    C_mk[m, k] = -2000

        # Graph Matching for optimum allocation
        assignments = self.munkres.compute(-C_mk)
        sum_capacity, min_capacity = GetCost(C_mk, assignments)
        return sum_capacity, min_capacity

