import numpy as np
from GenCUEandDUE import VehicleGenerator
from HighwayChannel import HwyChannelLargeScaleFadingGenerator
from utils import computeCapacity
from munkres import Munkres
from utils import GetCost
from copy import deepcopy
from GraphMatcher import GraphMatching3D

class GraphAllocator():
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
        self.gammaProp = self.gamma0/(-np.log(1-self.p0))
        self.Pd_max = 10**(self.dB_Pd_max/10)
        self.Pc_max = 10**(self.dB_Pc_max/10)
        # CUE/DUE number setup
        self.numDUE = config["numDUE"]
        self.numCUE = config["numCUE"]
        self.numCluster = config["numCluster"]
        # initialize vehicle sampler
        d0 = np.sqrt(self.radius**2 - self.disBstoHwy**2)
        self.vehicle_generator = VehicleGenerator(d0, self.laneWidth, self.numLane, self.disBstoHwy)
        # initialize channel large scale fading generator
        self.fading_generator = HwyChannelLargeScaleFadingGenerator(self.stdV2I, self.stdV2V, self.vehHgt, self.bsHgt,
                                                                    self.freq, self.vehAntGain, self.bsAntGain,
                                                                    self.bsNoiseFigure, self.vehNoiseFigure)
        # initialize Hungrian solver
        self.matcher = GraphMatching3D(self.numCUE, self.numCUE, self.numCluster)

    def sample_channel(self, indCUE, indDUETransmitter, indDUEReceiver, vehPos):
        alpha_mB = np.zeros(self.numCUE)
        alpha_kk = np.zeros((self.numDUE, self.numDUE))
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
                alpha_mk[m, k] = 10**(dB_alpha_mk/10)

        for k in range(self.numDUE):
            # 计算第K对DUE之间的路损以及所有的V2V链路之间的路损
            for kk in range(self.numDUE):
                dist_kk = np.sqrt(np.sum((vehPos[indDUETransmitter[k]]-vehPos[indDUEReceiver[kk]])**2))
                dB_alpha_kk = self.fading_generator.generate_fading_V2V(dist_kk)
                alpha_kk[k, kk] = 10**(dB_alpha_kk/10)
            # 计算第k对DUE的发射机对基站的干扰
            dist_k = np.sqrt(vehPos[indDUETransmitter[k], 0]**2 + vehPos[indDUETransmitter[k], 1]**2) # k-th DUE发射机到基站的距离
            dB_alpha_kB = self.fading_generator.generate_fading_V2I(dist_k)
            alpha_kB[k] = 10**(dB_alpha_kB/10)

        return alpha_mB, alpha_kk, alpha_kB, alpha_mk

    def compute_opt_power(self, alpha_mk, alpha_kk, alpha_mB, alpha_kB):
        numK = len(alpha_mk)
        alpha_mk = np.reshape(alpha_mk, (numK, 1))
        phi = -self.gammaProp * np.transpose(alpha_kk)
        for i in range(numK):
            phi[i, i] = alpha_kk[i, i]
        phi_inv = np.linalg.inv(phi)
        num = (self.Pd_max - self.gammaProp * self.sig2 * np.sum(phi_inv, 1))
        den = (self.gammaProp*np.dot(phi_inv, alpha_mk))
        num = np.reshape(num, (numK, 1))
        Pc_cand = num / den
        Pc_opt = min(np.append(Pc_cand, self.Pc_max))
        if Pc_opt <= 0:
            capacity = -1
            Pd_opt = 0
            return capacity, Pc_opt, Pd_opt
        Pd_opt = np.dot(phi_inv, self.gammaProp*(Pc_opt*alpha_mk + self.sig2))
        if np.sum(Pd_opt <= 0) > 1:
            capacity = -1
            return capacity, Pc_opt, Pd_opt
        signal = Pc_opt * alpha_mB
        interference = np.dot(Pd_opt.T, alpha_kB)
        capacity = np.log2(1 + signal/(self.sig2 + interference))
        return capacity, Pc_opt, np.squeeze(Pd_opt)

    def gen_cluster(self, alpha_kk):
        clusterMat = np.zeros((self.numCluster, self.numDUE))
        indPerm = np.random.permutation(self.numDUE)
        tmp = indPerm[:self.numCluster]
        # arbitrarily assign one link to each cluster
        for i in range(self.numCluster):
            clusterMat[i, tmp[i]] = 1
        # put a V2V link into one of the established clusters so that the minimum intra-cluster interference is achieved
        for i in range(self.numCluster, self.numDUE):
            delta = np.zeros(self.numCluster)
            for ic in range(self.numCluster):
                for kk in range(self.numDUE):
                    if clusterMat[ic, kk] == 1:
                        delta[ic] += (alpha_kk[kk, indPerm[i]] + alpha_kk[indPerm[i], kk])
            min_ind = np.argmin(delta)
            clusterMat[min_ind, indPerm[i]] = 1
        return clusterMat

    def run_allocation_baseline(self, v):
        # sample vehicles
        indCUE, indDUETransmitter, indDUEReceiver, vehPos = self.vehicle_generator.generate_CUE_and_DUE(2.5*v/3.6,
                                                                                                        self.numCUE,
                                                                                                        self.numDUE)

        # random large-scale fading generation
        alpha_mB, alpha_kk, alpha_kB, alpha_mk = self.sample_channel(indCUE, indDUETransmitter, indDUEReceiver, vehPos)

        # generate clusters for all V2V links
        clusterMat = self.gen_cluster(alpha_kk=alpha_kk)
        # run resource allocation for this channel and vehicle realization
        # single pair power allocation...
        h_mBf = np.random.randn(self.numCUE, self.numCUE, 2)
        h_mBf = (h_mBf[:, :, 0]**2 + h_mBf[:, :, 1]**2)/2
        h_kBf = np.random.randn(self.numDUE, self.numCUE, 2)
        h_kBf = (h_kBf[:, :, 0]**2 + h_kBf[:, :, 1]**2)/2
        C_mfk = np.zeros((self.numCUE, self.numCUE, self.numCUE))
        Pd_mfk = np.zeros((self.numCUE, self.numCUE, self.numDUE))
        Pc_mfk = np.zeros((self.numCUE, self.numCUE, self.numDUE))
        # Compute optimal Pc and Pd for each pair
        for f in range(self.numCUE):
            g_kB = alpha_kB * h_kBf[:, f]
            g_mB = alpha_mB * h_mBf[:, f]
            for m in range(self.numCUE):
                for n in range(self.numCUE):
                    indices = np.where(clusterMat[n, :] == 1)[0]
                    alpha_m_ = alpha_mk[m, indices]
                    alpha_kk_ = alpha_kk[indices]
                    alpha_kk_ = alpha_kk_[:, indices]
                    g_mB_ = g_mB[m]
                    g_kB_ = g_kB[indices]

                    # compute optimal power allocation strategy
                    capacity, Pc_opt, Pd_opt = self.compute_opt_power(alpha_m_, alpha_kk_, g_mB_, g_kB_)

                    if capacity < 0:
                        C_mfk[m, f, n] = -2000
                        continue

                    C_mfk[m, f, n] =  capacity
                    Pd_mfk[m, f, indices] = Pd_opt
                    Pc_mfk[m, f, indices] = Pc_opt
        # 3D Graph Matching
        Z, Vertex1, Vertex2, Vertex3, approchRate = self.matcher.run_3DMatching(C_mfk)
        return approchRate

    def run_allocation_greedy(self, v, num_iter):
        # sample vehicles
        indCUE, indDUETransmitter, indDUEReceiver, vehPos = self.vehicle_generator.generate_CUE_and_DUE(2.5 * v / 3.6,
                                                                                                        self.numCUE,
                                                                                                        self.numDUE)

        # random large-scale fading generation
        alpha_mB, alpha_kk, alpha_kB, alpha_mk = self.sample_channel(indCUE, indDUETransmitter, indDUEReceiver, vehPos)

        # generate clusters for all V2V links
        clusterMat = self.gen_cluster(alpha_kk=alpha_kk)
        # run resource allocation for this channel and vehicle realization
        # single pair power allocation...
        h_mBf = np.random.randn(self.numCUE, self.numCUE, 2)
        h_mBf = (h_mBf[:, :, 0] ** 2 + h_mBf[:, :, 1] ** 2) / 2
        h_kBf = np.random.randn(self.numDUE, self.numCUE, 2)
        h_kBf = (h_kBf[:, :, 0] ** 2 + h_kBf[:, :, 1] ** 2) / 2
        C_mfk = np.zeros((self.numCUE, self.numCUE, self.numCUE))
        Pd_mfk = np.zeros((self.numCUE, self.numCUE, self.numDUE))
        Pc_mfk = np.zeros((self.numCUE, self.numCUE, self.numDUE))
        # Compute optimal Pc and Pd for each pair
        for it in range(num_iter):
            for k in range(self.numDUE):
                criterion = np.zeros(self.numCluster)
                current_cluster = np.where(clusterMat[:, k]==1)[0][0]
                if sum(clusterMat[current_cluster, :]) == 1:
                    continue
                clusterMat[current_cluster, k] = 0
                for cluster in range(self.numCluster):
                    clusterMatTmp = deepcopy(clusterMat)
                    clusterMatTmp[cluster, k] = 1
                    for f in range(self.numCUE):
                        g_kB = alpha_kB * h_kBf[:, f]
                        g_mB = alpha_mB * h_mBf[:, f]
                        for m in range(self.numCUE):
                            for n in range(self.numCUE):
                                indices = np.where(clusterMatTmp[n, :] == 1)[0]
                                alpha_m_ = alpha_mk[m, indices]
                                alpha_kk_ = alpha_kk[indices]
                                alpha_kk_ = alpha_kk_[:, indices]
                                g_mB_ = g_mB[m]
                                g_kB_ = g_kB[indices]

                                # compute optimal power allocation strategy
                                capacity, Pc_opt, Pd_opt = self.compute_opt_power(alpha_m_, alpha_kk_, g_mB_, g_kB_)

                                if capacity < 0:
                                    C_mfk[m, f, n] = -2000
                                    continue

                                C_mfk[m, f, n] = capacity
                                Pd_mfk[m, f, indices] = Pd_opt
                                Pc_mfk[m, f, indices] = Pc_opt
                    # 3D Graph Matching
                    Z, Vertex1, Vertex2, Vertex3, approchRate = self.matcher.run_3DMatching(C_mfk)
                    criterion[cluster] = approchRate

                # put the k-th V2V link to cluster with maximum system sum rate
                bestClusterIdx = np.argmax(criterion)
                clusterMat[bestClusterIdx, k] = 1

        # Start Baseline algorithm with new cluster assignment
        for f in range(self.numCUE):
            g_kB = alpha_kB * h_kBf[:, f]
            g_mB = alpha_mB * h_mBf[:, f]
            for m in range(self.numCUE):
                for n in range(self.numCUE):
                    indices = np.where(clusterMat[n, :] == 1)[0]
                    alpha_m_ = alpha_mk[m, indices]
                    alpha_kk_ = alpha_kk[indices]
                    alpha_kk_ = alpha_kk_[:, indices]
                    g_mB_ = g_mB[m]
                    g_kB_ = g_kB[indices]

                    # compute optimal power allocation strategy
                    capacity, Pc_opt, Pd_opt = self.compute_opt_power(alpha_m_, alpha_kk_, g_mB_, g_kB_)

                    if capacity < 0:
                        C_mfk[m, f, n] = -2000
                        continue

                    C_mfk[m, f, n] = capacity
                    Pd_mfk[m, f, indices] = Pd_opt
                    Pc_mfk[m, f, indices] = Pc_opt

        # 3D Graph Matching
        Z, Vertex1, Vertex2, Vertex3, approchRate = self.matcher.run_3DMatching(C_mfk)
        return approchRate


