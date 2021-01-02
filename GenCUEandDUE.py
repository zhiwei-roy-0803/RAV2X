import numpy as np

class VehicleGenerator():

    def __init__(self, d0, lanewidth, numLane, distBStoHwy):
        self.d0 = d0
        self.lanewidth = lanewidth
        self.numLane = numLane
        self.distBStoHwy = distBStoHwy

    def generate_CUE_and_DUE(self, d_avg, numCUE, numDUE):
        while 1:
            vehPos = []
            # 每个车道产生根据车流密度利用柏松分布产生一个随机数确定这个车道有多少车，然后利用均匀分布在这个车道上进行实际的撒点
            for ilane in range(self.numLane):
                nveh = np.random.poisson(lam=2*self.d0/d_avg)
                posi_ilane = np.zeros((nveh, 2))
                posi_ilane[:, 0] = (2*np.random.rand(nveh) - 1) * self.d0
                posi_ilane[:, 1] = (self.distBStoHwy + ilane * self.lanewidth) * np.ones(nveh)
                vehPos.append(posi_ilane)
            vehPos = np.concatenate(vehPos)
            numVeh = vehPos.shape[0]
            if numVeh > numCUE + numDUE * 2:
                break
        # 从所有的汽车中随机选择一些车辆出来作为D2D通信的发起者和接收方，每个D2D通信的发起者的接收机都是距离它最近的那个
        indPerm = np.random.permutation(numVeh)
        indDUETransmitter = indPerm[:numDUE]
        indDUEReceiver = -np.ones(numDUE, dtype=np.int)
        for i in range(numDUE):
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
        cntCUE = numDUE + 1
        indCUE = []
        while cntCUE <= numVeh:
            if indPerm[cntCUE] not in indDUEReceiver:
                indCUE.append(indPerm[cntCUE])
            cntCUE += 1
            if len(indCUE) >= numCUE:
                break
        return indCUE, indDUETransmitter, indDUEReceiver, vehPos

if __name__ == "__main__":
    d0 = np.sqrt(500**2 - 35**2)
    Generator = VehicleGenerator(d0, 4, 6, 35)
    indCUE, indDUETransmitter, indDUEReceiver, vehPos = Generator.generate_CUE_and_DUE(2.5*60/3.6, 20, 20)



