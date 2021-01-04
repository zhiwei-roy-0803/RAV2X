import numpy as np

class HwyChannelLargeScaleFadingGenerator():
    def __init__(self, stdV2I, stdV2V, vehHeight, bsHeight, freq, vehAntGain, bsAntGain, bsNoiseFigure, vehNoiseFigure):
        self.stdV2I = stdV2I
        self.stdV2V = stdV2V
        self.vehHeight = vehHeight
        self.bsHeight = bsHeight
        self.freq = freq
        self.vehAntGain = vehAntGain
        self.bsAntGain = bsAntGain
        self.bsNoiseFigure = bsNoiseFigure
        self.vehNoiseFigure = vehNoiseFigure

    def generate_fading_V2I(self, dist_veh2bs):
        dist2 = (self.vehHeight - self.bsHeight)**2 + dist_veh2bs**2
        pathloss = 128.1 + 37.6 * np.log10(np.sqrt(dist2)/1000) # 路损公式中距离使用km计算
        combinedPL = -(np.random.randn() * self.stdV2I + pathloss)
        return combinedPL + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure

    def generate_fading_V2V(self, dist_DuePair):
        d_bp = 4*(self.vehHeight - 1)*(self.vehHeight - 1) * self.freq * (10**9) / (3*10**8)
        A = 22.7
        B = 41.0
        C = 20
        if dist_DuePair <= 3:
            pathloss = A * np.log10(3) + B + C * np.log10(self.freq/5)
        elif dist_DuePair <= d_bp:
            pathloss = A * np.log10(dist_DuePair) + B + C * np.log10(self.freq / 5)
        else:
            pathloss = 40*np.log10(dist_DuePair) + 9.45 - 17.3 * np.log10((self.vehHeight - 1)*(self.vehHeight - 1)) + 2.7*np.log10(self.freq/5)
        combinedPL = -(np.random.randn() * self.stdV2V + pathloss)
        return combinedPL + self.vehAntGain * 2 - self.vehNoiseFigure