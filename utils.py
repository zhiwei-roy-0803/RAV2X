import numpy as np
from scipy.integrate import quad

def f(x):
    return np.exp(-x)/x

def computeCapacity(a, b):
    c = a/((a - b)*np.log(2))
    if a >= 1/700 and b >= 1/700:
        int_a = quad(f, a=1/a, b=np.inf)[0]
        int_b = quad(f, a=1/b, b=np.inf)[0]
        return c * (np.exp(1/a)*int_a - np.exp(1/b)*int_b)
    elif a < 1/700 and b < 1/700:
        return c * (a - b)
    elif b < 1/700:
        int_a = quad(f, a=1 / a, b=np.inf)[0]
        return c * (np.exp(1/a)*int_a - b)
    else:
        int_b = quad(f, a=1/b, b=np.inf)[0]
        return c * (a - np.exp(1/b) * int_b)

def GetCost(costMat, assignments):
    sum_cost = 0
    min_cost = np.inf
    for row, column in assignments:
        cost = costMat[row][column]
        sum_cost += cost
        if cost < min_cost:
            min_cost = cost
    return sum_cost, min_cost

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import FuncFormatter
#
#
# memory_regular_5 = np.array([1971200, 1971200, 3957760, 3957760, 7930880, 7930880])
# memory_fast_5 = np.array([404480, 680960, 670720, 885760, 1356800, 1602560])
#
# ax = plt.figure(1, dpi=300)
# plt.rc('font',family='Times New Roman')
# plt.bar(np.arange(6), memory_fast_5, width=0.3, hatch='\\', color='c', linestyle='--')
# plt.bar(np.arange(6)+0.3, memory_regular_5, width=0.3, hatch="/", color='m')
# def formatnum(x, pos):
#     return '$%.0f$' % (x/1000000)
# formatter = FuncFormatter(formatnum)
# plt.gca().yaxis.set_major_formatter(formatter)
# plt.xticks(np.arange(6)+0.15, ["(128, 32)", "(128, 64)", "(256, 64)", "(256, 128)", "(512, 128)", "(512, 256)"])
# plt.ylabel('Memory Overhead (Bits)')
# plt.xlabel('Code Parameters')
# plt.text(-0.3, 8500000, s="x$10^{7}$", horizontalalignment='center', verticalalignment='center')
# plt.grid(linestyle='-.')
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['figure.dpi'] = 300
# plt.legend(["Memory-Efficient Quantized Decoder", "Regular Quantized Decoder"])
# plt.savefig("/Users/zhiweicao/Desktop/Low Latency Polar Decoder/FastLUT-Paper(Access)/memory.png")
