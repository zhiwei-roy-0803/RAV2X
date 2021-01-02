import matplotlib.pyplot as plt
import numpy as np
import os

result_root = os.path.join(os.getcwd(), "results")
figure_root = os.path.join(os.getcwd(), "Figures")

# simulation results for large scale CSI
SumCapacity_LargeScale_P23_DifferenV = np.loadtxt(os.path.join(result_root, "LargeScale/different_speed/MaxSum/Pmax_23dB/SumCapacity"), dtype=np.float32)

# simulation results for delay CSI
SumCapacity_DelayCSI_V50 = np.loadtxt(os.path.join(result_root, "DelayCSI/different_feedback/MaxSum/Velocity=50/SumCapacity"), dtype=np.float32)
SumCapacity_DelayCSI_V100 = np.loadtxt(os.path.join(result_root, "DelayCSI/different_feedback/MaxSum/Velocity=100/SumCapacity"), dtype=np.float32)
SumCapacity_DelayCSI_V150 = np.loadtxt(os.path.join(result_root, "DelayCSI/different_feedback/MaxSum/Velocity=150/SumCapacity"), dtype=np.float32)

# simulation results for delay CSI
SumCapacity_DelayCSI_Feedback_1_0 = np.loadtxt(os.path.join(result_root, "DelayCSI/different_speed/MaxSum/FeedBackPeriod=1.0/SumCapacity"), dtype=np.float32)
SumCapacity_DelayCSI_Feedback_0_2 = np.loadtxt(os.path.join(result_root, "DelayCSI/different_speed/MaxSum/FeedBackPeriod=0.2/SumCapacity"), dtype=np.float32)
SumCapacity_DelayCSI_Feedback_0_6 = np.loadtxt(os.path.join(result_root, "DelayCSI/different_speed/MaxSum/FeedBackPeriod=0.6/SumCapacity"), dtype=np.float32)

# Plot for Delay CSI

# Different FeedBack Period
feedbacktime = np.linspace(0.2, 1.2, 6)
plt.figure(dpi=300)
plt.plot(feedbacktime, SumCapacity_DelayCSI_V50, color='r', linestyle='-', marker="o", markerfacecolor='r',markersize=10)
plt.plot(feedbacktime, SumCapacity_DelayCSI_V100, color='k', linestyle='-', marker="o", markerfacecolor='r',markersize=10)
plt.plot(feedbacktime, SumCapacity_DelayCSI_V150, color='b', linestyle='-', marker="o", markerfacecolor='r',markersize=10)
plt.grid()
plt.xlabel("Feedback Period T (ms)")
plt.ylabel("Sum Capacity (bps/Hz)")
plt.ylim(220, 250)
plt.legend(["v=50 km/h", "v=100 km/h", "v=150 km/h"])
plt.savefig(os.path.join(figure_root, "DelayCSI", "Different_Feedback.png"))

# Different Speed with Different FeedBack Period
plt.figure(dpi=300)
speed = np.arange(60, 150, 10)
plt.plot(speed[::2], SumCapacity_LargeScale_P23_DifferenV[::2], color='b', linestyle='-', marker="o", markerfacecolor='r',markersize=10)
plt.plot(speed[::2], SumCapacity_DelayCSI_Feedback_0_2[::2], color='b', linestyle='-', marker="^", markerfacecolor='r',markersize=10)
plt.plot(speed[::2], SumCapacity_DelayCSI_Feedback_0_6[::2], color='b', linestyle='-', marker="*", markerfacecolor='r',markersize=10)
plt.plot(speed[::2], SumCapacity_DelayCSI_Feedback_1_0[::2], color='b', linestyle='-', marker="v", markerfacecolor='r',markersize=10)
plt.grid()
plt.xlabel("v km/h")
plt.ylabel("Sum Capacity (bps/Hz)")
plt.legend(["No Feedback", "T = 0.2 ms", "T = 0.6 ms", "T = 1.0 ms"])
plt.savefig(os.path.join(figure_root, "DelayCSI", "Different_Speed.png"))

# simulation results for graph-based methods
SumCapacity_Graph_DifferentFactor_23 = np.loadtxt(os.path.join(result_root, "GraphAllocation/different_factor/Baseline/Pmax_23dB/SumCapacity"), dtype=np.float32)
SumCapacity_Graph_DifferentFactor_17 = np.loadtxt(os.path.join(result_root, "GraphAllocation/different_factor/Baseline/Pmax_17dB/SumCapacity"), dtype=np.float32)
factors = np.arange(2, 8, 1)
plt.figure(dpi=300)
plt.plot(factors, SumCapacity_Graph_DifferentFactor_17, color='r', linestyle='-', marker="o", markerfacecolor='r',markersize=10)
plt.plot(factors, SumCapacity_Graph_DifferentFactor_23, color='b', linestyle='-', marker="o", markerfacecolor='r',markersize=10)
plt.grid()
plt.xlabel("K/M")
plt.ylabel("Sum Capacity (bps/Hz)")
plt.legend(["P_max = 17 dB", "P_max = 23 dB"])
plt.savefig(os.path.join(figure_root, "Graph", "Different_Factor.png"))

# simulation results for graph-based methods
SumCapacity_Graph_DifferentSpeed_23 = np.loadtxt(os.path.join(result_root, "GraphAllocation/different_speed/Baseline/Pmax_23dB/SumCapacity"), dtype=np.float32)
SumCapacity_Graph_DifferentSpeed_17 = np.loadtxt(os.path.join(result_root, "GraphAllocation/different_speed/Baseline/Pmax_17dB/SumCapacity"), dtype=np.float32)
speeds = np.arange(20, 180, 20)
plt.figure(dpi=300)
plt.plot(speeds, SumCapacity_Graph_DifferentSpeed_17, color='r', linestyle='-', marker="o", markerfacecolor='r',markersize=10)
plt.plot(speeds, SumCapacity_Graph_DifferentSpeed_23, color='b', linestyle='-', marker="o", markerfacecolor='r',markersize=10)
plt.grid()
plt.xlabel("v (km/h)")
plt.ylabel("Sum Capacity (bps/Hz)")
plt.legend(["P_max = 17 dB", "P_max = 23 dB"])
plt.savefig(os.path.join(figure_root, "Graph", "Different_Speed.png"))


