from ResourceAllocator import DelayedCSIAllocator
import numpy as np
from tqdm import tqdm
import os

import argparse

# simulation parameter
max_run = 1000
# Hyper-parameter configuration
config = {"dB_Pd_max":23,
          "dB_Pc_max":23,
          "stdV2V":3,
          "stdV2I":8,
          "freq":2,
          "radius":500,
          "bsHgt":25,
          "disBstoHwy":35,
          "bsAntGain":8,
          "bsNoiseFigure":5,
          "vehHgt":1.5,
          "vehAntGain":3,
          "vehNoiseFigure":9,
          "numLane":6,
          "laneWidth":4,
          "r0":0.5,
          "dB_gamma0":5,
          "p0":0.001,
          "dB_sigma2":-114,
          "numDUE":20,
          "numCUE":20}

def run_different_feedbacktime(obj, v=50):
    print("Run Different FeedBack Period")
    feedbacktime = np.linspace(0.2, 1.2, 6)
    allocator = DelayedCSIAllocator(config)
    sum_capacity_array = []
    min_capacity_array = []
    for t in feedbacktime:
        total_sum = 0
        total_min = 0
        pbar = tqdm(range(max_run))
        valid_cnt = 0
        for _ in pbar:
            sum_capacity, min_capacity = allocator.run_allocation(v=v, T=t)
            if min_capacity < 0:
                continue
            total_sum += sum_capacity
            total_min += min_capacity
            valid_cnt += 1
            pbar.set_description("SumCapcity={:2f}, MinCapacity={:.2f}".format(sum_capacity, min_capacity))

        avg_sum = total_sum/max_run
        avg_min = total_min/max_run
        print("Feedback Perio = {:.3f} ms, Avg_SumCapacity = {:.3f}, Avg_MinCapacity = {:.3f}".format(t, avg_sum, avg_min))
        sum_capacity_array.append(avg_sum)
        min_capacity_array.append(avg_min)
    # Save Statistics
    save_dir = "./results/DelayCSI/different_feedback/{:s}/Velocity={:d}".format(obj, v)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sum_capacity_path = os.path.join(save_dir, "SumCapacity")
    min_capacity_path = os.path.join(save_dir, "MinCapacity")
    sum_capacity_array = np.array(sum_capacity_array)
    min_capacity_array = np.array(min_capacity_array)
    np.savetxt(sum_capacity_path, sum_capacity_array, fmt='%.4f', delimiter='\n')
    np.savetxt(min_capacity_path, min_capacity_array, fmt='%.4f', delimiter='\n')

def run_different_speed(obj, t=1.0):
    print("Run Different Speed")
    speed = np.arange(60, 150, 10)
    allocator = DelayedCSIAllocator(config)
    sum_capacity_array = []
    min_capacity_array = []
    for v in speed:
        total_sum = 0
        total_min = 0
        pbar = tqdm(range(max_run))
        valid_cnt = 0
        for _ in pbar:
            sum_capacity, min_capacity = allocator.run_allocation(v=v, T=t)
            if min_capacity < 0:
                continue
            total_sum += sum_capacity
            total_min += min_capacity
            valid_cnt += 1
            pbar.set_description("SumCapcity={:.3f}, MinCapacity={:.3f}".format(sum_capacity, min_capacity))

        avg_sum = total_sum/valid_cnt
        avg_min = total_min/valid_cnt
        print("Speed = {:3f} km/h, Avg_SumCapacity = {:.3f}, Avg_MinCapacity = {:.3f}".format(v, avg_sum, avg_min))
        sum_capacity_array.append(avg_sum)
        min_capacity_array.append(avg_min)
    # Save Statistics
    save_dir = "./results/DelayCSI/different_speed/{:s}/FeedBackPeriod={:.1f}".format(obj, t)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sum_capacity_path = os.path.join(save_dir, "SumCapacity")
    min_capacity_path = os.path.join(save_dir, "MinCapacity")
    sum_capacity_array = np.array(sum_capacity_array)
    min_capacity_array = np.array(min_capacity_array)
    np.savetxt(sum_capacity_path, sum_capacity_array, fmt='%.4f', delimiter='\n')
    np.savetxt(min_capacity_path, min_capacity_array, fmt='%.4f', delimiter='\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=int, default=50)
    parser.add_argument("-feedback", type=float, default=1.0)
    args = parser.parse_args()
    v = args.v
    fbPeriod = args.feedback
    # run_different_feedbacktime(obj="MaxSum", v=v)
    run_different_speed(obj="MaxSum", t=fbPeriod)



