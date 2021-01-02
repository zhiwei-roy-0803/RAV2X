from ResourceAllocator import LargeScaleFadingOnlyAllocator
import numpy as np
from tqdm import tqdm
import os

# simulation parameter
max_run = 1000
# Hyper-parameter configuration
config = {"dB_Pd_max": 23,
          "dB_Pc_max": 23,
          "stdV2V": 3,
          "stdV2I": 8,
          "freq": 2,
          "radius": 500,
          "bsHgt": 25,
          "disBstoHwy": 35,
          "bsAntGain": 8,
          "bsNoiseFigure": 5,
          "vehHgt": 1.5,
          "vehAntGain": 3,
          "vehNoiseFigure": 9,
          "numLane": 6,
          "laneWidth": 4,
          "r0": 0.5,
          "dB_gamma0": 5,
          "p0": 0.001,
          "dB_sigma2": -114,
          "numDUE": 20,
          "numCUE": 20}

def run_different_speed(obj):
    speeds = np.arange(60, 150, 10)
    allocator = LargeScaleFadingOnlyAllocator(config)
    sum_capacity_array = []
    min_capacity_array = []
    for v in speeds:
        print("Start Simulation for Speed = {:d} km/h".format(v))
        total_sum = 0
        total_min = 0
        pbar = tqdm(range(max_run))
        valid_cnt = 0
        for _ in pbar:
            sum_capacity, min_capacity = allocator.run_allocation(v, obj=obj)
            if min_capacity < 0:
                continue
            total_sum += sum_capacity
            total_min += min_capacity
            valid_cnt += 1
            pbar.set_description("SumCapcity={:2f}, MinCapacity={:.2f}".format(sum_capacity, min_capacity))

        print("Speed = {:d} km/h, Avg_SumCapacity = {:.3f}, Avg_MinCapacity = {:.3f}".format(v,
                                                                                             total_sum/valid_cnt,
                                                                                             total_min/valid_cnt))
        sum_capacity_array.append(total_sum/valid_cnt)
        min_capacity_array.append(total_min/valid_cnt)

    # Save Statistics
    save_dir = "./results/LargeScale/different_speed/{:s}/Pmax_{:d}dB".format(obj, config["dB_Pd_max"])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sum_capacity_path = os.path.join(save_dir, "SumCapacity")
    min_capacity_path = os.path.join(save_dir, "MinCapacity")
    sum_capacity_array = np.array(sum_capacity_array)
    min_capacity_array = np.array(min_capacity_array)

    np.savetxt(sum_capacity_path, sum_capacity_array, fmt='%.4f', delimiter='\n')
    np.savetxt(min_capacity_path, min_capacity_array, fmt='%.4f', delimiter='\n')

def run_different_p0(obj):
    p0s = np.arange(0.001, 0.1, 5)
    allocator = LargeScaleFadingOnlyAllocator(config)
    sum_capacity_array = []
    min_capacity_array = []
    for p0 in p0s:
        allocator.p0 = p0
        total_sum = 0
        total_min = 0
        pbar = tqdm(range(max_run))
        valid_cnt = 0
        for _ in pbar:
            sum_capacity, min_capacity = allocator.run_allocation(100, obj=obj)
            if min_capacity < 0:
                continue
            total_sum += sum_capacity
            total_min += min_capacity
            valid_cnt += 1
            pbar.set_description("SumCapcity={:2f}, MinCapacity={:.2f}".format(sum_capacity, min_capacity))
        print("Speed = {:d} km/h, Avg_SumCapacity = {:.3f}, Avg_MinCapacity = {:.3f}".format(p0,
                                                                                             total_sum/valid_cnt,
                                                                                             total_min/valid_cnt))
        sum_capacity_array.append(total_sum/valid_cnt)
        min_capacity_array.append(total_min/valid_cnt)

    # Save Statistics
    save_dir = "./results/LargeScale/different_p0/{:s}/Pmax_{:d}dB".format(obj, config["dB_Pd_max"])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sum_capacity_path = os.path.join(save_dir, "SumCapacity")
    min_capacity_path = os.path.join(save_dir, "MinCapacity")
    sum_capacity_array = np.array(sum_capacity_array)
    min_capacity_array = np.array(min_capacity_array)
    np.savetxt(sum_capacity_path, sum_capacity_array, fmt='%.4f', delimiter='\n')
    np.savetxt(min_capacity_path, min_capacity_array, fmt='%.4f', delimiter='\n')

if __name__ == "__main__":
    run_different_speed(obj="MaxSum")
    run_different_p0(obj="MaxSum")
