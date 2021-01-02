from GraphResourceAllocator import GraphAllocator
import numpy as np
from tqdm import tqdm
import os
import argparse

# simulation parameter
max_run = 500
# Hyper-parameter configuration
config = {"dB_Pd_max": 17,
          "dB_Pc_max": 17,
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
          "p0": 0.01,
          "dB_sigma2": -114,
          "numDUE": 3*10,
          "numCUE": 10,
          "numCluster": 10}

def run_different_speed(AlgorithmName="Baseline", max_power_dB=23):
    print("Run Simulation for Different Speed")
    speeds = np.arange(20, 180, 20)
    allocator = GraphAllocator(config)
    allocator.dB_Pd_max = max_power_dB
    allocator.dB_Pc_max = max_power_dB
    allocator.Pd_max = 10 ** (max_power_dB / 10)
    allocator.Pc_max = 10 ** (max_power_dB / 10)
    sum_capacity_array = []
    for v in speeds:
        # print("Start Simulation for Speed = {:d} km/h".format(v))
        total_sum = 0
        pbar = tqdm(range(max_run))
        valid_cnt = 0
        for _ in pbar:
            if AlgorithmName == "Baseline":
                sum_capacity = allocator.run_allocation_baseline(v)
            elif AlgorithmName == "Greedy":
                sum_capacity = allocator.run_allocation_greedy(v, num_iter=5)
            elif AlgorithmName == "Random":
                pass
            else:
                raise NotImplementedError
            if sum_capacity < 0:
                continue
            total_sum += sum_capacity
            valid_cnt += 1
            pbar.set_description("SumCapcity={:2f}".format(sum_capacity))

        print("Speed = {:d} km/h, Avg_SumCapacity = {:.3f}".format(v, total_sum/valid_cnt))
        sum_capacity_array.append(total_sum/valid_cnt)


    # Save Statistics
    save_dir = "./results/GraphAllocation/different_speed/{:s}/Pmax_{:d}dB".format(AlgorithmName, max_power_dB)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sum_capacity_path = os.path.join(save_dir, "SumCapacity")
    sum_capacity_array = np.array(sum_capacity_array)
    np.savetxt(sum_capacity_path, sum_capacity_array, fmt='%.4f', delimiter='\n')

def run_different_DUE(AlgorithmName="Baseline", v=50, max_power_dB=23):
    print("Run Simulation for Different Number of DUE")
    factors = np.arange(2, 8, 1)
    allocator = GraphAllocator(config)
    allocator.dB_Pd_max = max_power_dB
    allocator.dB_Pc_max = max_power_dB
    allocator.Pd_max = 10 ** (max_power_dB / 10)
    allocator.Pc_max = 10 ** (max_power_dB / 10)
    sum_capacity_array = []
    for factor in factors:
        allocator.numDUE = allocator.numCUE * factor
        total_sum = 0
        pbar = tqdm(range(max_run))
        valid_cnt = 0
        for _ in pbar:
            if AlgorithmName == "Baseline":
                sum_capacity = allocator.run_allocation_baseline(v)
            elif AlgorithmName == "Greedy":
                sum_capacity = allocator.run_allocation_greedy(v, num_iter=5)
            elif AlgorithmName == "Random":
                pass
            else:
                raise NotImplementedError
            if sum_capacity < 0:
                continue
            total_sum += sum_capacity
            valid_cnt += 1
            pbar.set_description("SumCapcity={:2f}".format(sum_capacity))

        print("K/M = {:d}, Avg_SumCapacity = {:.3f}".format(factor, total_sum/valid_cnt))
        sum_capacity_array.append(total_sum/valid_cnt)


    # Save Statistics
    save_dir = "./results/GraphAllocation/different_factor/{:s}/Pmax_{:d}dB".format(AlgorithmName, max_power_dB)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    sum_capacity_path = os.path.join(save_dir, "SumCapacity")
    sum_capacity_array = np.array(sum_capacity_array)
    np.savetxt(sum_capacity_path, sum_capacity_array, fmt='%.4f', delimiter='\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=int, default=50)
    parser.add_argument("-power", type=int, default=23)
    parser.add_argument("-method", type=str, default="speed")
    args = parser.parse_args()
    v = args.v
    method = args.method
    max_power_dB = args.power
    if method == "speed":
        run_different_speed(AlgorithmName="Baseline", max_power_dB=max_power_dB)
    elif method == "factor":
        run_different_DUE(AlgorithmName="Baseline", v=v, max_power_dB=max_power_dB)