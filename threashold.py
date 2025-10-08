import os, time, csv, glob, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from library import *

start_time = time.time()


N          = int(sys.argv[1])
case       = str(sys.argv[2])
snap       = int(sys.argv[3])
cloud      = int(sys.argv[4])
max_cycles = int(sys.argv[5])
nd         = int(sys.argv[6])
seed       = int(sys.argv[7])

#path of the data:

datapath = os.path.join("thesis_los", str(case), str(snap))
data_name = f"DataBundle_MeanCD_andpathD_{seed}_{max_cycles}_{nd}_{cloud}.npz"
full_path = os.path.join(datapath, data_name)

if not os.path.exists(full_path):
    print(f"Error: data not found at: {full_path}")
    sys.exit(1)

#load the data:

data = np.load(full_path)

numb_densities = data["densities"]
mean_CD        = data["mean_column_densities"]
path_CD        = data["pathcolumn"]
x_init         = data["x_init_points"]
full_column  = data["full_columns"]
traj           = data["trajectories"]

N_steps = numb_densities.shape[0]
numb_densities_reshaped = numb_densities.reshape(N_steps, max_cycles, nd)
final_column_density = mean_CD[-1, :]

ratioN = np.divide(final_column_density, path_CD)
densthresh = 100
#find top five points:
top5indices = np.argsort(ratioN)[::-1][:5]


for point_idx in top5indices:

    # Use the 3D array 'numb_densities_reshaped'
    point_numdensities = numb_densities_reshaped[:, point_idx, :]
    
    # Get the ratio for printing
    current_ratio = ratioN[point_idx]
    sub_x = x_init[point_idx]

    for dir_idx in range(nd):
        # Line of sight density trajectory (N_steps long)
        line_densities = point_numdensities[:, dir_idx]
        
        # Find all steps where the density is non-zero
        non_zero_indices = np.where(line_densities > 0)[0]
        
        # --- Density Value Determination ---
        if len(non_zero_indices) == 0:
            # Case 1: The entire trajectory is zero (point was outside the cloud core)
            d_init_active = 0.00
            d_final_active = 0.00
        else:
            # Case 2: The ray was active for at least one step
            
            # The 'first' active density (the true density at the starting point)
            d_init_active = line_densities[non_zero_indices[0]] 
            
            # The 'last' active density (the density at the final position)
            # This is the density at the step where the mask failed, or at N_steps-1
            d_final_active = line_densities[non_zero_indices[-1]]
        
        # --- Check Logic ---
        
        # 1. Initial density check: Should be >= densthresh 
        # (Based on the selection method, d_init_active should be > 100 if the point selection worked)
        init_status = "OK (First Active >= Th)" if d_init_active >= densthresh else "ERROR (First Active < Th)"

        # 2. Final density check: Must be < densthresh for a successful exit
        if d_init_active < densthresh:
            final_status = "N/A (Init < Th)"
        elif d_final_active < densthresh:
            final_status = "PASSED (Final < Th)"
        else:
            # The last recorded density was still >= threshold. The line reached N_steps without exiting.
            final_status = "FAILED (Final >= Th)" 

        print(f"P {point_idx:03d} ({current_ratio:.2e}) | D {dir_idx:02d} | Init: {d_init_active:^8.2f} | Final: {d_final_active:^8.2f} | Final Status: {final_status:^20} | Init Status: {init_status}")
