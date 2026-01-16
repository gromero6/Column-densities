
import pandas as pd
import numpy as np
import matplotlib as plt
import os, time, sys
import math
from library import *
from stats import *
from load_data import Codedata

if len(sys.argv) > 6:
    case     = sys.argv[1]
    snapshot = sys.argv[2]
    m        = sys.argv[3]
    d        = sys.argv[4]
    cloudnum = sys.argv[5]
    seed     = sys.argv[6]
else:
    case = 'ideal'
    snapshot = '430'
    m = '1000'
    d = '20'
    cloudnum = '3'
    seed = '12345'

data = Codedata(case, snapshot, m, d, cloudnum, seed)


x_init  = data.points
mean_CD = data.mean_column_densities_los
path_CD_heun = data.crs_column
trajectories = data.traj_los

full_number_density_crs = data.number_density_crs
full_pos_crs = data.pos_crs

positions = data.pos_los
numberdens_los = data.number_density_los
cloud_number = data.cloudnum




final_column_density = mean_CD[-1,:]
ratio = final_column_density / path_CD_heun
highest_ratio_index = np.argmax(ratio)

full_densities_heun = full_number_density_crs
full_positions_heun = full_pos_crs
num_steps = 4001
centered_positions_b = np.zeros((4000, m, 3))
centeres_positions_los = np.zeros((d, 4001,3))

point_maxratio = trajectories[:,highest_ratio_index,:]
print(point_maxratio.shape)


reshape_positions = positions.reshape((4001,m,d,3))

positions_points = reshape_positions[:,highest_ratio_index,:,:]
path_positiosn = full_positions_heun[:,highest_ratio_index,:]

print(positions.shape) 
print(reshape_positions.shape)
print(full_densities_heun.shape)

arr_total_distance_heun = np.zeros((num_steps, m))
arr_total_distance_los = np.zeros((d,num_steps))

print("Number of steps along B field lines (Heun): ", num_steps)
print("Number of lines along B field lines (Heun): ", m)
print("shape of full positions (Heun): ", full_positions_heun.shape)
print("shape of full densities (Heun): ", full_densities_heun.shape)
print('position points', positions_points.shape)
print('centered positions los: ' , centeres_positions_los.shape)
plt.figure()
def cent_positions(full_positions_heun, positions_points):
    
    for k in range(m):
        centered_positions_b[:,k,:] = full_positions_heun[:,k,:] - full_positions_heun[0,k,:]

    i = 0
    while i < d:
        centeres_positions_los[i,:,:] = positions_points[:,i,:] - positions_points[0,i,:]
        i += 1
    return centered_positions_b, centeres_positions_los


centered_positions_b, centered_positions_los = cent_positions(full_positions_heun, positions_points)

total_distance_heun = np.zeros(m)
total_distance_los = 0
for i in range(1, 4000):

    step_distance_crs = np.linalg.norm(centered_positions_b[i, :, :] - centered_positions_b[i-1, :, :], axis=1)
    total_distance_heun += step_distance_crs
    arr_total_distance_heun[i, :] = total_distance_heun

    j=0
    while j < d:
        step_distance_los = np.linalg.norm(centered_positions_los[j,i,:] - centered_positions_los[j,i-1,:])
        total_distance_los += step_distance_los
        arr_total_distance_los[j,i] = total_distance_los
        j += 1


arr_total_distance_heun_cm = arr_total_distance_heun * pc_to_cm
arr_total_distance_los_cm = arr_total_distance_los * pc_to_cm

total_distance_interes_point = arr_total_distance_heun_cm[:,highest_ratio_index]
figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), sharey=True)

ax1.set_title(f"number density along B Field Lines (heun integration) - Cloud {int(cloud_number)}")
ax1.set_xlabel("Distance along b FIELD lINE (cm)")
ax1.set_ylabel("Number Density")
ax2.set_title(f"number density along lines of sight - Cloud {int(cloud_number)}")
ax2.set_xlabel("Distance along the LOS (cm)")
ax2.set_ylabel("Number Density")

ax1.grid(True, which="both", ls=":")
ax2.grid(True, which="both", ls=":")

densities_path = full_densities_heun[:,highest_ratio_index]
print('densities_path shape: ', densities_path.shape)
ax1.plot(total_distance_interes_point[:-1], full_densities_heun[:,highest_ratio_index])

line  = 0
print(numberdens_los.shape)
print(arr_total_distance_los_cm.shape)
arr_total_cm_corrected = arr_total_distance_los_cm[:,:-1]
arr_total_nocorrected = arr_total_distance_los[:,:-1]

print(full_densities_heun.shape)
print(full_positions_heun.shape)

print('lenght of arr_total_distance_heun ', arr_total_distance_heun_cm.shape)
print("shape of full densities (Heun): ", full_densities_heun.shape)
print('highest index', highest_ratio_index)
highest_number_density_index = 1
for j in range(2, d):
    max_density_along_los = np.max(numberdens_los[:,highest_ratio_index,j])  # Assuming density is at index 2
    if max_density_along_los > np.max(numberdens_los[:,highest_ratio_index,highest_number_density_index]):
        highest_number_density_index = j
print(f"Highest ratio LOS index: {highest_ratio_index}")
print(f"Highest number density along LOS index: {highest_number_density_index}")
number_dens_los_corrected = numberdens_los[:-1,:,:]
print(number_dens_los_corrected.shape)
#while line < d:
    #ax2.plot(arr_total_cm_corrected[line,:], numberdens_los[:,highest_ratio_index,line])
    #line = line+1
#distance from the intial point form the x init point
distancefromx = arr_total_cm_corrected[highest_number_density_index,2000] - arr_total_cm_corrected[highest_number_density_index,0]
ax2.plot(arr_total_cm_corrected[highest_number_density_index,:],numberdens_los[:,highest_ratio_index,highest_number_density_index])
plt.axvline(x=distancefromx, color='red', linestyle='--', label='point generated')
plt.tight_layout()
plt.show()