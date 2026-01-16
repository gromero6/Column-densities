import numpy as np
import matplotlib as plt
import os, time, sys
from library import *
import pandas as pd
from scipy import spatial
import h5py
from load_data import Codedata
import math



if len(sys.argv) > 6:
    case     = sys.argv[1]
    snapshot = sys.argv[2]
    cloudnum = sys.argv[5]
    seed     = sys.argv[6]
else:
    case = 'ideal'
    snapshot = '430'
    m = '1000'
    d = '20'
    cloudnum = '3'
    seed = '12345'

codedata = Codedata(case, snapshot, m=1000, d=20, cloudnum, seed=12345)


output_folder = "./Column-densities/graphs/Bfield_n"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_name = f"Bfield_{snapshot}_{m}_{d}_{cloudnum}.png"

print(codedata.abs_fields_crs.shape)
lineindex = int(input('which line want to compute?')) # Select the first line of sight for visualization
#convert to Gauss from CGS
B_field_gauss = codedata.abs_fields_crs[:, lineindex]*velocity_unit*math.sqrt((4*np.pi*(mass_unit)) / np.power(length_unit,3))
B_fields_micro = B_field_gauss*np.power(10,6)
print("pos_crs shape: ", codedata.pos_crs.shape)
crs_line = codedata.pos_crs[:,lineindex,:]

total = 0
crs_distance = np.zeros(4000)
for i in range(1, 4000):
    step = np.linalg.norm(crs_line[i,:]-crs_line[i-1,:])
    total += step
    crs_distance[i-1] = total

crs_distance_cm = crs_distance*pc_to_cm
figure = plt.figure(figsize=(12, 8))

plt.xlabel("Distance the field line travels (cm)")
plt.ylabel('B field (microgauss)')


crs_filteres = crs_distance_cm[B_fields_micro != 0]
B_filtered = B_fields_micro[B_fields_micro != 0]
B_cgs = codedata.abs_fields_crs[:,lineindex].copy()
Bcgs = B_cgs[B_fields_micro != 0]

plt.grid(True, which='both', ls=':')
plt.plot(crs_filteres,Bcgs)
plt.plot(crs_filteres,B_filtered)
plt.show()

#-7.12736735e+17,  5.32619924e+17,  1.58204658e+17
#-0.23098224497499614261, 0.17261036189460871038, 0.051270637917016421981



#generate a field line for that specific point
N = 2000
traj_fwd = np.zeros(1)
traj_bck = np.zeros(1)
pos_fwd = np.zeros(1,3)
pos_bck = np.zeros(1,3)
b_fwd = np.zeros(1,3)
b_bck = np.zeros(1,3)
absB_fwd = np.zeros(1)
absB_bck = np.zeros(1)
n_fwd = np.zeros(100)
n_bck = np.zeros(100)
threashold = 100


    

