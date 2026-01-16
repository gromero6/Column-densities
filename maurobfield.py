import numpy as np
import sys, math
import os
import pandas as pd
from load_data import Codedata
from snap_data import snap_data
from library import find_points_and_get_fields, Heun_step
from library import *

case = 'ideal'
snapshot = '430'
cloudnum = '0'
seed = '12345'


simulation = snap_data(snapshot, case, cloudnum)
data = Codedata(case, snapshot, m=1000, d=20, cloudnum=0, seed=12345)

Pos_abs = simulation.Pos.copy()
VoronoiPos_abs = simulation.VoronoiPos.copy()
Bfield = simulation.Bfield
Density = simulation.Density
Density_grad = simulation.Density_grad
Volume = simulation.Volume
Boxsize = simulation.Boxsize


cloud_dir = os.path.join("clouds", f"{case}_clouds.txt")
df = pd.read_csv(cloud_dir)
cloud_center = np.array([
    df.iloc[0]['CloudCord_X'],
    df.iloc[0]['CloudCord_Y'],
    df.iloc[0]['CloudCord_Z']
])


Pos = Pos_abs - cloud_center
VoronoiPos = VoronoiPos_abs - cloud_center
for dim in range(3):
    mask = Pos[:, dim] > Boxsize / 2
    Pos[mask, dim] -= Boxsize
    VoronoiPos[mask, dim] -= Boxsize
    mask = Pos[:, dim] < -Boxsize / 2
    Pos[mask, dim] += Boxsize
    VoronoiPos[mask, dim] += Boxsize


x0 = np.array([
    -0.23098224497499614261,
     0.17261036189460871038,
     0.051270637917016421981
])
x = x0[np.newaxis, :]  # Shape (1,3)


b0, absB0, n0, _ = find_points_and_get_fields(x, Bfield, Density, Density_grad, Pos, VoronoiPos)
threshold = 100.0
n0 *= gr_cm3_to_nuclei_cm3

pos_fwd = [x.copy()]  # list of (1,3) arrays
b_fwd   = [b0.copy()]
absb_fwd = [absB0.copy()]
n_fwd   = [n0.copy()]

current = x.copy()
n_current = n0.copy()

while n_current[0] >= threshold:
    next_pos, _, n_new, _ = Heun_step(
        current, np.array([0.5]), Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
    )
    b_vec, absB, n_new, _ = find_points_and_get_fields(
        next_pos, Bfield, Density, Density_grad, Pos, VoronoiPos
    )
    n_new *= gr_cm3_to_nuclei_cm3
    pos_fwd.append(next_pos.copy())
    b_fwd.append(b_vec.copy())
    n_fwd.append(n_new.copy())
    absb_fwd.append(absB.copy())
    current = next_pos
    n_current = n_new


pos_bck = [x.copy()]
b_bck   = [b0.copy()]
absb_bck = [absB0.copy()]
n_bck   = [n0.copy()]

current = x.copy()
n_current = n0.copy()

while n_current[0] >= threshold:
    next_pos, _, n_new, _ = Heun_step(
        current, np.array([-0.5]), Bfield, Density, Density_grad, Pos, VoronoiPos, Volume
    )
    b_vec, absB, n_new, _ = find_points_and_get_fields(
        next_pos, Bfield, Density, Density_grad, Pos, VoronoiPos
    )
    n_new *= gr_cm3_to_nuclei_cm3
    pos_bck.append(next_pos.copy())
    b_bck.append(b_vec.copy())
    absb_bck.append(absB)
    n_bck.append(n_new.copy())
    current = next_pos
    n_current = n_new


#
pos_bck_arr = np.vstack(pos_bck[::-1])    
pos_fwd_arr = np.vstack(pos_fwd[1:])      
full_path = np.vstack([pos_bck_arr, pos_fwd_arr])

b_bck_arr = np.vstack(b_bck[::-1])
b_fwd_arr = np.vstack(b_fwd[1:])
full_B = np.vstack([b_bck_arr, b_fwd_arr])

absb_bck_arr = np.hstack(absb_bck[::-1])
absb_fwd_arr = np.hstack(absb_fwd[1:])
full_absB = np.hstack([absb_bck_arr, absb_fwd_arr])

n_bck_arr = np.hstack(n_bck[::-1])
n_fwd_arr = np.hstack(n_fwd[1:])
full_n = np.hstack([n_bck_arr, n_fwd_arr])

print(f"Field line traced with {full_path.shape[0]} points.")

distance = 0
arr_distance = np.zeros(full_path.shape[0])
for j in range(1,full_path.shape[0]):
    step = np.linalg.norm(full_path[j,:]-full_path[j-1,:])
    distance += step
    arr_distance[j-1] = distance

print(arr_distance)

figure = plt.figure(figsize=(12, 8))

B_field_gauss = full_absB*velocity_unit*math.sqrt((4*np.pi*(mass_unit)) / np.power(length_unit,3))
B_fields_micro = B_field_gauss*np.power(10,6)

plt.xlabel("Distance the field line travels (cm)")
plt.ylabel('B field (microgauss)')

print(full_path.shape)
#distance from x to one tip:
distx = np.linalg.norm(x0 - full_path[0,:])
# Optional: mask out points where density < 100
valid_mask = full_n >= 100
if np.any(valid_mask):
    distance_valid = arr_distance[valid_mask]
    B_valid = B_fields_micro[valid_mask]
    path_filtered = full_path[valid_mask]
else:
    distance_valid = arr_distance
    B_valid = B_fields_micro
    #distance from x to one tip:
distx = np.linalg.norm(x0 - path_filtered[0,:])*pc_to_cm
plt.grid(True, which='both', ls=':')
plt.axvline(distx, color='r', linestyle='--', linewidth=2, label='point')
plt.plot(distance_valid*pc_to_cm, B_valid)
plt.show()

def crs_path(simulation, data):

    x = data.points
    
    
    