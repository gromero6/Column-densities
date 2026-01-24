import numpy as np
import os, sys
import pandas as pd
from load_data import Codedata
from snap_data import snap_data
from library import find_points_and_get_fields, Heun_step, Euler_step
from library import *


case = 'ideal'
snapshot = '430'
cloudnum = 0
seed = '12345'
m_use = 1  # number of points from DataBundle
N_max = 5000  # max steps per direction
DENSITY_THRESHOLD = 100.0 # Define the threshold

integrationmethod = input("euler or heun?").lower()

if integrationmethod == "heun":
    integration = Heun_step
elif integrationmethod == "euler":
    integration = Euler_step

simulation = snap_data(snapshot, case, str(cloudnum))
data = Codedata(case, snapshot, m=1000, d=20, cloudnum=cloudnum, seed=seed) # Generate more points initially
x_all = np.zeros((1,3))
x_all[0] = [-0.23098224497499614261, 0.17261036189460871038, 0.051270637917016421981]

cloud_dir = os.path.join("clouds", f"{case}_clouds.txt")
df = pd.read_csv(cloud_dir)
cloud_center = np.array([
    df.iloc[cloudnum]['CloudCord_X'],
    df.iloc[cloudnum]['CloudCord_Y'],
    df.iloc[cloudnum]['CloudCord_Z']
])

Pos = simulation.Pos - cloud_center
VoronoiPos = simulation.VoronoiPos - cloud_center
Boxsize = simulation.Boxsize
for dim in range(3):
    mask = Pos[:, dim] > Boxsize / 2
    Pos[mask, dim] -= Boxsize
    VoronoiPos[mask, dim] -= Boxsize
    mask = Pos[:, dim] < -Boxsize / 2
    Pos[mask, dim] += Boxsize
    VoronoiPos[mask, dim] += Boxsize


# Initial evaluation for all generated points
b_all, absB_all, n_all_mass, _ = find_points_and_get_fields(x_all, simulation.Bfield, simulation.Density, simulation.Density_grad, Pos, VoronoiPos)
n_all = n_all_mass * gr_cm3_to_nuclei_cm3

# Find points where density is >= 100.0
high_density_mask = (n_all >= DENSITY_THRESHOLD)
x_valid = x_all[high_density_mask]
n_valid = n_all[high_density_mask]
b_valid = b_all[high_density_mask]
absB_valid = absB_all[high_density_mask]

# Select only the required number of points (m_use)
if len(x_valid) < m_use:
    print(f"Warning: Only found {len(x_valid)} seed points with n >= {DENSITY_THRESHOLD}. Using all of them.")
    m_use = len(x_valid)

x_init = x_valid[:m_use].copy()
b0 = b_valid[:m_use].copy()
absB0 = absB_valid[:m_use].copy()
n0 = n_valid[:m_use].copy()

print(f"Starting integration with {m_use} seed points (all n0 >= {DENSITY_THRESHOLD}).")



# Note: m_use is now the filtered number of seed points
pos_fwd = np.full((N_max, m_use, 3), np.nan, dtype=np.float64)
b_fwd = np.full((N_max, m_use, 3), np.nan, dtype=np.float64)
absb_fwd = np.full((N_max, m_use), np.nan, dtype=np.float64)
n_fwd = np.full((N_max, m_use), np.nan, dtype=np.float64)

current = x_init.copy()
mask_fwd = np.ones(m_use, dtype=bool)

# Initial evaluation (already done, just storing the initial filtered points)
pos_fwd[0, mask_fwd] = current[mask_fwd]
b_fwd[0, mask_fwd] = b0[mask_fwd]
absb_fwd[0, mask_fwd] = absB0[mask_fwd]
n_fwd[0, mask_fwd] = n0[mask_fwd]

# Forward integration
for step in range(1, N_max):
    if not np.any(mask_fwd):
        step_fwd = step
        break
    
    # Get only the points that are still active
    active_indices = np.where(mask_fwd)[0]
    active_pos = current[active_indices]

    # Heun step: moves the point
    next_pos, _, n_new_mass, _ = integration(
        active_pos, np.ones(len(active_pos)), simulation.Bfield, simulation.Density,
        simulation.Density_grad, Pos, VoronoiPos, simulation.Volume
    )
    # Get fields at the new position
    b_vec, absB, n_new_mass, _ = find_points_and_get_fields(next_pos, simulation.Bfield, simulation.Density, simulation.Density_grad, Pos, VoronoiPos)
    n_new = n_new_mass * gr_cm3_to_nuclei_cm3



    save_mask_active = (n_new >= DENSITY_THRESHOLD)
    
 
    indices_to_save = active_indices[save_mask_active]


    pos_fwd[step, indices_to_save] = next_pos[save_mask_active]
    b_fwd[step, indices_to_save] = b_vec[save_mask_active]
    absb_fwd[step, indices_to_save] = absB[save_mask_active]
    n_fwd[step, indices_to_save] = n_new[save_mask_active]


    current[indices_to_save] = next_pos[save_mask_active]
    
    
    new_mask = np.zeros_like(mask_fwd, dtype=bool)
    new_mask[active_indices] = save_mask_active # Set to True only if n_new >= 100.0
    mask_fwd = new_mask
else:
    step_fwd = N_max + 1 # Use N_max + 1 to simplify slicing later


pos_bck = np.full((N_max, m_use, 3), np.nan, dtype=np.float64)
b_bck = np.full((N_max, m_use, 3), np.nan, dtype=np.float64)
absb_bck = np.full((N_max, m_use), np.nan, dtype=np.float64)
n_bck = np.full((N_max, m_use), np.nan, dtype=np.float64)

current = x_init.copy()
mask_bck = np.ones(m_use, dtype=bool)


pos_bck[0, mask_bck] = current[mask_bck]
b_bck[0, mask_bck] = b0[mask_bck]
absb_bck[0, mask_bck] = absB0[mask_bck]
n_bck[0, mask_bck] = n0[mask_bck]

for step in range(1, N_max):
    if not np.any(mask_bck):
        step_bck = step
        break
    active_indices = np.where(mask_bck)[0]
    active_pos = current[active_indices]


    next_pos, _, n_new_mass, _ = integration(
        active_pos, -np.ones(len(active_pos)), simulation.Bfield, simulation.Density,
        simulation.Density_grad, Pos, VoronoiPos, simulation.Volume
    )
    b_vec, absB, n_new_mass, _ = find_points_and_get_fields(next_pos, simulation.Bfield, simulation.Density, simulation.Density_grad, Pos, VoronoiPos)
    n_new = n_new_mass * gr_cm3_to_nuclei_cm3


    save_mask_active = (n_new >= DENSITY_THRESHOLD)
    indices_to_save = active_indices[save_mask_active]

    pos_bck[step, indices_to_save] = next_pos[save_mask_active]
    b_bck[step, indices_to_save] = b_vec[save_mask_active]
    absb_bck[step, indices_to_save] = absB[save_mask_active]
    n_bck[step, indices_to_save] = n_new[save_mask_active]


    current[indices_to_save] = next_pos[save_mask_active]
    

    new_mask = np.zeros_like(mask_bck, dtype=bool)
    new_mask[active_indices] = save_mask_active 
    mask_bck = new_mask
else:
    step_bck = N_max + 1



slice_fwd = min(step_fwd, N_max) 
slice_bck = min(step_bck, N_max)

pos_fwd = pos_fwd[:slice_fwd]
pos_bck = pos_bck[:slice_bck]
b_fwd = b_fwd[:slice_fwd]
b_bck = b_bck[:slice_bck]
absb_fwd = absb_fwd[:slice_fwd]
absb_bck = absb_bck[:slice_bck]
n_fwd = n_fwd[:slice_fwd]
n_bck = n_bck[:slice_bck]

pos_full = np.concatenate([pos_bck[::-1], pos_fwd[1:]], axis=0)
b_full = np.concatenate([b_bck[::-1], b_fwd[1:]], axis=0)
absb_full = np.concatenate([absb_bck[::-1], absb_fwd[1:]], axis=0)
n_full = np.concatenate([n_bck[::-1], n_fwd[1:]], axis=0)

# Compute cumulative distance (in cm)
dist_full = np.zeros_like(n_full)
for i in range(m_use):
    for j in range(1, dist_full.shape[0]):
        # Check for NaN to ensure the point was saved
        if np.all(~np.isnan(pos_full[j, i])):
            step = np.linalg.norm(pos_full[j, i] - pos_full[j-1, i])
            dist_full[j, i] = dist_full[j-1, i] + step
        else:
            break # Stop distance accumulation if NaN is encountered


output_dir = os.path.join("thesis_los", case, snapshot)
os.makedirs(output_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(output_dir, f"FieldLines_cloud{cloudnum}_m{m_use}_seed{seed}_{integrationmethod}.npz"),
    seed_points=x_init,
    positions=pos_full,
    B_fields=b_full,
    absB=absb_full,
    number_density=n_full,
    distance_cm=dist_full
)

print(f"Saved {m_use} field lines. Full array shape: {pos_full.shape}")