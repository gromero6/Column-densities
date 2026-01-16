import numpy as np
import matplotlib as plt
import os, time, sys
from load_data import Codedata
from library import *
from stats import *

if len(sys.argv) > 6:
    case     = str(sys.argv[1])
    snapshot = str(sys.argv[2])
    m        = str(sys.argv[3])
    d        = str(sys.argv[4])
    cloudnum = str(sys.argv[5])
    seed     = str(sys.argv[6])
else:
    case = 'ideal'
    snapshot = '430'
    m = '1000'
    d = '20'
    cloudnum = '0'
    seed = '12345'

data = Codedata(case, snapshot, m, d, cloudnum, seed)


cloud_center = np.array([data.xcore, data.ycore, data.zcore])
Pos_copy = Pos.copy()
VoronoiPos_copy = VoronoiPos.copy()

Pos_copy -= cloud_center
VoronoiPos_copy -= cloud_center

for dim in range(3):  # Loop over x, y, z
    pos_from_center = Pos_copy[:, dim]
    boundary_mask = pos_from_center > Boxsize / 2
    Pos_copy[boundary_mask, dim] -= Boxsize
    VoronoiPos_copy[boundary_mask, dim] -= Boxsize
    
    boundary_mask = pos_from_center < -Boxsize / 2
    Pos_copy[boundary_mask, dim] += Boxsize
    VoronoiPos_copy[boundary_mask, dim] += Boxsize


tree = KDTree(Pos_copy)

mean_CD = data.mean_column_densities_los
trajectories = data.distance_los
path_CD_heun = data.crs_column
positions = data.pos_los
full_densities_los = data.number_density_los
x_init = data.points
print("shape of trajectories: ", trajectories.shape)
print('Shape of the mean column densities', mean_CD.shape)
#find point with the highest ratio of NLOS vs NBpath

final_column_density = mean_CD[-1, :]
print('shape of inal_N', final_column_density.shape)
ratio = final_column_density /path_CD_heun
print(ratio.shape)

highest_ratio_index = np.argmax(ratio)

point_maxratio = trajectories[:,highest_ratio_index,:].copy()
print(point_maxratio.shape)

reshape_positions = positions.reshape((4001,1000,20,3))



    # final_column_density = mean_CD[-1,:]
#ratio = final_column_density / path_CD_heun
#highest_ratio_index = np.argmax(ratio)
d = int(d)
#check the line of sight with the highest number density on its path
highest_number_density_index = 1
for j in range(2, d):
    max_density_along_los = np.max(full_densities_los[:,highest_ratio_index,j])  # Assuming density is at index 2
    if max_density_along_los > np.max(full_densities_los[:,highest_ratio_index,highest_number_density_index]):
        highest_number_density_index = j
print(f"Highest ratio LOS index: {highest_ratio_index}")
print(f"Highest number density along LOS index: {highest_number_density_index}")

magnitudes = np.linalg.norm(reshape_positions[:, highest_ratio_index, highest_number_density_index, :], axis=1)
print("Magnitudes around index 2000-2055:", magnitudes[1995:2055])
non_zero_indices = np.where(magnitudes > 0)[0]

if non_zero_indices.size > 0:
    last_nonzero_index = non_zero_indices[-1]
else:
    last_nonzero_index = 2000
# Get the x_init point for this LOS (starting point)
r_last = reshape_positions[last_nonzero_index+1,highest_ratio_index,highest_number_density_index,:]
x_init_point = x_init[highest_ratio_index]  # Already cloud-centered
vectorlos = r_last - x_init_point
los_length = np.linalg.norm(vectorlos)
unit_los = vectorlos / los_length 

print(x_init_point)
print(f"x_init point (cloud-centered): {x_init_point}")
print(f"LOS direction: {unit_los}")
print(f"LOS length: {los_length:.3f} pc")



#rotation matrix implementation

#new z axis
newz_nonorm = [-unit_los[1],unit_los[0],0]
newz = newz_nonorm / np.linalg.norm(newz_nonorm)
newx = unit_los
newy = np.cross(newx,newz)

R_forward = np.array([
    newx,
    newy,
    newz
])

# Use LOS length 
grid_size_pc = max(los_length * 2, 2)  # At least 10 pc, or 2x LOS length for better coverage
grid_size_pc = min(grid_size_pc, 10.0)  # Cap at 30 pc to avoid excessive computation

Grid_resolution = 5000


print(f"LOS length: {los_length:.3f} pc")


xprime1d = np.linspace(-grid_size_pc,grid_size_pc,Grid_resolution)
yprime1d = np.linspace(-grid_size_pc, grid_size_pc, Grid_resolution)

#create the mesh 2d
XPRIME, YPRIME = np.meshgrid(xprime1d, yprime1d)
ZPRIME = np.zeros_like(XPRIME)

RPRIME = np.vstack([
    XPRIME.flatten(),
    YPRIME.flatten(),
    ZPRIME.flatten()
])

inverse_rotation = R_forward.T

R_original = inverse_rotation @ RPRIME

XORIGINAL = R_original[0,:] + x_init_point[0]
YORIGINAL = R_original[1,:] + x_init_point[1]
ZORIGINAL = R_original[2,:] + x_init_point[2]

total_points_original = np.vstack([XORIGINAL,YORIGINAL,ZORIGINAL]).T

distances, indices = tree.query(total_points_original)

# Get densities at query points (pc)
_, _, dens, _ = find_points_and_get_fields(
    total_points_original, Bfield, Density, Density_grad, Pos_copy, VoronoiPos_copy
)

print(f"Raw Density Range: [{dens.min():.3e}, {dens.max():.3e}] g/cm³")
dens *= gr_cm3_to_nuclei_cm3 
print(f"Scaled Density Range: [{dens.min():.3e}, {dens.max():.3e}] cm⁻³")

dens_grid = dens.reshape(Grid_resolution, Grid_resolution)

masked_data = ma.masked_where(dens_grid<100, dens_grid)



# For pcolormesh, we need to handle the grid properly
# Create coordinate arrays for pcolormesh (one element larger)
A_edges = np.linspace(-grid_size_pc, grid_size_pc, Grid_resolution + 1)
B_edges = np.linspace(-grid_size_pc, grid_size_pc, Grid_resolution + 1)
A_grid_edges, B_grid_edges = np.meshgrid(A_edges, B_edges)

cmap = cm.viridis.copy()
cmap.set_bad(color='white', alpha=0)



interest_line = reshape_positions[:,highest_ratio_index,highest_number_density_index,:].copy()
#convert line coordinates to the plane coordinates
centered_interest_line = interest_line - x_init_point
plane_coords = R_forward @ centered_interest_line.T

# Plotting
figure = plt.figure(figsize=(10, 8))


im = plt.pcolormesh(XPRIME, YPRIME, np.log10(masked_data), shading='auto', cmap=cmap)
plt.colorbar(im, label='log10(Number Density (cm⁻³))')

# Add contour lines at specified densities
#if len(valid_levels) > 0:
#    contours = plt.contour(A_grid, B_grid, dens_grid, levels=valid_levels, 
#                          colors='white', linewidths=1.0, alpha=0.9)
#   plt.clabel(contours, inline=True, fontsize=11, fmt='%d cm⁻³')
# Add contour lines at specified densities
#vectorlos in plane coordinates
vectorlos_plane = R_forward @ vectorlos

# Mark the x_init point (which is at the origin of the plane coordinates)
plt.plot(0.0, 0.0, 'ro', markersize=10, label='x_init point', zorder=5, markeredgecolor='white', markeredgewidth=2)
# Plot the line of sight path
plt.plot(vectorlos_plane[0], vectorlos_plane[1], label='line of sight', linewidth=2, zorder=10)


plt.xlabel('x (pc)', fontsize=12)
plt.ylabel('y (pc)', fontsize=12)
plt.title(f'Number Density Map on Plane - Cloud {int(data.cloudnum)}', fontsize=14)
plt.legend(loc='best', fontsize=11)
plt.axis('equal')  # Make aspect ratio equal for better visualization
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()
