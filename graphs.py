import os, sys, glob, time, csv
import numpy as np, h5py
from scipy import spatial
import matplotlib.pyplot as plt
from library import *
from los_stats_gabriel import seed, m, d

data = np.load(f"thesis_los\ideal\430\DataBundle_MeanCD_{seed}_{m}_{d}.npz")

print(data.files)
output_folder = ".\Column-densities\graphs"
file_name = f"ColumnDensity_vs_RadialDistance_{seed}_{m}_{d}.png"


x_init = data['x_init_points']
mean_CD = data['mean_column_densities']

final_column_density = mean_CD[-1, :]
radial_distance_pc = np.linalg.norm(x_init, axis=1)
radial_distance_cm = radial_distance_pc * pc_to_cm

plt.figure(figsize=(10, 6))
plt.scatter(radial_distance_cm, final_column_density, s=10, alpha=0.5, color = "blue", label = " mean final column density along LOS")


plt.xscale('log')
plt.yscale('log')

plt.xlabel('$cm$', fontsize=14)
plt.ylabel('$cm^{-2}$', fontsize=14)
plt.title('Column Density vs. Distance', fontsize=16)

plt.show()

full_path = os.path.join(output_folder, file_name)
plt.savefig(full_path, dpi=300)
print("plot saved to: ", full_path)