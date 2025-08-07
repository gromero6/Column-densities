import os, sys, glob, time, csv
import numpy as np, h5py
from scipy import spatial
import matplotlib.pyplot as plt
from library import *
from los_stats_gabriel_field import seed, m, d

data = np.load(f"thesis_los\ideal\430\DataBundle_MeanCDandBCD_{seed}_{m}_{d}.npz")

print(data.files)
folder_name = "N vs r plots"
file_name = "ColumnDensity_vs_RadialDistance.png"


x_init = data['x_init_points']
mean_CD = data['mean_column_densities']
B_column_density = data["Bcolumndensities"]

final_column_density = mean_CD[-1, :]
radial_distance_pc = np.linalg.norm(x_init, axis=1)
radial_distance_cm = radial_distance_pc * pc_to_cm

plt.figure(figsize=(10, 6))
plt.scatter(radial_distance_cm, final_column_density, s=10, alpha=0.5, color = "blue", label = " mean final column density along LOS")
plt.scatter(radial_distance_cm, B_column_density, s=10, alpha=0.5, color = "red", label= "column density along a B field")

plt.xscale('log')
plt.yscale('log')

plt.xlabel('$cm$', fontsize=14)
plt.ylabel('$cm^{-2}$', fontsize=14)
plt.title('Column Density vs. Distance', fontsize=16)

plt.show()

