import os, sys, glob, time, csv
import numpy as np, h5py
from scipy import spatial
import matplotlib.pyplot as plt
from library import *

seed  = input("Enter seed: ")
m     = input("Enter number of points: ")
d     = input("Enter Number of lines of sight per point: ")
index = input("Enter index of the file to be read: ")
case  = input("Enter case (ideal/amb): ")
snap  = input("Enter snapshot number: ")

data_directory = os.path.join("thesis_los", case, snap)
data_name = f"DataBundle_MeanCD_{seed}_{m}_{d}_{index}.npz"
full_data_path = os.path.join(data_directory, data_name)
data = np.load(full_data_path)


print(data.files)
output_folder = ".\graphs"
png_name = f"ColumnDensity_vs_RadialDistance_{seed}_{m}_{d}_{index}.png"


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

full_path = os.path.join(output_folder, png_name)
plt.savefig(full_path, dpi=300)
print("plot saved to: ", full_path)

plt.show()

